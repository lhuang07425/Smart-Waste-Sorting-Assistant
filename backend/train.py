"""Fine-tune a ResNet18 model on the TrashNet dataset."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "trashnet" / "data" / "dataset-resized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backend" / "models"


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    epochs: int = 8
    batch_size: int = 32
    val_split: float = 0.2
    lr: float = 1e-4
    seed: int = 42


CATEGORY_MAP: Dict[str, str] = {
    "cardboard": "recycle",
    "glass": "recycle",
    "metal": "recycle",
    "paper": "recycle",
    "plastic": "recycle",
    "trash": "trash",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train TrashNet classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the TrashNet dataset (ImageFolder structure)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store model weights and metadata",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    return TrainConfig(
        data_dir=resolve_path(args.data_dir),
        output_dir=resolve_path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        lr=args.lr,
        seed=args.seed,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# rest of file unchanged

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_datasets(cfg: TrainConfig) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {cfg.data_dir}")

    full_dataset = datasets.ImageFolder(cfg.data_dir, transform=TRAIN_TRANSFORMS)
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_dataset.dataset.transform = VAL_TRANSFORMS
    return train_dataset, val_dataset


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = create_datasets(cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.dataset.classes
    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    best_val_accuracy = 0.0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cfg.output_dir / "trashnet_resnet18.pth"
    meta_path = cfg.output_dir / "trashnet_meta.json"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} - train", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} - val", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        tqdm.write(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), weights_path)
            meta = {
                "class_names": class_names,
                "category_map": CATEGORY_MAP,
                "best_val_accuracy": best_val_accuracy,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            tqdm.write(f"Saved new best model with val_acc={val_acc:.3f}")

    tqdm.write(f"Training complete. Best val accuracy: {best_val_accuracy:.3f}")
    if not weights_path.exists():
        torch.save(model.state_dict(), weights_path)
        meta = {
            "class_names": class_names,
            "category_map": CATEGORY_MAP,
            "best_val_accuracy": best_val_accuracy,
        }
        meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    config = parse_args()
    train(config)
