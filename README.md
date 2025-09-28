# Smart Waste Sorting API

This FastAPI service powers the Smart Waste Sorting Assistant frontend. It now fine-tunes a ResNet18 model on both the TrashNet image dataset and cropped instances from the TACO (Trash Annotations in Context) dataset so the classifier can tell plastic film and wrappers apart from clean paper and cardboard.

## Setup

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r backend/requirements.txt
```

### Prepare the datasets

1. **TrashNet** - already checked into `trashnet/data/dataset-resized` in this repo. No extra work is needed.
2. **TACO (COCO format)** - download the release from [tacodataset.org](http://tacodataset.org/) or the GitHub release `COCO_format.zip`, extract it under `./taco`, and make sure the annotation JSON files and `batch_x/*.JPG` images stay in the same relative layout. By default the trainer looks for annotation files under `taco/` and uses the `train` split.

### Train (or re-train) the model

```powershell
python backend/train.py --epochs 4 --batch-size 32 --taco-root taco --taco-split train
```

Key switches:
- `--taco-root` lets you point at another extraction directory.
- `--taco-min-area` filters tiny segmentation masks (default 3200 px^2).
- `--taco-max-per-class` caps the number of cropped TACO samples per canonical class (set <=0 to disable).
- `--no-taco` falls back to the legacy TrashNet-only training routine.

The script writes updated assets into `backend/models/`:
- `trashnet_resnet18.pth` - weights for the fine-tuned backbone
- `trashnet_meta.json` - class list, category mapping, validation accuracy, and dataset provenance

### Run the API

```powershell
uvicorn backend.app:app --reload
```

CORS is enabled for local development so the static frontend can call the API from `http://localhost` or when opened directly via `file://`.

## Request format

Send a `multipart/form-data` POST to `/classify` with the file field named `file`:

```powershell
curl -X POST http://localhost:8000/classify \
  -F "file=@trashnet/data/dataset-resized/plastic/plastic1.jpg"
```

Example response:

```json
{
  "category": "recycle",
  "emoji": "recycle",
  "color": "#0ea5e9",
  "confidence": 0.78,
  "reason": "Model predicted 'glass', routing to recycle",
  "predictions": [
    {"label": "glass", "probability": 0.78},
    {"label": "plastic", "probability": 0.18},
    {"label": "metal", "probability": 0.02}
  ]
}
```

The frontend consumes `category`, `emoji`, `color`, `confidence`, and `reason` to render the results UI.
