# Smart Waste Sorting API

This FastAPI service powers the Smart Waste Sorting Assistant frontend. It loads a ResNet18 model fine-tuned on the TrashNet dataset to predict which bin an item belongs in.

## Setup

`powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r backend/requirements.txt
`

### Train the model (once)

`powershell
python backend/train.py --epochs 2 --batch-size 32
`

The command downloads the TrashNet dataset (already cloned in 	rashnet/), fine-tunes the classifier, and writes:

- ackend/models/trashnet_resnet18.pth – model weights
- ackend/models/trashnet_meta.json – label metadata and validation accuracy

### Run the API

`powershell
uvicorn backend.app:app --reload
`

The API enables CORS for local development so the static frontend can call it from http://localhost or via ile:// when opened directly.

## Request format

Send a multipart/form-data POST to /classify with the field named ile:

`powershell
curl -X POST http://localhost:8000/classify \
  -F "file=@trashnet/data/dataset-resized/plastic/plastic1.jpg"
`

Example response:

`json
{
  "category": "recycle",
  "emoji": "♻️",
  "color": "#0ea5e9",
  "confidence": 0.78,
  "reason": "Model predicted 'glass'",
  "predictions": [
    {"label": "glass", "probability": 0.78},
    {"label": "plastic", "probability": 0.18},
    {"label": "metal", "probability": 0.02}
  ]
}
`

The frontend consumes category, emoji, color, confidence, and eason to render the result UI.
