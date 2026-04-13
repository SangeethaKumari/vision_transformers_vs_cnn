# Vision Transformers vs CNN: Tree Classification

## Overview
This project contrasts a Vision Transformer (ViT) and a ResNet-50 CNN on a multi-class tree identification dataset. The `src/svlearn_vit_cnn/transfer_learning/trees_classification.py` script performs head-only transfer learning with Hugging Face Transformers, while `docs/notebooks/trees_classification.ipynb` collects qualitative and quantitative evaluations (predictions, ROC curves, timing, and metrics saved during training). The accompanying `evaluation_results.csv` extends the comparison to additional public datasets for quick reference.

## Repository Layout
- `config.yaml` – toggles the experiment (`current_task`) and records dataset/model paths.
- `src/svlearn_vit_cnn/transfer_learning/trees_classification.py` – main training loop shared by ViT and ResNet runs.
- `src/svlearn_vit_cnn/utils/train_utils.py` & `dataset_tools/` – preprocessing, transforms, ROC plotting, and device helpers.
- `docs/notebooks/trees_classification.ipynb` – post-training analysis notebook (requires trained checkpoints).
- `evaluation_results.csv` – summary table of accuracy vs. inference speed across datasets.

## Environment Setup
1. Install the uv virtual environment:
   ```
   uv sync
   ```
   This pulls in `transformers[torch]`, `datasets`, `evaluate`, and the `svlearn-bootcamp` helpers referenced throughout the codebase.

## Dataset Preparation
1. Download the tree dataset (from the course portal) and arrange it as class-specific folders:
   ```
   /Users/<you>/data/trees/
     ├── Oak/
     └── WeepingWillow/
   ```
2. Update `config.yaml`:
   - `tree-dataset.path`: absolute path to the directory that contains the `trees` folder.
   - `cnn.results` / `vision-transformer.results`: writable directories where checkpoints, metrics, and plots will be stored for each model.
3. The preprocessing pipeline (`dataset_tools/preprocess.py`) automatically:
   - Enumerates every image path under `trees/<class_name>`.
   - Label-encodes classes, shuffles, and performs an 80/20 split.
   - Writes `train.json` and `validation.json` into `preprocessed` directories when invoked from the training script.

## Training the Models
The same training script handles both architectures; change `config['current_task']` to switch models.

1. Open `config.yaml` and set:
   - `current_task: vit_classification` for the ViT run (defaults to `google/vit-base-patch16-224-in21k`).
   - `current_task: resnet_classification` for the ResNet-50 run (`microsoft/resnet-50`).
2. Launch training:
   ```
   uv run src/svlearn_vit_cnn/transfer_learning/trees_classification.py
   ```
3. What the script does:
   - Detects the best device via `get_device()` (CUDA, Apple Silicon/MPS, or CPU). Mixed precision (fp16/bf16) is enabled when available.
   - Builds augmentation-heavy train transforms (random rotation, jitter, horizontal flip, resized crop) and deterministic eval transforms using the appropriate Hugging Face processor.
   - Preprocesses the dataset, persists the fitted `LabelEncoder` as `label_encoder.joblib`, and wraps the Pandas splits with `datasets.Dataset`.
   - Loads the chosen backbone, freezes all layers, unfreezes only `model.classifier`, and prints trainable parameter counts.
   - Trains for up to 50 epochs with `TrainingArguments` (batch size 16, LR 2e-4, step-based eval/save cadence of 500, `load_best_model_at_end=True`).
   - Saves artifacts under the configured `results` directory:
     - `pytorch_model.bin`, tokenizer/processor config, and `trainer_state.json`.
     - `train_results.json`, `eval_results.json`, and TensorBoard-friendly logs.
     - `roc.png` generated from `compute_metrics`, plus confusion-metric JSON.

**Tip:** Run the script twice (once per `current_task`). The evaluation notebook expects both result directories to exist and contain the saved label encoders and metrics.

## Evaluation Notebook (`docs/notebooks/trees_classification.ipynb`)
1. Ensure both training passes completed; the notebook will load checkpoints from the directories defined in `config.yaml`.
2. Open the notebook (VS Code, JupyterLab, or `uvx jupyter lab`). The first cell imports shared helpers via `%run supportvectors-common.ipynb`.
3. Notebook capabilities:
   - Loads both ViT and ResNet models in evaluation mode, including the saved `LabelEncoder` to convert logits back to class names.
   - Demonstrates single-image inference on an unseen validation photo (provided path: `<tree-dataset.path>/test_oak_tree.webp`).
   - Reads `eval_results.json` and `train_results.json` for each model to display accuracy, precision, recall, F1, and training-time metadata.
   - Shows saved ROC curves (`roc.png`) from both experiments.
   - Provides hooks for further qualitative inspection (e.g., overlay attention maps if you extend the notebook).
4. Re-run cells after toggling `current_task` and retraining if you change hyperparameters or dataset composition.

## Metrics Snapshot
`evaluation_results.csv` captures aggregate results (accuracy shown as a percentage; inference time as seconds per image):

| Dataset | Model | Accuracy (%) | Inference Time (s/image) |
| --- | --- | --- | --- |
| cifar-10 | ViT | 8.94 | 0.0020 |
| cifar-10 | ResNet | 12.72 | 0.0017 |
| oxford_flowers | ViT | 0.016 | 0.0015 |
| oxford_flowers | ResNet | 0.016 | 0.0019 |
| trees | ViT | 45.69 | 0.0018 |
| trees | ResNet | 48.68 | 0.0025 |

Use these as a baseline; rerunning training with different augmentations, deeper fine-tuning, or longer schedules will change the final numbers.

