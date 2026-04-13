# Vision Transformers vs CNN: Image & Audio Classification

## Overview
This repository explores the application of Vision Transformers (ViT) and CNNs (ResNet-50) across two distinct domains:
1. **Tree Identification**: Multi-class classification of tree species from images.
2. **Family Voice Identification**: Classification of family members ('mine', 'son', 'daughter') using LogMel spectrograms generated from audio recordings.

The project demonstrates how the same underlying architectures (ViT and ResNet) can be applied to diverse data types once converted to a visual format.

## Repository Layout
- `config.yaml` – Central configuration for task switching and dataset paths.
- `src/svlearn_vit_cnn/transfer_learning/`
    - `trees_classification.py` – Training script for the tree identification task.
    - `audio_classification.py` – Training script for the family voice classification task.
- `src/svlearn_vit_cnn/utils/` – Shared utilities for training, metrics, and visualization.
- `docs/notebooks/`
    - `trees_classification.ipynb` – Evaluation and inference for tree species.
    - `family_voice_classification.ipynb` – Audio preprocessing and voice identification analysis.
- `voice_classification_analysis.md` – Detailed breakdown of the audio pipeline, known gaps (e.g., multilingual support), and optimization tips.

## Environment Setup
1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```
   This environment includes `transformers`, `torch`, `datasets`, and `librosa` (for audio processing).

---

## 1. Tree Classification (Image Domain)

### Dataset Preparation
1. Download the tree dataset and arrange it as follows:
   ```
   data/trees/
     ├── Oak/
     └── WeepingWillow/
     ...
   ```
2. Update `tree-dataset.path` in `config.yaml`.

### Training
1. Set `current_task` in `config.yaml` to either `vit_classification` or `resnet_classification`.
2. Run the training script:
   ```bash
   uv run src/svlearn_vit_cnn/transfer_learning/trees_classification.py
   ```

---

## 2. Family Voice Classification (Audio Domain)

### Workflow: Audio to Spectrograms
Instead of raw audio, this pipeline uses **LogMel Spectrograms**—visual representations of sound frequencies over time.

1.  **Extraction**: Audio is extracted from family videos.
2.  **Slicing**: Long audio is sliced into fixed-duration segments (e.g., 6 seconds).
3.  **Visualization**: Segments are converted to `.png` spectrogram images using `librosa`.
4.  **Organization**: Images are stored in class-specific folders (`mine/`, `son/`, `daughter/`).

### Training
1.  Configure the `family-voice-dataset` section in `config.yaml` with your image paths.
2.  Set `current_task` (Vit or ResNet).
3.  Launch the audio training script:
    ```bash
    uv run src/svlearn_vit_cnn/transfer_learning/audio_classification.py
    ```

**Note on Specialization**: The `audio_classification.py` script is optimized for this task by unfreezing the final encoder layer (Layer 11 for ViT) to learn the specific nuances of vocal cord characteristics.

### Evaluation & Gaps
The `voice_classification_analysis.md` file tracks critical findings, such as:
*   **Duration Consistency**: Ensuring training and inference use the same audio segment length.
*   **Linguistic Diversity**: The importance of training on multiple languages (e.g., English and Tamil) to ensure the model learns *who* is speaking rather than *what* is being said.

---

## Metrics Snapshot
`evaluation_results.csv` captures baseline results across different experiments:

| Dataset | Model | Accuracy (%) | Inference Time (s/image) |
| --- | --- | --- | --- |
| cifar-10 | ViT | 8.94 | 0.0020 |
| cifar-10 | ResNet | 12.72 | 0.0017 |
| oxford_flowers | ViT | 0.016 | 0.0015 |
| oxford_flowers | ResNet | 0.016 | 0.0019 |
| trees | ViT | 45.69 | 0.0018 |
| trees | ResNet | 48.68 | 0.0025 |

*Note: Voice classification metrics are typically tracked directly in the `results/` folder via `eval_results.json`.*

