# Family Voice Classification: Workflow & Gaps

You are successfully adapting the gunshot classification pipeline for family voice identification. Below is a breakdown of the steps you've taken and the key gaps that need to be addressed for the system to work reliably.

## Current Workflow

1.  **Data Acquisition**: Extracting audio from family videos using `moviepy`.
2.  **Data Preparation**:
    *   Slicing long audio into short segments (6s or 8s).
    *   Organizing segments into subfolders by class (`mine`, `son`, `daughter`).
3.  **Feature Extraction**:
    *   Running a notebook function to convert audio segments into Mel Spectrogram images (`.png`).
    *   Storing these images in a directory that `svlearn` tools can read.
4.  **Model Configuration**:
    *   Updating `config.yaml` with the new dataset paths and target F1 score.
    *   Adapting `audio_classification.py` to handle 3 classes (multiclass) and save the resulting model.
5.  **Metrics Update**:
    *   Modifying `train_utils.py` to use `macro` averaging for precision, recall, and F1, ensuring the model performs well across all family members.
6.  **Inference (Testing)**:
    *   Implementing a prediction function in the notebook to test the model on new audio.

---

## Critical Gaps & Missing Steps

### 1. Audio Duration Inconsistency (High Priority)
The model's performance depends heavily on the "look" of the spectrogram, which is determined by the audio duration.
*   **Training**: Your `create_spectrograms` function uses `duration=6.0`.
*   **Prediction**: Your `predict_voice` function uses `duration=8.0`.
*   **Fix**: Standardize on one duration (e.g., 6 seconds) for both training image generation and inference.

### 2. Model Path Mismatch
The location where the model is saved during training does not match where the notebook looks for it.
*   **Save Path** (`audio_classification.py`): `/Users/sangeetha/.../results/family_vit`
*   **Load Path** (Notebook): `/Users/sangeetha/.../docs/results/family_voice_vit`
*   **Fix**: Update both to use the same consistent path. It is recommended to use the path defined in `config['family-voice-dataset']['results_vit']`.

### 3. Image Processor Synchronization
You are using slightly different pre-trained processors.
*   **Training**: `ViTImageProcessor.from_pretrained(model_name_or_path)` (usually `google/vit-base-patch16-224-in21k`).
*   **Prediction**: `ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")`.
*   **Fix**: Always load the processor from your *saved model directory* during inference:
    ```python
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    ```

### 4. Raw vs. Processed Folder Confusion
In `config.yaml`, the `path` for training currently points to your *images* (`docs/data/processed/family_voice`). 
*   **Note**: The `svlearn` `Preprocessor` will treat these images as "raw" data and split them into train/validation sets. This works, but ensures you don't accidentally mix `.wav` and `.png` files in the same folders.

### 5. ROC Curve Handling
The binary `plot_roc_curve` will likely fail for 3 classes (which you've handled with a `try-except`). To truly evaluate multiclass performance, you might eventually want to implement a "One-vs-Rest" ROC curve or focus on the **Confusion Matrix**, which is much more informative for 3+ classes.

---

### 6. Linguistic Overfitting (New Discovery)
Your model currently performs perfectly (1.0 F1) on your existing data but fails when a different language (e.g., Tamil) is spoken.
*   **The Cause**: The model has learned phonetic shapes (how words are pronounced in English) rather than just purely vocal cord characteristics. When it "hears" different phonemes in Tamil, it gets confused.
*   **The Fix**: You must include segments of **both languages** in the training set for every person. This teaches the model to ignore "what" is being said and focus only on "who" is saying it.

---

## Recommended Next Steps

1.  **Multilingual Training Data**: Collect and slice 10-15 segments of each person speaking the second language (e.g., Tamil) and add them to your training folders.
2.  **Re-run Spectrograms**: Regenerate the images to include this linguistic diversity.
3.  **Re-train with Unfrozen Layers**: Use the optimized `audio_classification.py` (with unfreezed Layer 11) to learn these more complex, language-independent features.
