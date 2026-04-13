# Steps for a Fresh Run: Family Voice Classification

Follow these steps to reset your project and train a clean model for family voice identification.

### 1. Cleanup (Manual)
Delete existing data to prevent mixing old and new versions:
*   **Audio Segments**: Delete the files inside `docs/data/raw/family_voice/` (`mine`, `son`, `daughter`).
*   **Full Audio (WAV)**: Delete the files inside `docs/data/wav/`. These are the intermediate files extracted from videos.
*   **Spectrogram Images**: Delete the files inside `docs/data/processed/family_voice/`.
*   **JSON Manifests**: Delete `train.json` and `validation.json` in `docs/data/processed/family_voice/`. These are automatically recreated during training to map images to labels.
*   **Results**: Delete the contents of the `docs/results/family_voice_vit/` directory (or the whole directory; the training script will recreate it).

---

### 2. Audio Extraction & Slicing (Notebook)
Open `sound_classification.ipynb`:
1.  **Extract**: Run the audio extraction cells for your family videos (now located in the `videos/` folder) to get the full `.wav` files. These will be saved to `docs/data/wav/`.
2.  **Slice**: Run the **Slicing Loop** cells.
    *   **CRITICAL**: Ensure `SECONDS = 6` is set in the config part of these cells.
    *   This will populate `docs/data/raw/family_voice/` with 6-second clips from the audio in `docs/data/wav/`.

---

### 3. Spectrogram Generation (Notebook)
In `sound_classification.ipynb`:
1.  Run the `create_spectrograms()` cell.
2.  **Verify**: Check that `docs/data/processed/family_voice/` now contains `.png` images for each person.

---

### 4. Training (Terminal)
1.  **Config**: Open `config.yaml` and ensure `current_task: vit_classification`.
2.  **Execute**: Run the training script from your project root:
    ```bash
    ./.venv/bin/python src/svlearn_vit_cnn/transfer_learning/audio_classification.py
    ```
3.  **Wait**: The script will train the model and save the best version to `docs/results/family_voice_vit/final_model`.

---

### 5. Inference / Testing (Notebook)
In `sound_classification.ipynb`:
1.  Scroll to the **VOICE ANALYSIS BREAKDOWN** cell.
2.  Set `TEST_AUDIO_PATH` to a new audio file you want to test.
3.  Run the cell. It will load your newly trained model and show the classification results.
