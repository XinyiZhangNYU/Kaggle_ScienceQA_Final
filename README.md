# SmolVLM-500M LoRA Fine-tuning & Inference Pipeline for ScienceQA

This repository contains the complete pipeline for fine-tuning `SmolVLM-500M-Instruct` to answer multiple-choice science questions given an image, lecture, hint, and a list of candidate answers, designed for the **Pixels to Predictions: DL Vision Challenge**.

The project is structured into four phases:

- **Data Verification**: Comprehensive sanity check on competition data before training.
- **Model Training**: Parameter-efficient LoRA fine-tuning on Google Colab, capped at 5M trainable parameters.
- **Local Sanity Check**: Quick 5-sample validation before deploying to Kaggle.
- **Kaggle Inference**: Robust offline inference pipeline producing the final `submission.csv`.

---

## Repository Structure

```
.
├── Colab_Train_final.ipynb          # Main training notebook (Google Colab)
├── inference_final.ipynb            # Kaggle offline inference notebook
├── sanity_check_v2.py            # 13-category dataset audit script
├── generate_split_metadata.py    # Generates val_unseen.csv from val.csv
├── README.md                     # This file
└── requirements.txt              # Pinned dependency versions
```

> **Note**: `train.csv`, `val.csv`, `test.csv` and image folders are not included in this repository. Please download them from the official Kaggle competition page.

---

## Phase 1: Data Preparation

Before training, you need to prepare two things locally:

### Prerequisites
Place the following files in the same directory:
- `train.csv`, `val.csv`, `test.csv` — official competition data
- `train/`, `val/`, `test/` — official image folders
- `sanity_check_v2.py` — dataset audit script
- `generate_split_metadata.py` — unseen-validation generator

### Execution Steps

**1. Run the Sanity Check** (recommended, optional)
```bash
python sanity_check_v2.py
```
This audit reports across 13 categories: missing values, choice parseability, label distribution, cross-split overlap, image integrity, etc. Expected output: zero errors and zero corruption (the official data is structurally clean).

**2. Generate the Unseen Validation Subset** (required)
```bash
python generate_split_metadata.py
```
**Output**: `val_unseen.csv` — the subset of `val.csv` whose questions never appear in `train.csv`. This is used during training as a generalization-tracking metric and is **required** by the training notebook.

---

## Phase 2: Training on Google Colab

The training notebook (`Colab_Train_v3.ipynb`) is designed for Google Colab Free Tier (T4 GPU). Training takes approximately **6 hours on T4**.

### Hardware Requirements
- **GPU**: T4 (16GB VRAM, free tier) — sufficient
- **Peak VRAM usage**: ~10 GB
- **Training time on T4**: ~6 hours for 3 epochs

### Cloud Storage Setup

To prevent data loss when the Colab runtime disconnects, the notebook integrates with Google Drive.

**Steps:**

1. Open your Google Drive.
2. At the root of "My Drive", create a folder named exactly: **`Kaggle_ScienceQA`**
3. Upload the following into this `Kaggle_ScienceQA/` folder:
   ```
   Kaggle_ScienceQA/
   ├── train.csv
   ├── val.csv
   ├── val_unseen.csv          ← Generated in Phase 1, required
   ├── train/                  ← Folder containing all training images
   └── val/                    ← Folder containing all validation images
   ```

> **Naming matters.** The folder must be exactly `Kaggle_ScienceQA` and the files must be exactly named as listed above. The training notebook does not search for alternate names.

### Execution

1. Open `Colab_Train_v3.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**.
4. The notebook will:
   - Install pinned dependencies (`transformers==4.57.6`, `peft==0.18.1`, `accelerate==1.13.0`)
   - Mount Google Drive (you will be prompted to authorize)
   - Load the dataset and apply image padding + text truncation
   - Configure LoRA at `r=16`, targeting `q_proj/k_proj/v_proj/o_proj` (4.16M trainable params)
   - Train for 3 epochs with weighted sampling and unseen-loss-driven model selection
   - Save the best LoRA artifact to Google Drive
   - Run a 5-sample sanity check on validation data

### Output

After training, find the trained LoRA at:
```
MyDrive/Kaggle_ScienceQA/smolvlm_lora_output/final_lora_weights/
```

This folder contains 12 files (adapter weights, processor configs, tokenizer files). **Download all of them as a single folder** — you will upload this to Kaggle in Phase 4.

---

## Phase 3: Local Sanity Check

The final cell of the training notebook runs an automatic 5-sample sanity check on validation data:
- Loads the saved LoRA back on top of the base model
- Runs full inference (with `pad_to_square` and the same prompt format used at Kaggle)
- Reports per-sample correctness and overall accuracy

**Expected result**: ≥3 out of 5 correct. If 0/5 correct, the LoRA save may have failed or the prompt format is misaligned — re-check Cell 7 (Save) and Cell 3 (Prompt Builder) before proceeding.

---

## Phase 4: Kaggle Inference

### Step 4.1: Upload Three Datasets to Kaggle

Create three separate Kaggle datasets. **Naming is critical** — the auto-discovery code in `inference_v3.ipynb` searches for folders containing specific keywords, so the dataset names below must match exactly (case-insensitive):

#### Dataset 1: Trained LoRA Weights
- **Dataset name**: `my-scienceqa-smolvlm-lora`
- **Contents**: Upload the entire `final_lora_weights/` folder you downloaded from Drive in Phase 2.
- The folder structure on Kaggle should look like:
  ```
  my-scienceqa-smolvlm-lora/
  └── final_lora_weights/
      ├── adapter_config.json
      ├── adapter_model.safetensors
      ├── processor_config.json
      ├── tokenizer.json
      └── ... (12 files total)
  ```

#### Dataset 2: SmolVLM-500M Base Model
- **Dataset name**: `smolvlm-500m-instruct`
- **Contents**: The full `SmolVLM-500M-Instruct` model from HuggingFace.
- The folder structure on Kaggle should look like:
  ```
  smolvlm-500m-instruct/
  └── SmolVLM-500M-Instruct/
      ├── config.json
      ├── model.safetensors
      ├── tokenizer.json
      └── ... (~12 files)
  ```
  > **Note**: The auto-discovery code will recursively find `model.safetensors`, so the nested subfolder is fine — you do not need to flatten the structure.


### Step 4.2: Configure the Kaggle Notebook

1. Open `inference_v3.ipynb` in Kaggle.
2. **Settings panel**:
   - **Accelerator**: GPU T4 × 2
   - **Internet**: **Off** (mandatory — competition rule)
   - 
3. **Input panel** (right sidebar): Click **"+ Add Input"** and add all two datasets:
   - Pixels to Predictions: DL Vision Challenge (the competition itself)
   - `my-scienceqa-smolvlm-lora`
   - `smolvlm-500m-instruct`

   Your Input panel should look like the screenshot below:
   ```
   COMPETITIONS
     Pixels to Predictions: DL Vision Challenge
   DATASETS
     my-scienceqa-smolvlm-lora
       └── final_lora_weights
     smolvlm-500m-instruct
       └── SmolVLM-500M-Instruct
   ```

### Step 4.3: Run Inference

Click **Run All**. The notebook executes three cells in order:

#### Cell 0 — Path Auto-Discovery 

Recursively scans `/kaggle/input/` to locate the LoRA folder, base model folder, and dataset paths. **Verify the output before proceeding**:

```
Model weight directory: /kaggle/input/.../smolvlm-500m-instruct/SmolVLM-500M-Instruct
LoRA weight directory:  /kaggle/input/.../my-scienceqa-smolvlm-lora/final_lora_weights
MODEL_PATH contents: ['config.json', 'model.safetensors', ...]
LORA_PATH contents:  ['adapter_config.json', 'adapter_model.safetensors', ...]
```

If any path is `None` or contents look wrong, the dataset names or folder structure are off. Re-check Step 4.1.

#### Cell 1 — Environment Check 

Confirms `AutoModelForImageTextToText` (transformers 5.0) is importable. If running on a Kaggle image with transformers 4.x, falls back automatically.

#### Cell 2 — Inference Loop 

Loads model + LoRA, then iterates through all 1,008 test items with a tqdm progress bar. At completion:

```
submission.csv generated.
Answer distribution:
0    ~360
1    ~330
2    ~240
3    ~70
4    ~5
Total predictions: 1008
```

If the answer distribution is healthy (roughly matching the train distribution `36/33/24/7/0.5%`), the model is working correctly. If everything is `0`, the LoRA likely did not load — re-check Cell 0's output.

### Step 4.4: Submit

After Cell 2 finishes, the file `submission.csv` is automatically saved at the top level of the Kaggle notebook. Click **"Submit to Competition"** in the right sidebar to submit it to the public leaderboard.

> **Total Kaggle notebook runtime**: approximately **25-30 minutes** end-to-end.

---

## Compliance Checklist

- ✅ Base model is `HuggingFaceTB/SmolVLM-500M-Instruct`
- ✅ Trainable parameters strictly under 5,000,000 (asserted at every training run)
- ✅ No external data used
- ✅ Inference notebook runs with **Internet: Off** on Kaggle T4 x2
- ✅ All seeds fixed at 42 for reproducibility

## Contact

Xinyi Zhang — `xz3388@nyu.edu` — New York University

For reproduction issues or questions, please open a GitHub Issue.
