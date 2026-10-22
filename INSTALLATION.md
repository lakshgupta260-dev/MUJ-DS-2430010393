# Installation & Execution Guide

This document provides step-by-step instructions to set up the environment, run inference, train models, and reproduce the benchmarks on both local hardware and cloud GPUs (Google Colab).

---

## 1. Prerequisites
- Python 3.9+ or Python 3.10 / 3.11
- CUDA-capable GPU (Recommended for training, e.g., NVIDIA T4 / RTX 3060+) or CPU for inference
- Git

---

## 2. Local Environment Setup

### Step A: Clone the Repository
```bash
git clone https://github.com/lakshgupta260/MUJ-DS-2430010393.git
cd MUJ-DS-2430010393
```

### Step B: Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step C: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Running Single-Image Inference
To test an arbitrary tomato leaf image with the fine-tuned Vision Transformer:

```bash
python code/predict.py --image path/to/leaf_sample.jpg --model results/best_vit_model.pth
```

Output:
```text
Predicted Disease: Late_blight
Confidence: 99.12%
```

---

## 4. Training on Google Colab (Recommended)
1. Open Google Colab: https://colab.research.google.com
2. Change runtime type to **T4 GPU**: `Runtime -> Change runtime type -> T4 GPU`.
3. Upload `notebooks/04_Vision_Transformer_ViT.ipynb` (or `notebooks/Tomato_Disease_Complete_Pipeline.ipynb`).
4. Upload your dataset zip: `tomato_data.zip.zip` to `/content/`.
5. Run all cells sequentially.
6. The training script will automatically save `best_vit_model.pth` and display the final test classification report and confusion matrix.
