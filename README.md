
# NuInsSeg Instance Segmentation (Colab)

This repository contains a Google Colab notebook for training and evaluating an instance segmentation model using the **NuInsSeg** dataset with custom augmentations and a LoRA-efficient SAM model.

## 📁 Files

- `nuinsseg_instance_segmentation_notebook.ipynb`: Main Colab notebook for data preparation, training, evaluation, and visualization.

---

## 🚀 Getting Started

Follow the steps below to run this notebook successfully on Google Colab.

---

### ✅ 1. Open the Notebook in Google Colab

Click the Colab badge (or manually upload the notebook to Google Colab):


---

### 📂 2. Set Up Your Google Drive

The notebook uses Google Drive for:
- Reading the NuInsSeg dataset
- Saving training logs, visualizations, and results

**Modify the following line to point to your dataset location:**

```python
DATASET_DIR = "/content/drive/MyDrive/medical_image_computing/NuInsSeg/"

```

Replace "/content/drive/MyDrive/medical_image_computing/NuInsSeg/" with the actual path to your NuInsSeg dataset in Google Drive.

### Run Each Section Sequentially

Proceed through these notebook sections in order:

1. Environment Setup: Mount Google Drive, install dependencies.
2. Dataset Preparation: Load and preprocess NuInsSeg.
3. Model Definition: Load EfficientSAM with LoRA QKV integration.
4. Training Loop: Use K-Fold CV or single fold.
5. Evaluation & Metrics: Dice, AJI, PQ, and more.
6. Visualization: Save overlay images and predictions to Drive.
