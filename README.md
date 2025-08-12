# GlioSurv: A Multimodal Transformer for Personalized Survival Prediction in Adult Diffuse Glioma

GlioSurv is a multimodal transformer model for personalized survival prediction in adult diffuse glioma. It integrates multiparametric MRI, clinical data, molecular markers, and treatment information to provide accurate risk stratification.

This repository contains the official implementation for the paper, including code for data preprocessing, training, and inference.

## Requirements

- Python 3.8+
- PyTorch
- MONAI
- ... (add other requirements)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/snuh-rad-aicon/GlioSurv.git
    cd GlioSurv_github
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Pre-trained Models

The pre-trained weights for the GlioSurv model will be uploaded upon publication of the manuscript.

## Usage

### Data Preprocessing

Run the preprocessing script. This will handle tasks like image registration, normalization, etc.

```bash
bash scripts/preprocessing.sh
```

### Training

The training process consists of two stages. First, we pre-train a 3D Vision Transformer (ViT-3D) on the imaging data. Then, we use the weights from the pre-trained ViT-3D to initialize and train the final GlioSurv model.

#### Stage 1: Pre-train 3D Vision Transformer

Run the following script to pre-train the ViT-3D model. The trained weights will be saved and used in the next stage.

```bash
bash scripts/run_vit3d.sh
```

#### Stage 2: Train GlioSurv Model

After pre-training the ViT-3D, run the following script to train the GlioSurv model. This script will load the pre-trained weights and train the full multimodal model.

```bash
bash scripts/run_gliosurv.sh
```

### Inference

To run inference on new data, use the `inference.sh` script. Make sure to modify the script to point to your data and desired output paths.

```bash
bash scripts/inference.sh
```

## Analysis

The `analysis.ipynb` notebook contains further analysis and visualization of the results.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
