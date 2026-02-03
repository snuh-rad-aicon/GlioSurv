# GlioSurv: Interpretable Transformer for Multimodal, Individualized Survival Prediction in Diffuse Glioma

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41746--025--02018--x-blue)](https://doi.org/10.1038/s41746-025-02018-x)

GlioSurv is an interpretable multimodal transformer for personalized survival prediction in adult diffuse glioma. It integrates multiparametric MRI, clinical data, molecular markers, and treatment information to provide accurate risk stratification.

This repository contains the official implementation for the paper *[GlioSurv: interpretable transformer for multimodal, individualized survival prediction in diffuse glioma](https://doi.org/10.1038/s41746-025-02018-x)* (npj Digital Medicine, 2025).

## Citation

This is the **official implementation** of **GlioSurv**, published in *npj Digital Medicine*. If you find this code or the pre-trained weights useful, please cite our paper:

> **GlioSurv: interpretable transformer for multimodal, individualized survival prediction in diffuse glioma**, *npj Digital Medicine*, 2025.  
> [https://doi.org/10.1038/s41746-025-02018-x](https://doi.org/10.1038/s41746-025-02018-x)

```bibtex
@article{lee2025gliosurv,
  title={GlioSurv: interpretable transformer for multimodal, individualized survival prediction in diffuse glioma},
  author={Lee, Junhyeok and Jang, Joon and Eum, Heeseong and Jang, Han and Kim, Minchul and Park, Sung Hye and Park, Chul Kee and Choi, Seung Hong and Ahn, Sung Soo and Han, Yoseob and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={660},
  year={2025},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41746-025-02018-x},
  url={https://doi.org/10.1038/s41746-025-02018-x}
}
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU training)
- MONAI, nibabel (medical imaging)
- PyTorch Lightning, torchmetrics, torchsurv
- See `requirements.txt` for the full list.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/snuh-rad-aicon/GlioSurv.git
    cd GlioSurv
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 🚀 Pre-trained Weights

The official pre-trained checkpoints for GlioSurv are available on **[GitHub Releases (v1.0)](https://github.com/snuh-rad-aicon/GlioSurv/releases/tag/v1.0)**.

Please download the files from the release page and place them in your designated model directory. Use the GlioSurv checkpoint for inference (e.g. `--pretrain` in `scripts/inference.sh`). Optionally, the ViT-3D checkpoint can be used to initialize the vision encoder when training GlioSurv from scratch (set `vision_pretrain` in `configs/gliosurv.yaml`).

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
