# Image-to-Image Translation: Photo ↔ Monet Style Transfer

A comparative study of non-Bayesian and Bayesian approaches to unpaired image-to-image translation, applying three distinct models to translate photographs into Monet-style paintings.

## Project Overview

This project explores how machines can learn to render photographs in the style of Claude Monet. We implement and compare three models that span different paradigms — adversarial training, variational inference, and Bayesian neural networks:

- **CycleGAN** — Adversarial training with cycle consistency loss; bidirectional (Photo ↔ Monet)
- **UNIT** — Shared latent space with VAE-GAN; bidirectional (Photo ↔ Monet)
- **BNN Style Transfer** — Bayesian residual blocks with perceptual loss; unidirectional (Photo → Monet)

The models are evaluated using FID and MiFID scores alongside qualitative visual comparisons.

## Dataset

The [Kaggle GAN Getting Started](https://www.kaggle.com/competitions/gan-getting-started/data) dataset, containing 300 Monet paintings and 7,038 photographs, resized to 256×256.

## Repository Structure

```
├── 1_CycleGAN_final.ipynb  # CycleGAN implementation
├── 2_UNIT_final.ipynb      # UNIT implementation
├── 3_BNN_final.ipynb       # Bayesian feedforward style transfer
├── Report.pdf              # 10-page project report
└── README.md
```

## Development Environment

All notebooks were developed using the **Google Colab extension for VS Code** and trained on **Google Colab's GPU runtime** with **CUDA** acceleration (T4 / A100 depending on availability). Key dependencies include PyTorch, torchvision, `bayesian-torch` (for BNN layers), and `torchmetrics` (for FID/MiFID evaluation).
