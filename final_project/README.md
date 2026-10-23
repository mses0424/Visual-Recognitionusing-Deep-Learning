# Visual Recognition using Deep Learning

## Project Overview

This project focuses on **×4 Image Super-Resolution** using the **Hybrid Attention Transformer (HAT)** architecture on the Kaggle video game super-resolution dataset.

We explore and fine-tune two pretrained models, **HAT** and **HAT-S**, to enhance low-resolution images and improve reconstruction quality.

## Objectives

- Fine-tune pretrained HAT and HAT-S models for ×4 image super-resolution.
- Improve training data quality through PSNR-based data cleaning.
- Explore different training strategies and hyperparameter settings.
- Apply 8× Test-Time Augmentation (TTA) to improve reconstruction performance.
- Compare the performance of HAT and HAT-S using PSNR scores.

## Results

| Model | Best Private PSNR | Best Public PSNR |
|---|---:|---:|
| HAT | 33.607 | 33.613 |
| HAT-S | **33.629** | **33.631** |

Both models achieve competitive super-resolution performance, with HAT-S obtaining slightly higher PSNR scores while using a smaller architecture.
