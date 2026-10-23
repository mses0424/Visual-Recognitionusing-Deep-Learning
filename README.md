# Visual Recognition using Deep Learning

This repository contains four homework assignments and a final project for the **Visual Recognition using Deep Learning** course.

The projects cover various computer vision tasks, including image classification, object detection, instance segmentation, image restoration, and image super-resolution.

## Homework Assignments

### [HW1 — Image Classification](./hw1)
Implement a **100-class image classification** model using a pretrained **ResNet-101** backbone. Explore transfer learning, data augmentation, Mixup, CutMix, and learning rate scheduling to improve classification accuracy.

### [HW2 — Object Detection](./hw2)
Develop a **digit detection** system based on a hybrid **Deformable DETR** architecture with a ResNet-50 backbone. Utilize multi-scale deformable attention and Transformer-based decoding for accurate object localization.

### [HW3 — Instance Segmentation](./hw3)
Implement a **cell nucleus instance segmentation** model using **Mask R-CNN with ResNet-50 FPN**. Focus on detecting and segmenting individual nuclei through customized anchor sizes, data augmentation, and fine-tuning.

### [HW4 — Image Restoration](./hw4)
Implement **PromptIR**, a prompt-based Transformer architecture, for all-in-one image restoration. The goal is to recover clean images degraded by **rain and snow**, using a unified restoration model.

## [Final Project — Image Super-Resolution](./project)

The final project focuses on **×4 image super-resolution** for video game images using the **Hybrid Attention Transformer (HAT)** architecture.

Two pretrained models, **HAT and HAT-S**, are fine-tuned and evaluated to improve image reconstruction quality.

The experiments explore data cleaning, training strategies, and Test-Time Augmentation (TTA), with performance evaluated using PSNR.

| Model | Private PSNR | Public PSNR |
|---|---:|---:|
| HAT | 33.607 | 33.613 |
| HAT-S | **33.629** | **33.631** |

## Technologies

**Python · PyTorch · CNN · Vision Transformer · Transfer Learning · Computer Vision**

Detailed implementations, training configurations, and experimental results are provided in the corresponding directories.
