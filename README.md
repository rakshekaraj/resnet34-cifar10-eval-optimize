# ResNet34 on CIFAR-10: Evaluation and Optimization

This project investigates the performance of a ResNet34 model on the CIFAR-10 image classification dataset. Rather than just training the model end-to-end, the focus here is on evaluating its baseline performance, applying optimization strategies, and understanding how different training configurations affect accuracy, convergence, and generalization.

The code is written using PyTorch and includes detailed evaluation steps, accuracy/loss tracking, and techniques to improve model training.

## Overview

We start with a pretrained ResNet34 model, adjust it for the 10-class CIFAR-10 problem, and evaluate how well it performs out of the box. From there, we explore several optimization techniques to improve its training and generalization performance.

Key aspects of the project include:

- A reproducible training and evaluation pipeline using PyTorch
- Baseline accuracy benchmarking
- Optimizer and scheduler experimentation
- Loss and accuracy tracking across epochs
- Potential for further extension with techniques like Grad-CAM and ONNX export

## Optimization Strategies Covered

- Learning rate scheduling (StepLR and ReduceLROnPlateau)
- Weight decay tuning and regularization
- Data augmentation (random crop, horizontal flip)
- Optional dropout insertion
- Early stopping heuristics (manual, not automated)
- Train/validation/test split tracking

## Results Summary

After running multiple training configurations, here are some representative outcomes:

- Training accuracy: up to ~94%
- Validation accuracy: ~91%
- Test accuracy: ~90%
- Optimizer used: Adam
- Epochs: typically 20–30 depending on LR schedule and early stopping

These numbers vary slightly based on the specific hyperparameters, but show clear improvements after applying regularization and learning rate schedules.

## Files in This Repository

- `resnet34_eval_opt.ipynb`: Jupyter notebook with training, evaluation, and optimization routines
- `saved_model.pth`: (optional) weights of the trained model for reuse or inference
- `README.md`: this file

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

To install the dependencies:

```bash
pip install torch torchvision matplotlib
