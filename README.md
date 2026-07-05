# Text Classification using TextCNN

## Overview
This project was developed as part of a university Deep Learning course.

It trains a TextCNN model on the SNIPS dataset using PyTorch.

## Features
- SNIPS dataset training
- Dynamic feature extraction (n-gram filters)
- YAML-based configuration management
- TensorBoard integration for training logs
- Learning rate scheduler
- Validation accuracy monitoring
- Model checkpoint saving

## Tech Stack
- Python
- PyTorch
- YAML

## Key Achievements / Results
The model's performance was evaluated based on the validation accuracy of the SNIPS intent classification task. 

Several variants of the TextCNN architecture were tested to determine the most effective configuration, with the following outcomes:

- CNN-rand: 98.71%
- CNN-static: 99.14%
- CNN-non-static: 99.00%
- CNN-multi-channel: 99.57%

The CNN-multi-channel configuration achieved the highest performance with an accuracy of 99.57%. This indicates that leveraging multiple channels—typically by combining static and non-static word embeddings—effectively captures rich, multi-faceted semantic features, leading to superior classification results in this intent detection task.

## Note
The SNIPS dataset files are not included in this repository. Please ensure you place the data in the appropriate directories or update the paths in the configuration file accordingly.

The core architecture of this project is based on course materials provided by the university. 
I have implemented the data preprocessing pipeline, conducted hyperparameter tuning, and developed the training log monitoring system.
