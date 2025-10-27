
Alzheimer MRI Classification 
A project for classifying Alzheimer's disease stages from MRI scans using convolutional neural networks.

This project implements a CNN model to classify MRI images into different stages of Alzheimer's disease. The model achieves 67.48% validation accuracy on the Alzheimer MRI dataset from Hugging Face.

Model Performance
Training Accuracy: 63.96%

Validation Accuracy: 67.48%

Training Loss: 0.7235

Validation Loss: 0.7175

Dataset
The project uses the Alzheimer MRI Dataset from Hugging Face (Falah/Alzheimer_MRI), containing MRI scans categorized into different stages of Alzheimer's disease.

Dataset Structure
Multiple classes representing different Alzheimer's stages

Pre-processed MRI images

Training and validation splits

Installation
Prerequisites
Python 3.7+

pip package manager

Install Dependencies
pip install -r requirements.txt

The CNN model consists of:

Convolutional Layers:

3 convolutional blocks with BatchNorm and ReLU

MaxPooling for downsampling

Feature maps: 32 → 64 → 128

Classifier:

Fully connected layers with dropout

Output dimension matching number of classes

Run model : python alzheimer_classification.py


The script will:

Load and preprocess the dataset

Split data into training/validation sets

Train the CNN model for 10 epochs

Save the best performing model

Generate performance visualizations

Output Features
Automatic model saving (best_alzheimer_model.pth)

Training curves (loss and accuracy)

Confusion matrix visualization

Classification report with precision/recall metrics

Sample predictions with confidence scores





Data Preprocessing
Image resizing to 64×64 pixels

Random horizontal flipping (p=0.3) for augmentation

Normalization using ImageNet statistics

80-20 train-validation split

Training Configuration
Batch Size: 16

Epochs: 10

Optimizer: Adam (lr=0.001)

Scheduler: StepLR (step_size=5, gamma=0.1)

Loss Function: CrossEntropyLoss


The model provides comprehensive evaluation including:

Training Progress: Loss and accuracy curves

Classification Metrics: Precision, recall, F1-score

Confusion Matrix: Visual class-wise performance

Sample Predictions: Test predictions with confidence scores


Note: This model is designed for research purposes and should not be used for clinical diagnosis without proper validation and medical supervision.