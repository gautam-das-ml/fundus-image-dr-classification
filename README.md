# Fundus-Image-DR-Classification
# Diabetic Retinopathy Classification

Deep learning-based classification of retinal fundus images into **Normal** and **Diabetic Retinopathy** categories.

## Models

* ResNet-18 (pre-trained CNN that used residual blocks and skip connections, contains 11.7 million parameter, was fine tuned for diabetic retinopathy classification)
* EfficientNet-B0 (pre-trained CNN that used compound scaling and MBConv blocks, contains 5.3 million parameter, was fine tuned for the same purpose)

## Code Layout

* Names of code files with '....py' format represents resent-18 files.
* Names of code files with '...._2.py' format represents efficientnet-b0 files.
* app.py and index.html represent the web interfaces files, not deployed, only locally hosted in flask.

## Dataset

APTOS 2019 Blindness Detection Dataset from Kaggle

## Technologies

PyTorch, Torchvision, OpenCV, Flask, NumPy, Pandas, Scikit-Learn

## Results

| Model           | Accuracy |
| --------------- | -------- |
| ResNet-18       | 98.36%   |
| EfficientNet-B0 | 98.09%   |

## Features

* Single Image Prediction
* Multiple Image Prediction
* Web-based Interface

## Future Work
* Multi-class DR Severity Classification
* Lesion Localization
* Mobile Application Deployment

