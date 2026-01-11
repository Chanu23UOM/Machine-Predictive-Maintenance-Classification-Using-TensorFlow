# Machine Predictive Maintenance Classification

A neural network model built with TensorFlow to predict machine failures based on sensor data.

## Overview

This project implements a deep learning approach to predictive maintenance, helping identify potential equipment failures before they occur. The model analyzes various machine parameters to classify whether a failure is likely.

## Dataset

The dataset is sourced from Kaggle: [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

### Features

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Type (encoded)

### Target

- Binary classification: Failure (1) or No Failure (0)

## Model Architecture

The neural network consists of:

- Input layer: 6 features
- Hidden layer 1: 16 units with ReLU activation
- Hidden layer 2: 8 units with ReLU activation
- Hidden layer 3: 4 units with ReLU activation
- Output layer: 1 unit with Sigmoid activation

### Model Parameters

| Layer   | Weights Shape | Biases Shape |
|---------|---------------|--------------|
| layer1  | (6, 16)       | (16,)        |
| layer2  | (16, 8)       | (8,)         |
| layer3  | (8, 4)        | (4,)         |
| output  | (4, 1)        | (1,)         |

## Model Performance

### Test Results

| Metric        | Value   |
|---------------|---------|
| Test Loss     | 0.0629  |
| Test Accuracy | 97.85%  |

### Classification Report

```
              precision    recall  f1-score   support

  No Failure       0.99      0.99      0.99      1939
     Failure       0.67      0.59      0.63        61

    accuracy                           0.98      2000
   macro avg       0.83      0.79      0.81      2000
weighted avg       0.98      0.98      0.98      2000
```

### Confusion Matrix

```
[[1921   18]
 [  25   36]]
```

|                    | Predicted: No Failure | Predicted: Failure |
|--------------------|----------------------:|-------------------:|
| Actual: No Failure | 1921 (TN)             | 18 (FP)            |
| Actual: Failure    | 25 (FN)               | 36 (TP)            |

### Sample Predictions

| Probability | Predicted    | Actual       |
|-------------|--------------|--------------|
| 0.0007      | No Failure   | No Failure   |
| 0.0000      | No Failure   | No Failure   |
| 0.0001      | No Failure   | No Failure   |
| 0.0000      | No Failure   | No Failure   |
| 0.1214      | No Failure   | No Failure   |

## Requirements

```
numpy
pandas
matplotlib
tensorflow
scikit-learn
kagglehub
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/predictive-maintenance.git
cd predictive-maintenance
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn kagglehub
```

3. Configure Kaggle API credentials (required for dataset download)

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook C2_W1_Lab02_Machine_Predictive_Maintenance_Classification_TF.ipynb
```

2. Run all cells sequentially to:
   - Download the dataset
   - Preprocess and normalize the data
   - Train the neural network
   - Evaluate model performance
   - Generate predictions

## Project Structure

```
predictive-maintenance/
 C2_W1_Lab02_Machine_Predictive_Maintenance_Classification_TF.ipynb  # Main notebook
 README.md                                                            # Project documentation
 predictive_maintenance_model.keras                                   # Saved model (after training)
```

## License

This project is for educational purposes.

## Acknowledgments

- Dataset provided by Shivam Bansal on Kaggle
- Built using TensorFlow and Keras
