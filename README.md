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
jupyter notebook C2_W1_Lab02_CoffeeRoasting_TF.ipynb
```

2. Run all cells sequentially to:
   - Download the dataset
   - Preprocess and normalize the data
   - Train the neural network
   - Evaluate model performance
   - Generate predictions

## Results

The model provides:

- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- Feature importance analysis
- Training and validation loss/accuracy curves

## Project Structure

```
predictive-maintenance/
├── C2_W1_Lab02_CoffeeRoasting_TF.ipynb    # Main notebook
├── README.md                               # Project documentation
└── predictive_maintenance_model.keras      # Saved model (after training)
```

## License

This project is for educational purposes.

## Acknowledgments

- Dataset provided by Shivam Bansal on Kaggle
- Built using TensorFlow and Keras
