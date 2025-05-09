# League of Legends Match Predictor

A machine learning model that predicts the outcome of League of Legends matches using logistic regression implemented with PyTorch.

## Overview

This project implements a binary classification model to predict whether a team will win or lose a League of Legends match based on various in-game metrics. The model uses a simple logistic regression approach with PyTorch to achieve high prediction accuracy.

## Features

- Data preprocessing with standardization
- Logistic regression model implementation using PyTorch
- Hyperparameter tuning (learning rate optimization)
- Regularization techniques (L2 regularization)
- Comprehensive model evaluation:
  - Accuracy metrics
  - Classification reports
  - Confusion matrix visualization
  - ROC curve analysis
  - Feature importance analysis

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Dataset

The model uses the `league_of_legends_data_large.csv` dataset (not included in this repository), which contains various in-game metrics as features and a binary 'win' column as the target variable.

## Model Performance

The logistic regression model achieves:
- Training Accuracy: ~85%
- Test Accuracy: ~83%

## Usage

### Training the model

```python
python lol_match_predictor.py
```

### Loading a pre-trained model

```python
loaded_linear = nn.Linear(input_dim, 1)
loaded_linear.load_state_dict(torch.load('trained_model.pth'))
loaded_linear.eval()
```

## Results

The model analyzes feature importance to identify which in-game metrics are most predictive of match outcomes. The confusion matrix and ROC curve visualizations provide insights into the model's performance.

## Future Improvements

- Experiment with more complex architectures (e.g., neural networks)
- Feature engineering to create more predictive variables
- Implement cross-validation for more robust evaluation
- Add early-game prediction capabilities
- Create a web interface for predictions

## Acknowledgements

This project was developed using PyTorch and scikit-learn libraries.

