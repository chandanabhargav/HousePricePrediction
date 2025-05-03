
# Housing Data Analysis Flask App

This is a simple Flask web application that performs data visualization and prediction using the California Housing dataset.

## Features

- Loads and cleans the housing dataset
- Displays:
  - Histograms of numeric features
  - Box plots of numeric features
  - Count plots for categorical columns
  - Correlation heatmap
  - Scatter plots between selected features and target variable
- Trains a Random Forest Regressor
- Displays MSE, MAE, and RMSE metrics

## Installation

1. Clone the repository or download the files.
2. Install dependencies:

```bash
pip install flask pandas matplotlib seaborn scikit-learn
```

## Run the App

```bash
python predict.py
```

Then visit `http://localhost:5000/predict` in your browser.

## Screenshot

All plots and metrics are embedded directly into the output HTML page.