from flask import Flask, Response
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
import base64
import os

app = Flask(__name__)

@app.route('/predict')
def visualize_and_predict():
    # Load the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv', nrows=1000)

    # Fill missing values
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

    # Select numerical columns to plot
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    cols_to_plot = numerical_cols[:min(len(numerical_cols), 6)]

    # Start building HTML
    html = "<h1>Housing Data Analysis</h1>"

    # Prepare features and target
    X = df.drop('median_house_value', axis=1)
    X = pd.get_dummies(X)
    y = df['median_house_value']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Metrics Block
    html += f"""
    <div style='padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc; margin-bottom: 20px;'>
      <strong>Model Metrics:</strong><br>
      <strong>MSE:</strong> {mse:.2f}<br>
      <strong>MAE:</strong> {mae:.2f}<br>
      <strong>RMSE:</strong> {rmse:.2f}
    </div>
    """

    # Helper function to convert plots to base64
    def plot_to_img():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        return f"<img src='data:image/png;base64,{img_base64}'/><br>"

    # Histogram
    plt.figure(figsize=(12,6))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(2,3,i+1)
        sns.histplot(df[col], kde=True)
        plt.title(col)
    plt.tight_layout()
    html += "<h2>Histograms</h2>" + plot_to_img()

    # Box plot
    plt.figure(figsize=(12,6))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(2,3,i+1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    html += "<h2>Box Plots</h2>" + plot_to_img()

    # Count plot
    plt.figure(figsize=(12,6))
    for i, col in enumerate(categorical_cols):
        plt.subplot(2,3,i+1)
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        plt.xticks(rotation=45, ha='right')
        plt.title(col)
    plt.tight_layout()
    html += "<h2>Count Plots</h2>" + plot_to_img()

    # Heatmap
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12,6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='coolwarm')
    plt.title("Heatmap")
    html += "<h2>Correlation Heatmap</h2>" + plot_to_img()

    # Scatter plot
    plt.figure(figsize=(20,6))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(2,3,i+1)
        sns.scatterplot(x=col, y='median_house_value', data=df)
        plt.title(col)
    plt.suptitle('Scatter plot of columns chosen', fontsize=16)
    plt.tight_layout()
    html += "<h2>Scatter Plots</h2>" + plot_to_img()

    return Response(html, mimetype='text/html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
