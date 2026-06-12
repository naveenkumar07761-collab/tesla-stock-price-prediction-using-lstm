# Tesla Stock Price Prediction using LSTM 🚀

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict Tesla (TSLA) stock prices based on historical market data. The model learns patterns from past stock prices and generates future price predictions.

The project demonstrates the application of deep learning techniques for time-series forecasting in the financial domain.

---

## Features
- Historical Tesla stock data analysis
- Data preprocessing and normalization
- LSTM-based deep learning model
- Model training and evaluation
- Visualization of actual vs predicted stock prices
- Future stock price forecasting

---

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Dataset
The model is trained using historical Tesla (TSLA) stock market data containing:

- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume

Data Source: Yahoo Finance

---

## Model Architecture
The model consists of:

1. Input Layer
2. LSTM Layers
3. Dropout Layers
4. Dense Output Layer

To improve performance, the data is normalized using MinMaxScaler before training.

---

## Results
The trained model predicts Tesla stock prices and compares them with actual market values. The prediction graph demonstrates how well the model captures overall trends in the stock market.

Evaluation metrics can include:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/tesla-stock-price-prediction.git
```

Navigate to the project directory:

```bash
cd tesla-stock-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project:

```bash
python main.py
```

or open the notebook:

```bash
jupyter notebook
```

---

## Project Structure

```text
tesla-stock-price-prediction/
│
├── data/
├── notebooks/
├── images/
├── model/
├── requirements.txt
├── main.py
└── README.md
```

---

## Disclaimer
This project is intended for educational purposes only. Stock price predictions are inherently uncertain and should not be considered financial advice.

---

## Author

**Naveen Kumar**

GitHub: https://github.com/naveenkumar07761
