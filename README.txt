
---

# 📊 Gold Price Prediction using Linear Regression

📌 Overview

This project focuses on predicting gold prices using a **Linear Regression model** based on historical data from 2013 to 2023. The model uses feature engineering techniques like lag values and rolling statistics to improve prediction accuracy.

The goal is to build a reliable machine learning model that can forecast gold prices with high accuracy and low error.

---

🚀 Features

* Data preprocessing and cleaning
* Feature engineering (lag features, rolling mean, volatility)
* Linear Regression model training
* Model evaluation using multiple metrics
* Visualization of results:

  * Actual vs Predicted prices
  * Residual analysis
  * Feature importance
  * Scatter plot

---

## 📂 Dataset

* **Name**: Gold Price Dataset (2013–2023)
* **Type**: Time-series tabular data
* **Features**:

  * Date
  * Price
  * Open
  * High
  * Low

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 🧠 Machine Learning Model

* **Algorithm**: Linear Regression
* **Type**: Supervised Learning (Regression)

The model predicts gold prices using:

[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n
]

---

## 🔧 Feature Engineering

The following features are created:

* Lag features: `lag_1`, `lag_3`, `lag_7`, `lag_14`, `lag_30`
* Rolling averages: `roll_mean_7`, `roll_mean_14`, `roll_mean_30`
* Rolling standard deviation: `roll_std_7`, `roll_std_14`
* Volatility: `lag_hl_range`
* Time features: `day_of_week`, `month`, `year`

---

## 📊 Model Performance

| Metric   | Value   |
| -------- | ------- |
| MAPE     | 0.6655% |
| R² Score | 0.9515  |
| MAE      | $11.98  |
| RMSE     | $16.03  |

✅ High R² score indicates strong prediction capability
✅ Low error values indicate accurate predictions

---

## 📈 Output Visualizations

The model generates the following plots:

* Actual vs Predicted price trend
* Residual error distribution
* Feature importance chart
* Scatter plot (Actual vs Predicted)

---

## 📁 Project Structure

```
Gold-Price-Prediction/
│
├── Gold Price (2013-2023).csv
├── gold_price_prediction.py
├── gold_price_prediction.png
├── README.md
```

---

▶️ How to Run the Project

1. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

2. Run the script

```bash
python gold_price_prediction.py
```

---

Code Reference

Main implementation file:
📄 

---

📌 Results Summary

* The model successfully captures gold price trends
* Rolling mean and lag features are the most important
* Predictions closely match actual values

---

🔮 Future Improvements

* Use LSTM for better time-series prediction
* Include external economic indicators
* Hyperparameter tuning
* Deploy as a web app

---

🤝 Contributing

Feel free to fork this repository and improve the model!

---

📜 License

This project is for educational purposes.

---

👨‍💻 Author

**Deepak G**

---