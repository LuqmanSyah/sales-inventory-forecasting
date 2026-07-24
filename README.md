# 📈 Sales & Demand Forecasting System for Inventory Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning pipeline built with **XGBoost Regressor** to predict daily retail store item demand and automate inventory restocking decisions. 

By converting time-series historical sales data into a supervised regression problem, this system accurately forecasts product demand and provides automated **Restock Recommendations** to prevent out-of-stock and overstock scenarios.

---

## 📌 Features

- **Time-Series Feature Engineering**: Extraction of temporal patterns (calendar indicators, multi-day lag sales, rolling average trends).
- **High-Performance Regression**: Trained using XGBoost to capture complex non-linear trends and seasonal patterns across multiple stores and items.
- **Out-of-Time Validation**: Strict temporal train-test split (2013–2016 for training, 2017 for evaluation) to mimic real-world forecasting scenarios.
- **Smart Restock Decision Logic**: Automated safety stock calculation to issue real-time reorder alerts based on forecasted demand.

---

## 📊 Dataset Overview

The model is trained on the **Store Item Demand Forecasting Challenge** dataset from Kaggle, which includes 5 years of daily sales data across **10 stores** and **50 products** (500 total unique time-series).

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `date` | Datetime | Date of the sale data |
| `store` | Integer | Store ID (1 to 10) |
| `item` | Integer | Product Item ID (1 to 50) |
| `sales` | Integer | Number of items sold (Target variable) |

---

## ⚙️ Feature Engineering Pipeline

Key features created from the base dataset:
* **Calendar Features**: `dayofweek`, `month`, `year`, `dayofmonth`, `weekofyear`, `is_weekend`.
* **Lag Features**: Historical sales values shifted at `lag_1`, `lag_7`, `lag_14`, and `lag_30` days.
* **Rolling Features**: Moving average trends calculated over `7-day` and `30-day` windows (`rolling_mean_7`, `rolling_mean_30`).

---

## 📈 Model Performance & Results

Evaluated on unseen test data from the entire year of **2017** (~182,500 samples):

- **Mean Absolute Error (MAE)**: `~6.12` units
- **Root Mean Squared Error (RMSE)**: `~7.95` units

> **Key Insight**: Feature importance analysis shows that `rolling_mean_7` and `sales_lag_7` are the strongest drivers for predicting daily demand, aligning with real-world weekly retail consumption cycles.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/sales-inventory-forecasting.git](https://github.com/your-username/sales-inventory-forecasting.git)
cd sales-inventory-forecasting

```

### 2. Install Dependencies

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn joblib kagglehub

```

### 3. Run the Notebook

Open `notebooks/forecasting_pipeline.ipynb` in Google Colab or Jupyter Notebook and run all cells.

---

## 💡 Restock Recommendation System Example

```python
import joblib

# Load trained model
model = joblib.load('xgboost_forecasting_model.pkl')

# Example Restock Check
check_restock_status(store_id=1, item_id=1, current_stock=100, safety_stock=20)

```

**Output:**

```text
=== LAPORAN STOK: Toko 1 | Barang 1 ===
Stok Saat Ini            : 100 unit
Estimasi Penjualan 7 Hari: 165 unit
Safety Stock Required    : 20 unit
Total Stok Dibutuhkan    : 185 unit
--------------------------------------------------
STATUS: 🚨 PERLU REORDER SEGERA! (Saran Tambahan Stok: +85 unit)

```

---

## 📂 Repository Structure

```text
├── data/                      # Raw and processed datasets (ignored in git)
├── models/                    # Saved model artifacts (.pkl)
├── notebooks/                 # Jupyter notebooks for EDA, training & evaluation
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

```

---

## 📜 License

This project is open-source under the [MIT License](https://www.google.com/search?q=LICENSE).
