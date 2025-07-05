# House Price Prediction App

This is a **Streamlit web application** that predicts house prices based on user inputs using a **trained Linear Regression model**. It uses the **Ames Housing Dataset** from Kaggle and provides interactive visualizations and model explanations.

---

##  Features

- Predict house prices using key input features
- Visual explanation of top influencing features (Linear Regression coefficients)
- Feature correlation heatmap


---

##  Live Demo


*https://house-price-prediction-app-vidushi.streamlit.app/*

---

## Project Structure

```
house-price-app/
│
├── app/
│   └── streamlit_app.py         # Streamlit web app
│
├── model/
│   ├── house_price_model.pkl    # Trained Linear Regression model
│   ├── scaler.pkl               # Scaler used for features
│   └── features.pkl             # Feature columns used by the model
│
├── data/
│   ├── train.csv                # Raw training dataset from Kaggle
│   └── test.csv                 # Raw test dataset from Kaggle
│
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/house-price-app.git
cd house-price-app
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

- On Windows:
  ```bash
  .venv\Scripts\activate
  ```

- On macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Model Details

- **Algorithm**: Linear Regression (`sklearn.linear_model`)
- **Target variable**: `SalePrice` (log-transformed)
- **Input features**:
  - Selected numeric and categorical features from the Ames Housing Dataset
  - Engineered features: `TotalSF`, `Age`, and `QualAge`
- **Preprocessing**:
  - Missing value imputation (median for numeric, mode for categorical)
  - One-hot encoding of categorical variables
  - Standard scaling
- **Evaluation**:
  - Mean Squared Error (MSE)
  - R² Score

---

##  Dependencies

Some key libraries used:

- `streamlit`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `pickle`

---

##  Author

**Vidushi Singh**  
GitHub: [@vidushisingh23](https://github.com/vidushisingh23)

---
