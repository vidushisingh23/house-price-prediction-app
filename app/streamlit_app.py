import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, scaler, and features
@st.cache_data
@st.cache_resource
def load_artifacts():
    with open("model/house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

def get_user_input():
    st.title("\U0001F3E1 House Price Prediction App")
    st.markdown("Enter house details to predict its **expected sale price** ")

    LotArea = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=20000, value=8500)
    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    GrLivArea = st.number_input("Ground Living Area (sq ft)", min_value=500, max_value=4000, value=1500)
    TotalBsmtSF = st.number_input("Total Basement SF", min_value=0, max_value=3000, value=800)
    GarageCars = st.slider("Garage Capacity", 0, 4, 2)
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
    BedroomAbvGr = st.slider("Bedrooms Above Ground", 0, 10, 3)
    GarageArea = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=400)
    YearBuilt = st.number_input("Year Built", min_value=1900, max_value=2025, value=2005)

    Neighborhood = st.selectbox("Neighborhood", [
        'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards',
        'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill',
        'NridgHt', 'NWAmes', 'OldTown', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr',
        'SWISU', 'Timber', 'Veenker'
    ])

    TotalSF = TotalBsmtSF + GrLivArea
    Age = 2025 - YearBuilt
    QualAge = OverallQual * Age

    data = {
        'LotArea': LotArea,
        'OverallQual': OverallQual,
        'GrLivArea': GrLivArea,
        'TotalBsmtSF': TotalBsmtSF,
        'GarageCars': GarageCars,
        'FullBath': FullBath,
        'BedroomAbvGr': BedroomAbvGr,
        'GarageArea': GarageArea,
        'Neighborhood': Neighborhood,
        'TotalSF': TotalSF,
        'QualAge': QualAge
    }

    return pd.DataFrame([data])

def explain_with_coefficients(model, features):
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm', ax=ax)
        ax.set_title("Top 10 Influential Features")
        st.pyplot(fig)
    else:
        st.info("Model does not support coefficient-based explanation.")

def show_correlation_heatmap(df):
    st.subheader(" Feature Correlation Heatmap (Based on Training Data)")

    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        st.warning("Training data not found at 'data/train.csv'. Please add it to show correlation heatmap.")
        return

    df = pd.read_csv(train_path)

    # Clean and prepare data
    df = df.drop(['Id'], axis=1)
    df = df.select_dtypes(include=[np.number])  # numeric features only

    # Feature Engineering for correlation (optional: mimic model)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Age'] = 2025 - df['YearBuilt']
    df['QualAge'] = df['OverallQual'] * df['Age']

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr[['SalePrice']].sort_values(by='SalePrice', ascending=False).head(15), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Top Features Correlated with SalePrice")
    st.pyplot(fig)


def main():
    model, scaler, features = load_artifacts()
    input_df = get_user_input()

    # One-hot encode and align
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    for col in features:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[features]

    # Scale and predict
    input_scaled = scaler.transform(input_df_encoded)
    log_price = model.predict(input_scaled)[0]
    price = np.expm1(log_price)

    st.success(f"Predicted House Price: ${np.round(price, 2):,.0f}")

    # Show coefficients (since it's LinearRegression)
    st.subheader("Model Explanation")
    st.markdown("See which features influenced the prediction the most:")
    explain_with_coefficients(model, features)

    # Show correlation heatmap
    show_correlation_heatmap(input_df)

if __name__ == '__main__':
    main()
