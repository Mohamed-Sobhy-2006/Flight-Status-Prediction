# ✈️ Flight Status Prediction App

**Production-ready Machine Learning application** for predicting flight delays, featuring advanced preprocessing, feature engineering, and deployment using Streamlit.

---

## 🚀 Features
- Real-time flight delay prediction
- Automatic distance calculation between cities
- Feature engineering:
  - Seasonal features (Winter, Spring, Summer, Autumn)
  - Weekend detection
  - Cyclical encoding for day-of-week and month
- Encoding techniques:
  - OneHot Encoding for categorical variables
  - Target Encoding for origin and destination cities
- Scaling:
  - StandardScaler for numeric features
  - RobustScaler for distance
- Model:
  - XGBoost classifier for high accuracy
- Interactive web interface built with Streamlit

---

## 🛠 Technologies Used
- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Category Encoders

---

## 📂 File Structure

```text
Flight-Status-Prediction/
│
├── app.py                   # Streamlit app
├── xgb_model.pkl            # Trained XGBoost model
├── one_hot_encoder.pkl      # OneHotEncoder for categorical features
├── target_encoder.pkl       # TargetEncoder for city features
├── standard_scaler.pkl      # StandardScaler for numerical features
├── robust_scaler.pkl        # RobustScaler for numerical features
├── distances_symmetric.csv  # Distance data between airports
├── requirements.txt         # Python dependencies
└── README.md                # This file
```
## ▶️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Mohamed-Sobhy-2006/Flight-Status-Prediction.git
cd Flight-Status-Prediction

#Run the Streamlit app
streamlit run app.py

```
👨‍💻 Author

Mohamed Sobhy     

 AI & Machine Learning Enthusiast
```
🔗 Demo / Portfolio

Add your hosted Streamlit app link here once deployed to showcase the live project.
```
⚡ Notes

Make sure all .pkl files are present for the app to work correctly.

If some files are too large for GitHub, provide download links in the README.

Predicts whether a flight is ON TIME ✅ or DELAYED 🚨 based on historical data and features. 
