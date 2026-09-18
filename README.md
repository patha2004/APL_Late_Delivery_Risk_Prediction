# APL Late Delivery Risk Prediction

## 📌 Project Overview

APL Late Delivery Risk Prediction is an end-to-end Machine Learning project that predicts whether an order is at risk of being delivered late.

The project uses historical logistics and supply-chain data to identify patterns associated with delivery delays. A Random Forest Classification model is trained and integrated into a Streamlit web application for real-time prediction.

---

## 🎯 Problem Statement

Late deliveries can negatively affect customer satisfaction, operational efficiency, and supply-chain performance.

The objective of this project is to build a machine learning system that can predict the risk of late delivery using information available before the delivery outcome.

The prediction can help logistics teams identify potentially risky shipments and take preventive actions.

---

## 🎯 Objectives

- Analyze historical logistics and order data.
- Perform data cleaning and exploratory data analysis.
- Identify important factors associated with late deliveries.
- Build machine learning classification models.
- Compare different classification models.
- Select the best-performing model.
- Deploy the prediction system using Streamlit.
- Classify shipments into Low, Medium, and High risk based on predicted probability.

---

## 📊 Dataset

The project uses the APL Logistics dataset containing approximately:

- **180,519 records**
- **40 original features**
- Target variable: `Late_delivery_risk`

### Target Variable

`Late_delivery_risk`

- `0` → Not at risk of late delivery
- `1` → At risk of late delivery

The dataset contains information related to customers, products, orders, markets, shipping methods, and scheduled shipment time.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Loaded the CSV dataset using appropriate encoding.
- Checked dataset dimensions and data types.
- Checked missing values.
- Checked duplicate records.
- Removed rows containing missing values.
- Analyzed the target variable distribution.
- Performed categorical and numerical feature analysis.
- Identified potential data leakage.
- Excluded post-delivery information from model training.

### Important Data Leakage Prevention

Features such as:

- `Days for shipping (real)`
- `Delivery Status`

were excluded because they contain information that becomes available only after or during the delivery process.

This ensures that the model focuses on information that can reasonably be available before delivery.

---

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand relationships between shipment characteristics and late-delivery risk.

The analysis included:

- Late delivery risk by shipping mode
- Late delivery risk by scheduled shipment days
- Late delivery risk by market
- Numerical feature analysis
- Categorical feature analysis
- Target distribution analysis
- Feature importance analysis

---

## 🛠️ Machine Learning Features

The final model uses the following 14 input features:

1. Type
2. Days for shipment (scheduled)
3. Benefit per order
4. Sales per customer
5. Category Name
6. Customer Segment
7. Customer Country
8. Customer State
9. Department Name
10. Market
11. Order Country
12. Order Region
13. Product Price
14. Shipping Mode

Categorical variables were transformed using One-Hot Encoding.

The preprocessing pipeline generated **313 encoded features** from the selected input variables.

---

## 🤖 Machine Learning Models

Two baseline classification models were evaluated:

### Logistic Regression

- Accuracy: **68.97%**
- Precision: **79.18%**
- Recall: **58.88%**
- F1 Score: **67.54%**
- ROC-AUC: **0.7264**

### Random Forest

- Accuracy: **71.28%**
- Precision: **76.39%**
- Recall: **68.93%**
- F1 Score: **72.47%**
- ROC-AUC: **0.7851**

### Tuned Random Forest

A controlled tuning experiment was also performed, but its performance was lower than the original Random Forest model.

- Accuracy: **69.92%**
- Precision: **82.65%**
- Recall: **57.13%**
- F1 Score: **67.56%**
- ROC-AUC: **0.7645**

---

## 🏆 Final Model

The **Random Forest Classifier** was selected as the final model because it provided the best overall performance among the evaluated models.

### Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 71.28% |
| Precision | 76.39% |
| Recall | 68.93% |
| F1 Score | 72.47% |
| ROC-AUC | 0.7851 |

The Random Forest model achieved a better balance between precision and recall and produced the highest ROC-AUC among the tested models.

---

## ⭐ Feature Importance

Feature importance analysis was performed using the Random Forest model.

Some of the important features included:

- Benefit per order
- Sales per customer
- Days for shipment (scheduled)
- Shipping-related variables
- Other order and customer characteristics

Feature importance helps understand which variables contribute most to the model's predictions.

---

## 🌐 Streamlit Application

The trained model was integrated into a Streamlit web application.

The application allows users to enter shipment and order information and receive a late-delivery risk prediction.

### Application Features

- User-friendly input form
- Real-time prediction
- Late-delivery probability
- Risk classification
- Model information section
- Three-level risk visualization

### Risk Classification

| Probability | Risk Level |
|---|---|
| < 40% | 🟢 Low Risk |
| 40% – 60% | 🟡 Medium Risk |
| > 60% | 🔴 High Risk |

The three risk levels are a presentation layer based on the predicted probability. The underlying machine learning model remains a binary classification model.

---

## 📁 Project Structure

```text
APL_Late_Delivery_Risk_Prediction/
│
├── app/
│   └── app.py
│
├── Data/
│   └── APL_Logistics.csv
│
├── models/
│   ├── random_forest_model.pkl
│   ├── preprocessor.pkl
│   └── feature_info.pkl
│
├── notebooks/
│   └── 01_Data_Understanding_EDA.ipynb
│
├── reports/
│
├── src/
│
├── README.md
└── requirements.txt

Technologies Used

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Streamlit
Jupyter Notebook

How to Run the Project

1. Clone the repository
git clone <your-github-repository-url>
2. Navigate to the project directory
cd APL_Late_Delivery_Risk_Prediction
3. Install required libraries
pip install -r requirements.txt
4. Run the Streamlit application
python -m streamlit run app/app.py

The application will open in your browser.

🔮 Future Improvements

Future versions of the project could include:

Hyperparameter optimization
Advanced ensemble models
Model explainability using SHAP
Real-time logistics data integration
Automated model retraining
Cloud deployment
Monitoring model performance over time
More detailed business recommendations based on risk level


✅ Conclusion

This project demonstrates an end-to-end Machine Learning workflow for predicting late delivery risk.

It covers:

Data Collection → Data Cleaning → EDA → Feature Selection → Preprocessing → Model Training → Model Evaluation → Model Saving → Streamlit Deployment

The final Random Forest model achieved an ROC-AUC of 0.7851 and was successfully integrated into an interactive Streamlit application capable of displaying Low, Medium, and High delivery-risk levels.

👨‍💻 Project

APL Late Delivery Risk Prediction

Machine Learning Classification Project


### Step 3

Press:

**Ctrl + S**

That's it for now. ✅

After saving, **send me a screenshot of the README**. Then we'll move to the next important part: **final project cleanup + screenshots for your submission**. 🚀

