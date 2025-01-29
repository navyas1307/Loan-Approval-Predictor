# Loan Approval System

This project is a comprehensive Loan Approval System that utilizes machine learning for predicting loan approvals. It includes multiple components such as data preprocessing, machine learning model training, EMI calculation, and interactive visualizations.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [About Data](#about-data)
- [Project Structure](#project-structure)
- [Files Overview](#files-overview)
- [Loan Features](#loan-features)
- [Conclusion](#conclusion)


---

## Introduction

This Loan Approval System helps predict loan approval decisions based on various applicant features such as income, credit score, and loan term. It integrates:

1. **Machine Learning Model**: A Random Forest Classifier trained to predict loan approval.
2. **EMI Calculator**: A web-based tool to calculate Equated Monthly Installments (EMI).
3. **Data Visualizations**: Insights into the factors influencing loan approvals, displayed using heatmaps and feature importance graphs.

---

## Features

1. Loan approval prediction based on applicant details.
2. EMI calculation for different loan types.
3. Interactive visualizations showcasing feature correlations and importance.

---

## Technologies Used

- **Python**: For data processing and machine learning.
- **Scikit-learn**: For building and optimizing the Random Forest model.
- **Imbalanced-learn**: To handle class imbalance using SMOTEENN.
- **Joblib**: For saving and loading the trained model and preprocessing objects.
- **HTML/CSS/JavaScript**: For the frontend EMI calculator and visualizations.
- **Tailwind CSS**: For styling the EMI Calculator.

---

## About Data

- This project uses the **loan_approval_dataset** obtained from Kaggle. The dataset contains information on various loan applicants and includes key features such as applicant income, debt-to-income ratio, loan amount, term, and other factors that influence loan approval.
- The data is used to train and evaluate a machine learning model that predicts loan approval based on these features.
- You can access the dataset on [Kaggle's loan approval dataset page](https://www.kaggle.com/datasets).


## Project Structure

```
loan-approval-system/
├── app.py                     # Main application file for running the backend
├── loan_data9.py              # Script for data preprocessing and ML model training
├── templates/                 # Templates for rendering HTML pages
│   ├── index.html             # Landing page for the loan prediction system
│   ├── loan_prediction.html   # Loan prediction form
│   ├── emi.html               # EMI Calculator interface
│   └── visualizations.html    # Visualization dashboard
├── static/                    # Static files for the web interface
│   ├── styles.css             # Stylesheet for the application
│   ├── script1.js             # JavaScript for interactivity
│   ├── heatmap.html           # Correlation heatmap visualization
│   ├── feature_importance.html # Feature importance visualization
│   └── All-You-Need-to-Know-About-the-Four-Levels-of-Digital-Lending.jpg"            
├── best_rf_model.pkl          # Trained Random Forest model
├── scaler.pkl                 # Scaler object for data normalization
├── feature_names.pkl          # Feature names used in the model
├── label_encoders.pkl         # Label encoders for categorical variables
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation


```

---

## Files Overview

### **1. Machine Learning Files**
- **`loan_data9.py`**:
  - Loads, preprocesses, and encodes loan dataset.
  - Performs feature engineering (e.g., debt-to-income ratio, asset ratios).
  - Trains and tunes a Random Forest Classifier.
  - Saves the trained model, scaler, and encoders for future use.

### **2. EMI Calculator**
- **`emi.html`**:
  - Provides a simple and clean UI for EMI calculation using TailwindCSS.
  - Allows users to select loan type, input loan amount, and term.
  - Displays the calculated EMI dynamically.

### **3. Data Visualizations**
- **`visualizations.html`**:
  - Explains key factors affecting loan approval with icons and descriptions.
  - Embeds:
    - **Heatmap**: Correlation between loan features.
    - **Feature Importance Chart**: Key determinants of loan approval.
- **`static/heatmap.html`**: Displays the correlation heatmap.
- **`static/feature_importance.html`**: Displays the feature importance chart.

---

## Loan Features

The following features play a significant role in determining loan approvals:

1. **CIBIL Score**: Measures the applicant's creditworthiness. Higher scores increase the likelihood of loan approval.
2. **Income Level**: Higher income levels indicate better repayment capacity.
3. **Loan Amount**: Larger loan amounts require higher income or collateral for approval.
4. **Debt-to-Income Ratio**: A lower ratio signifies better financial stability, positively influencing approval chances.
5. **Education Level**: Applicants with higher education levels are often deemed more employable and financially stable.
6. **Marital Status**: Married applicants may have dual income, improving repayment capacity.
7. **Dependents**: Fewer dependents suggest higher disposable income.
8. **Self-Employment**: Self-employed individuals might face stricter scrutiny due to income variability.
9. **Loan Term**: Longer loan terms reduce monthly EMIs but may increase total interest payable.
10. **Luxury Assets**: High-value assets can serve as collateral, enhancing approval chances.
11. **Residential Assets**: Owning residential properties indicates financial stability.
12. **Commercial Assets**: Commercial properties can be used as collateral to secure larger loans.

---



## Conclusion

This Loan Approval System demonstrates the importance of data-driven decision-making in financial applications. By analyzing applicant features such as income, credit score, and assets, the system predicts loan approval outcomes with high accuracy. The inclusion of EMI calculations and visualizations enhances user experience and transparency.

Future improvements could include:

1. Incorporating more advanced machine learning models.
2. Adding real-time data integration for live predictions.
3. Expanding the feature set to include external economic factors.
