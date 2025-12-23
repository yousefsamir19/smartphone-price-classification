# Smartphone Price Classification

This project was developed as part of an **Artificial Intelligence course** at **Ain Shams University (ASU)**.  
The goal of the project is to build and evaluate machine learning models that can **predict the price range of smartphones** based on their specifications.

---

## Project Overview

Smartphone pricing depends on multiple features such as RAM, battery capacity, internal storage, camera quality, and more.  
In this project, we apply **data preprocessing, feature engineering, and machine learning classification algorithms** to predict the smartphone price category.

---

## Objectives

- Understand the full **machine learning pipeline**
- Handle **data cleaning and preprocessing**
- Avoid **data leakage**
- Train and evaluate different ML models
- Analyze model performance using **classification reports and confusion matrices**

---

## Dataset

- The dataset contains smartphone specifications and a target label representing the **price range**.
- Target variable: `price`
- Features include hardware and technical specifications.

> Note: All preprocessing steps were applied **after splitting the data** to avoid data leakage.

---

## Technologies & Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Models Used

- Decision Tree Classifier
- SVM Classifier
- XGBoost Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix (visualized using heatmaps)

---

## Deployment

The application was deployed using Streamlit Community Cloud.

---

## Dependencies

All dependencies are listed in `requirements.txt` and optimized for cloud deployment.

---

## Live Demo

👉 **[Click here to try the app](https://smartphone-price-classification-o8zqxjxe2gvrmqobkahds5.streamlit.app/)**
