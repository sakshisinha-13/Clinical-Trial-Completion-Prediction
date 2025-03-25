# Clinical-Trial-Completion-Prediction
This project predicts whether a clinical trial will be **completed or not completed** using both structured and unstructured data. It was developed as part of the **NEST Hackathon 2024** (Team: BitByters), focusing on handling **class imbalance**, improving **model performance**, and ensuring **explainability** with SHAP.

---

## Problem Statement

Predict the completion status (`Completed` or `Not Completed`) of clinical trials using features like:

- **Structured**: Enrollment numbers, study type, funder type, phases
- **Unstructured**: Study title, conditions, brief summaries, interventions

---

## Project Highlights

- 📊 **Data Preprocessing**:
  - Cleaned and standardized the “Study Status” column
  - Merged text fields and applied **TF-IDF** (`max_features=500`) for unstructured data
  - Applied **OneHotEncoding** for categorical variables and passed through numeric fields

  -**Modeling & Tuning**:
  - Built a classification pipeline using **XGBoost**
  - Used **RandomizedSearchCV** for hyperparameter tuning
  - Handled class imbalance with **scale_pos_weight**
  - Applied **threshold tuning** for better recall on minority class

- **Performance Metrics**:
  - **Accuracy**: 91%
  - **ROC AUC**: 0.8514
  - **Recall (Completed)**: 98%, **F1 Score**: 0.95
  - **Recall (Not Completed)**: 46%, **F1 Score**: 0.58

- **Explainability**:
  - Incorporated **SHAP** to interpret model predictions
  - Identified key features like `Enrollment`, `Study Type`, and text tokens affecting outcomes

---

## Key Learnings

- Importance of handling **class imbalance** in real-world datasets
- Leveraging **text and numeric features** together with a robust ML pipeline
- Applying **model explainability** for healthcare-focused predictions
- Balancing **performance** and **interpretability**

---

## Tech Stack

- Python 3.9
- XGBoost
- Scikit-learn
- Pandas, NumPy
- SHAP (Explainability)
- TF-IDF (Text Feature Engineering)
- Matplotlib (Visualizations)

---
