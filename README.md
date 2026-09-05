# Loan Approval Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a loan application will be **approved or rejected** using machine learning classification algorithms.

The project compares multiple models and identifies the best-performing model based on accuracy, recall, and F1 score.

---

## 🎯 Objectives

* Predict loan approval using applicant and loan-related information.
* Preprocess and prepare the dataset for machine learning.
* Engineer useful features.
* Train and compare different classification models.
* Select the best-performing model.

---

## 📊 Dataset & Features

The dataset contains **614 loan applications** with the following features:

| Feature             | Description                            |
| ------------------- | -------------------------------------- |
| `Gender`            | Gender of the applicant                |
| `Married`           | Applicant's marital status             |
| `Dependents`        | Number of dependents                   |
| `Education`         | Education level                        |
| `Self_Employed`     | Whether the applicant is self-employed |
| `ApplicantIncome`   | Applicant's income                     |
| `CoapplicantIncome` | Co-applicant's income                  |
| `LoanAmount`        | Requested loan amount                  |
| `Loan_Amount_Term`  | Loan repayment period                  |
| `Credit_History`    | Applicant's credit history             |
| `Property_Area`     | Property location type                 |
| `Loan_Status`       | Target variable — Approved or Rejected |

`Loan_ID` is removed because it is only an identifier and does not provide useful predictive information.

---

## 🧹 Data Preprocessing

Missing values are handled using appropriate **mode or median imputation**. Categorical values are converted into numerical form using **encoding**, while numerical features are standardized using **StandardScaler**. Additional features such as **TotalIncome, EMI, and Income_to_Loan** are created to improve the model. The target variable is encoded as `0` for rejected and `1` for approved loans. Finally, the dataset is split into **80% training and 20% testing data**.

---

## 🤖 Machine Learning Models

The following models are trained and compared:

* **Logistic Regression**
* **Random Forest**
* **XGBoost**
* **Stacking Classifier**

The Stacking Classifier combines Logistic Regression, Random Forest, and XGBoost.

---

## 📈 Model Comparison

| Model                   |   Accuracy |     Recall |   F1 Score |
| ----------------------- | ---------: | ---------: | ---------: |
| **Logistic Regression** | **86.18%** | **98.82%** | **0.9081** |
| Random Forest           |     83.74% |     91.76% |     0.8864 |
| XGBoost                 |     81.30% |     92.94% |     0.8729 |
| Stacking Classifier     |     81.30% |     92.94% |     0.8729 |

---

## 🏆 Conclusion

**Logistic Regression** achieved the best overall performance with:

* **Accuracy:** 86.18%
* **Recall:** 98.82%
* **F1 Score:** 0.9081

Therefore, Logistic Regression is selected as the best-performing model for this loan approval prediction task.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── dataset.csv
├── loan_approval_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn xgboost jupyter
```

---

## ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run the cells sequentially to reproduce the results.
