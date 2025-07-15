# Predictive Financial Modeling for Software Project Success

 A comparative analysis of traditional estimation methods like **COCOMO** versus modern **machine learning models** to predict the success of software projects.

---

##  Project Overview

This research project evaluates and contrasts:
- Traditional software estimation using the **COCOMO model**
- Machine learning models including:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes

Our objective is to predict **project success/failure** using financial, schedule, and development metrics and to offer a novel hybrid decision framework to resolve conflicts in model predictions.

---

##  Repository Contents

- `SEE_PROJECT FINAL.ipynb` – Jupyter Notebook with code, ML model training, evaluation, and visualizations.
- `see research paper.pdf` – A detailed research paper covering methodology, background, results, and insights.
- `finaldataset.xlsx` – Dataset used for training and evaluating the ML models
- `project_success_analysis.xlsx` – Output file containing results from COCOMO calculations, ML model predictions, and profitability analysis.


---
##  Project Workflow Summary

The notebook processes software project data using the following input features:

- **Actual Budget Spent**  
- **Cost Overrun**  
- **Actual Duration**  
- **Schedule Overrun**  
- **Revenue Generated**  
- **Estimated Duration**  
- **Lines of Code (LOC)**

It then calculates:

- **Profit** = Revenue − Actual Budget  
- **Profitability** = (Profit / Budget) × 100

Two prediction paths are implemented:

1. **COCOMO-Based Estimation**  
   - LOC is used to determine project mode (e.g., Semi-detached for 50 < KLOC ≤ 300)  

2. **Machine Learning Models**  
   - ML models (Logistic Regression, SVM, KNN, Naive Bayes) are applied to the same dataset  
   - Predictions, probabilities, and model performance metrics are calculated

 **All results are saved in a separate Excel file named** `project_success_analysis.xlsx` for review and comparison.

---
##  Key Findings

- The **SVM model** achieved the highest accuracy: **97%**
- All ML models outperformed traditional **COCOMO** (which had only **76% accuracy**)
- Financial indicators (e.g., revenue, cost overrun) were more predictive than technical metrics like lines of code
- A custom **hybrid framework** was proposed to handle conflicts between ML and COCOMO predictions

---

##  How to Run This Project

### Clone the Repository
```bash
git clone https://github.com/rahimakk/predictive-financial-modelling-.git
cd predictive-financial-modelling-


###  Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn


###  Launch the Notebook

You can open the `.ipynb` notebook in any of the following platforms:

- **Jupyter Notebook** (locally on your machine)
- **Google Colab** (recommended for cloud-based execution)
- **Kaggle Notebooks** (if dataset is available there)

---

##  Technologies Used

- **Python**
- **Jupyter Notebook**
- **Scikit-learn**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- Manual implementation of the **COCOMO** cost estimation model

---

##  About the Research

The attached research paper (`see research paper.pdf`) includes:

-  **Background** on project estimation techniques  
-  **Implementation** of machine learning models and COCOMO  
-  **Comparative performance evaluation**  
-  **Insights** from feature importance and exploratory data analysis  
-  A **hybrid decision framework** to resolve prediction disagreements

---
