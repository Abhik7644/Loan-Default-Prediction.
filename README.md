# Loan Default Prediction Model
## Project Overview
This project focuses on building and evaluating machine learning models to predict loan defaults. By analyzing various features of loan applicants, the goal is to classify applicants who are likely to default on their loans, thereby helping financial institutions minimize risk.

This project was completed as part of a summer internship with **Tata Steel**.

## Table of Contents
1.  [Problem Statement](#problem-statement)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Model Evaluation](#model-evaluation)
5.  [Key Findings](#key-findings)
6.  [How to Run the Code](#how-to-run-the-code)
7.  [Requirements](#requirements)
8.  [Project Structure](#project-structure)

## Problem Statement
Loan default poses a significant threat to financial institutions. This project addresses the challenge of predicting loan defaults using supervised binary classification. The objective is to train a machine learning model that can accurately predict which loan applicants are at a high risk of defaulting, based on their past profiles.

## Dataset
The project utilizes a dataset containing 20,000 observations and 15 variables, including the target variable `bad_loan`. The features include a mix of numerical and categorical data such as:
* `id`: Unique ID of the loan application.
* `grade`: Loan grade assigned by the institution.
* `annual_inc`: Annual income of the borrower.
* `dti`: Debt-to-Income Ratio.
* `purpose`: The stated purpose of the loan.
* `term`: The loan term in months (36 or 60).
* `bad_loan`: The target variable, where `1` indicates a defaulted loan and `0` indicates a paid loan.

## Methodology
The project follows a standard machine learning pipeline:

### 1. Exploratory Data Analysis (EDA)
* Initial data inspection to understand the shape, data types, and missing values.
* Visualization of numerical and categorical variables to identify patterns and distributions.
* Analysis of data imbalance, revealing that about 20% of loans resulted in a default.
* Correlation analysis using a heatmap to understand relationships between variables.

### 2. Data Wrangling
* **Handling Missing Values**: Missing data in columns like `home_ownership` and `dti` were imputed using the mode and mean, respectively. The `last_major_derog_none` column was dropped due to a high percentage of missing values.
* **Outlier Treatment**: Outliers in variables like `annual_inc` and `revol_util` were managed to improve model performance.
* **Feature Engineering and Encoding**: Categorical features like `grade`, `term`, `home_ownership`, and `purpose` were converted into a numerical format suitable for machine learning models. The `grade` column was mapped to an ordinal scale, and others were one-hot encoded.

### 3. Machine Learning Models
* **Handling Imbalanced Data**: The SMOTE (Synthetic Minority Over-sampling Technique) method was applied to balance the dataset, ensuring models don't get biased towards the majority class.
* **Model Training**: The preprocessed data was split into training and testing sets. Four different models were trained and evaluated:
    * Logistic Regression
    * Support Vector Machine (SVM)
    * Decision Tree Classifier
    * Random Forest Classifier
* **Hyperparameter Tuning**: For the Random Forest model, `RandomizedSearchCV` was used to find the optimal hyperparameters, further enhancing its performance.

## Model Evaluation
The performance of each model was evaluated using the **F1 score**, a metric chosen for its effectiveness in handling imbalanced datasets.

| Model                  | F1 Score (Test Data) |
| ---------------------- | -------------------- |
| Logistic Regression    | 0.70                 |
| SVM                    | 0.81                 |
| Decision Tree          | 0.82                 |
| **Random Forest** | **0.85** |

The **Random Forest Classifier** achieved the highest F1 score, making it the best-performing model for this project.

## Key Findings
* The project successfully demonstrates the application of various machine learning models for loan default prediction.
* The Random Forest model, after hyperparameter tuning, proved to be the most effective, highlighting the importance of ensemble methods for such a task.
* The F1 score was an appropriate metric for this project due to the imbalanced nature of the dataset.

## How to Run the Code
1.  Clone this repository: `git clone [Your Repository URL]`
2.  Navigate to the project directory: `cd Loan-Default-Prediction`
3.  Install the required libraries: `pip install -r requirements.txt`
4.  Run the Python script: `python Loan_Default_Prediction_Soumen_Mandal.py`

## Requirements
All the necessary libraries can be installed using the `requirements.txt` file.

## Project Structure
* `Loan_Default_Prediction_Soumen_Mandal.py`: The main script with all the code.
* `tata_dataset.csv`: The dataset used in the project.
* `README.md`: This file.
* `requirements.txt`: List of dependencies.