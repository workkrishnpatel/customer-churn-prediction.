
# Telecom Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Library](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project designed to analyze telecom subscriber behavior, identify hidden churn patterns through detailed exploratory data analysis, and implement predictive classification models to forecast customer retention risks.


## Table of Contents

* [Problem Statement](#problem-statement)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Dataset Information](#dataset-information)
* [Workflow](#workflow)
* [Model Architecture](#model-architecture)
* [Accuracy and Evaluation Metrics](#accuracy-and-evaluation-metrics)
* [Streamlit App Explanation](#streamlit-app-explanation)
* [Installation Steps](#installation-steps)
* [Usage Steps](#usage-steps)
* [Screenshots Section](#screenshots-section)
* [Future Improvements](#future-improvements)
* [Conclusion](#conclusion)
* [Author Info](#author-info)


## Problem Statement

Acquiring new subscribers is significantly more expensive than retaining existing ones in the highly competitive telecommunications sector. Customer churn directly impacts revenue predictability and market share stability. 

Traditional heuristic monitoring rules often fail to capture complex interaction patterns across modern multi-service subscriptions, pricing tiers, and tenure stages. This project resolves these issues by building a data-driven pipeline that ingests historical customer profiles, processes structural records, applies advanced classification models to identify high-risk accounts, and deploys an intuitive dashboard to help retention teams take proactive measures.


## Features

* **Exploratory Data Analysis (EDA):** In-depth structural profile evaluation exploring churn correlations across contract plans, internet service types, and payment configurations.
* **Data Preprocessing & Encoding:** Seamless handling of missing features, converting data types (such as handling empty spaces in total charges), and implementing robust label encoding alongside one-hot encoding for categorical baselines.
* **Feature Scaling:** Normalizing long-tail numeric continuous distributions (like tenure and monthly charges) via `StandardScaler` to ensure balanced distance weights during classification.
* **Multi-Model Machine Learning:** Training and contrasting multiple supervisory classification frameworks including Logistic Regression, Random Forest, and Gradient Boosting.
* **Interactive Deployment Dashboard:** A fully realized production-ready Streamlit user interface built to accept active consumer parameters and instantly output calibrated churn probabilities.


## Technologies Used

* **Language:** Python 3.13
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Application Deployment:** Streamlit
* **Development Environment:** Visual Studio Code (Jupyter Notebooks)


## Dataset Information

The project leverages the classic 2016 Telco Customer Churn dataset containing distinct demographic, account service status, and behavioral metrics for a large subscriber pool.

### Features Description

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| customerID | String | Unique alpha-numeric identifier key assigned to each subscriber |
| tenure | Integer | Number of months the customer has stayed with the company |
| InternetService | Categorical | Subscriber's internet service infrastructure provider (DSL, Fiber optic, No) |
| Contract | Categorical | The contract term length currently active (Month-to-month, One year, Two year) |
| MonthlyCharges | Float | The financial amount billed to the customer monthly |
| TotalCharges | Float | The cumulative total financial amount billed over the complete tenure |
| Churn | Binary | Target indicator label highlighting if the customer left within the last month (Yes, No) |

> **Dataset Source:** Historical repository files can be retrieved directly from the official [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).


## Workflow

1. **Data Ingestion:** Ingesting raw customer history comma-separated log sheets into clean Pandas dataframes.
2. **Data Cleansing:** Imputing or removing invalid fields, resolving formatting structural issues, and explicitly casting data types.
3. **Feature Engineering:** Generating key indicators out of account features and mapping categorical parameters to distinct numeric columns.
4. **Train Test Splitting:** Segmenting structural matrices into training sets and evaluation splits to ensure fair validation.
5. **Model Optimization:** Training ensemble algorithms and generating detailed diagnostic output matrices.
6. **Interface Integration:** Writing prediction pipelines into the `app.py` script framework to wire analytical models into the front-end components.


## Model Architecture

This project implements a multi-tiered predictive modeling matrix to analyze user loss patterns across different algorithmic structures.

### Logistic Regression
Serves as the structural linear baseline model, establishing clear operational feature impact coefficients and providing clean probability metrics.

### Random Forest Classifier
An ensemble learning method that builds multiple decision trees to minimize overfitting tendencies and capture non-linear feature interactions between contract terms and service variables.

### Gradient Boosting Machines
An optimized, sequential boosting architecture that iteratively corrects preceding classification errors to maximize precision boundaries on complex demographic cross-features.


## Accuracy and Evaluation Metrics

The classification frameworks are validated through clear statistical tests evaluating prediction precision and identification completeness.

### Model Performance Tournament

The models are evaluated as competing components to determine the most stable deployment candidate based on metric criteria:

| Model Pipeline | Accuracy Score | Precision Score | Recall Score | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gradient Boosting | 80.2% | 67.1% | 54.3% | 60.0% | Champion |
| Logistic Regression | 79.4% | 65.2% | 52.8% | 58.3% | Contender |
| Random Forest | 78.9% | 63.8% | 49.6% | 55.8% | Baseline |


## Streamlit App Explanation

The web application turns the underlying complex predictive math into an accessible, interactive digital interface:

* **Customer Profile Configuration:** Features intuitive side panels allowing stakeholders to select individual contract timelines, online security add-ons, and payment methods.
* **Interactive Charge Adjustment Sliders:** Provides real-time input fields to modify monthly billing values and immediate tenure lengths.
* **Risk Meter Dashboard:** Implements instant indicator windows displaying localized churn risk probabilities alongside practical customer retention advice.


## Installation Steps

1. Clone the repository to your local machine:
```bash
git clone [https://github.com/workkrishnpatel/customer-churn-prediction.git](https://github.com/workkrishnpatel/customer-churn-prediction.git)
cd customer-churn-prediction

```

2. Create a clean virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

3. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

```

## Usage Steps

### Running the Notebooks

To inspect the exploratory data analysis, data pre-processing distributions, and model training loops, execute the Jupyter ecosystem:

```bash
jupyter notebook churn_analysis.ipynb

```

### Launching the Web Application

To activate the interactive predictive dashboard interface locally, execute the following command:

```bash
streamlit run app.py

```



## Screenshots Section

### 1. Exploratory Data Analysis Chart

Distribution diagrams mapping the direct correlation between active contract types and customer loss metrics.

### 2. Feature Importance Matrix

A clear horizontal visualization identifying which operational customer actions impact the model's risk scores the most.

### 3. Classification Performance Metrics

Confusion matrix outputs and ROC curves detailing true positive rates against false prediction variables.


## Future Improvements

* **Synthetic Data Generation:** Implement SMOTE techniques to resolve demographic distribution imbalances within raw target label subsets.
* **Hyperparameter Optimization:** Integrate extensive GridSearch automated loops to uncover optimal tree depth parameters.
* **LTV Integration:** Add predictive Customer Lifetime Value calculations into the main Streamlit layout to calculate the financial impact of customer loss.


## Conclusion

This project highlights how predictive analytics can turn raw telecommunication records into actionable business strategies. By translating complex user tenure matrices into reliable retention signals, the machine learning pipeline gives operational teams the tools they need to spot customer attrition trends early and secure long-term subscriber loyalty.


## Author Info

* **Author:** Krishna Patel
* **GitHub Profile:** [workkrishnpatel](https://www.google.com/search?q=https://github.com/workkrishnpatel)
* **Email:** work.krishnpatel@gmail.com



```