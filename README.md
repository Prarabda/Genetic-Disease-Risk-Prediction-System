# Genetic-Disease-Risk-Prediction-System

An end-to-end data analytics and machine learning project that predicts genetic disorders using patient clinical and hereditary information. The project demonstrates the complete data analytics lifecycle, including ELT, Exploratory Data Analysis (EDA), Data Warehousing, OLAP operations, Machine Learning, and predictive analytics.

---

## 📌 Project Overview

Genetic disorders can often be identified through hereditary, demographic, and clinical indicators. This project applies Machine Learning techniques to predict genetic disorders while also demonstrating data warehouse design and OLAP analysis for healthcare data.

The project combines data engineering and data science concepts into a single workflow suitable for academic and practical learning.

---

## 🚀 Features

- Data Extraction, Transformation, and Loading (ELT)
- Missing value handling
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Star Schema Data Warehouse Design
- SQLite Data Warehouse
- OLAP Operations
  - Roll-up
  - Drill-down
  - Slice
  - Dice
- Decision Tree Machine Learning Model
- Cross Validation
- Model Evaluation
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Interactive Patient Disease Prediction

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SQLite

---

## 🔄 Project Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning & ELT
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Warehouse (Star Schema)
      │
      ▼
SQLite Database
      │
      ▼
OLAP Operations
      │
      ▼
Machine Learning
      │
      ▼
Model Evaluation
      │
      ▼
Patient Disease Prediction
```

---

## 📊 Data Warehouse Design

The project uses a **Star Schema** consisting of:

### Fact Table

- Patient_ID
- Disease_ID
- Gene_ID
- Hospital_ID
- Family_ID
- Blood_Cell_Count
- White_Blood_Cell_Count
- Respiratory_Rate
- Heart_Rate

### Dimension Tables

- Patient
- Disease
- Gene
- Hospital
- Family

---

## 🤖 Machine Learning

### Model Used

- Decision Tree Classifier

### Features Used

- Patient Age
- Gender
- Maternal Gene
- Paternal Gene
- Genes in Mother's Side
- Inherited from Father
- Blood Cell Count
- Respiratory Rate
- Heart Rate
- Birth Defects
- Symptom 1–5

### Target

- Genetic Disorder

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation
- Confusion Matrix

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Genetic-Disease-Risk-Prediction.git
cd Genetic-Disease-Risk-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python predicting_genetic_disease_risk.py
```

---

## 📷 Outputs

The project generates:

- Data Cleaning Reports
- Statistical Summaries
- EDA Visualizations
- Correlation Heatmap
- Star Schema Warehouse
- SQLite Warehouse Tables
- OLAP Reports
- Decision Tree Visualization
- Confusion Matrix
- Disease Prediction Results

---

## 🎯 Learning Outcomes

This project demonstrates:

- Data Engineering
- Data Warehousing
- ETL/ELT
- SQL & SQLite
- OLAP Analysis
- Exploratory Data Analysis
- Machine Learning
- Healthcare Data Analytics

---

## 👨‍💻 Author

**Prarabda Singh Mahat**

Bachelor in Business Information Systems (BBIS)

Kathmandu University

---

## 📜 License

This project is intended for educational and research purposes.
