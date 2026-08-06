
# **Predicting Genetic Disease Risk Using Machine Learning and Data Warehouse Techniques**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns
pd.set_option('display.max_columns', None)

"""# **ELT Process**

## **Extract**
"""

#Loading dataset
file_name = "genetic_disease_dataset.csv"

df = pd.read_csv(file_name)

print("Dataset Loaded Successfully!")
# First 5 rows
display(df.head())

# Last 5 rows
display(df.tail())

"""## **Transform**"""

#Inspect Dataset
print("Dataset Shape:", df.shape)

df.info()

print("\nData Types:")
print(df.dtypes)

#checking Missing Value
print(df.isnull().sum())

# Missing value percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage)

#Handle Missing Values
# Fill numerical columns with median
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for column in numeric_columns:
    df[column].fillna(df[column].median(), inplace=True)

# Fill categorical columns with mode
categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

print("Missing values handled successfully.")

#Check Duplicate Records

duplicates = df.duplicated().sum()

print("Duplicate Rows:", duplicates)

#Remove Extra Spaces from Text Columns
text_columns = df.select_dtypes(include='object').columns

for column in text_columns:
    df[column] = df[column].str.strip()

print("Extra spaces removed.")

#Display Summanry
display(df.describe(include='all'))

print("Dataset Shape:", df.shape)

display(df.head())

"""## **LOAD**"""

#saving cleaned Dataset
df.to_csv("genetic_disease_dataset_cleaned.csv", index=False)

print("Cleaned dataset saved successfully.")

"""# **Exploratory Data Analysis (EDA)**"""

import matplotlib.pyplot as plt
import seaborn as sns

# Plot style
sns.set_style("whitegrid")

# Figure size
plt.rcParams['figure.figsize'] = (8,5)

#Visualizing the genetic Disorder Distribution
plt.figure(figsize=(10,5))

sns.countplot(
    x='Genetic Disorder',
    data=df,
    order=df['Genetic Disorder'].value_counts().index
)

plt.title("Distribution of Genetic Disorders")
plt.xlabel("Genetic Disorder")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)

plt.show()

#gender distribution

plt.figure(figsize=(6,5))

sns.countplot(x='Gender', data=df)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

#age distribution

plt.figure(figsize=(8,5))

sns.histplot(df['Patient Age'], bins=20, kde=True, color='skyblue')

plt.title("Patient Age Distribution")
plt.xlabel("Patient Age")
plt.ylabel("Frequency")

plt.show()

#Gene Mutation Frequency


#Mother's Side
plt.figure(figsize=(6,5))

sns.countplot(x="Genes in mother's side", data=df)

plt.title("Genes in Mother's Side")
plt.xlabel("Genes in Mother's Side")
plt.ylabel("Count")

plt.show()


#Maternal Gene
plt.figure(figsize=(6,5))

sns.countplot(x="Maternal gene", data=df)

plt.title("Maternal Gene Distribution")
plt.xlabel("Maternal Gene")
plt.ylabel("Count")

plt.show()

#From Father

plt.figure(figsize=(6,5))

sns.countplot(x="Inherited from father", data=df)

plt.title("Inherited from Father")
plt.xlabel("Inherited from Father")
plt.ylabel("Count")

plt.show()

#Family History Analysis

plt.figure(figsize=(7,5))

sns.countplot(
    x="History of anomalies in previous pregnancies",
    data=df
)

plt.title("Family History Analysis")
plt.xlabel("History of Anomalies")
plt.ylabel("Count")

plt.show()

#Correlation Matrix
plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=['int64','float64'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Matrix")

plt.show()

#Gender vs Genetic Disorder
plt.figure(figsize=(10,5))

sns.countplot(
    x="Genetic Disorder",
    hue="Gender",
    data=df
)

plt.xticks(rotation=45)

plt.title("Genetic Disorder by Gender")

plt.show()

#Blood count and genetic
plt.figure(figsize=(10,5))

sns.boxplot(
    x="Genetic Disorder",
    y="Blood cell count (mcL)",
    data=df
)

plt.xticks(rotation=45)

plt.title("Blood Cell Count vs Genetic Disorder")

plt.show()

#relationships between multiple variables
sns.pairplot(df[
    [
        'Patient Age',
        'Blood cell count (mcL)',
        'White Blood cell count (thousand per microliter)'
    ]
])

plt.show()

"""# **Data Warehouse**

## Ware House design

We have here used the Star scheme as we have studied in the class.

### Fact Table

| Column                 |
| ---------------------- |
| Patient_ID             |
| Disease_ID             |
| Gene_ID                |
| Hospital_ID            |
| Family_ID              |
| Blood_Cell_Count       |
| White_Blood_Cell_Count |
| Respiratory_Rate       |
| Heart_Rate             |

**Dimension 1 : Patient**
| Column         |
| ---------------- |
| Patient_ID     |
| Patient_Age    |
| Gender         |
| Place_of_Birth |

**Dimension 2 : Disease**
| Column            |
| ----------------- |
| Disease_ID        |
| Genetic_Disorder  |
| Disorder_Subclass |

**Dimension 3:  Genes**
| Column                |
| ----------------------- |
| Gene_ID               |
| Maternal_Gene         |
| Paternal_Gene         |
| Mother's_Side_Gene    |
| Inherited_from_Father |

**Dimension 4: Hospital**
| Column         |
| -------------- |
| Hospital_ID    |
| Institute_Name |
| Location       |
| Status         |

**Dimension 5: Family History**
| Column             |
| ------------------ |
| Family_ID          |
| Family_Anomalies   |
| Previous_Abortions |
| Maternal_Illness   |
| Radiation_Exposure |
| Substance_Abuse    |

## **Creating Warehouse Table**
"""

## Connect to SQLite Database
import sqlite3

# Create warehouse database
conn = sqlite3.connect("genetic_disease_warehouse.db")

cursor = conn.cursor()

print("Database connected successfully.")

print(df.columns.tolist())

"""Creating Dimension table

"""

# Creating Dimension Tables

# 1. Dim_Patient
cursor.executescript("""
DROP TABLE IF EXISTS Dim_Patient;
CREATE TABLE IF NOT EXISTS Dim_Patient (

    Patient_ID TEXT PRIMARY KEY,
    Patient_Age INTEGER,
    Gender VARCHAR(20),
    Place_of_Birth VARCHAR(100)

);
""")

# 2. Dim_Disease
cursor.executescript("""
DROP TABLE IF EXISTS Dim_Disease;
CREATE TABLE IF NOT EXISTS Dim_Disease (

    Disease_ID INTEGER PRIMARY KEY,
    Genetic_Disorder VARCHAR(100),
    Disorder_Subclass VARCHAR(100)

);
""")

# 3. Dim_Gene
cursor.executescript("""
DROP TABLE IF EXISTS Dim_Gene;
CREATE TABLE IF NOT EXISTS Dim_Gene (

    Gene_ID INTEGER PRIMARY KEY,
    Maternal_Gene VARCHAR(20),
    Paternal_Gene VARCHAR(20),
    Mother_Side_Gene VARCHAR(20),
    Inherited_From_Father VARCHAR(20)

);
""")

# 4. Dim_Hospital
cursor.executescript("""
DROP TABLE IF EXISTS Dim_Hospital;
CREATE TABLE IF NOT EXISTS Dim_Hospital (

    Hospital_ID INTEGER PRIMARY KEY,
    Institute_Name VARCHAR(150),
    Location VARCHAR(100),
    Status VARCHAR(50)

);
""")

# 5. Dim_Family
cursor.executescript("""
DROP TABLE IF EXISTS Dim_Family;
CREATE TABLE IF NOT EXISTS Dim_Family (

    Family_ID INTEGER PRIMARY KEY,
    Maternal_Illness VARCHAR(20),
    Radiation_Exposure VARCHAR(20),
    Substance_Abuse VARCHAR(20),
    Previous_Abortions INTEGER

);
""")

# Create Fact Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Fact_Genetic_Disorder (

    Patient_ID INTEGER,
    Disease_ID INTEGER,
    Gene_ID INTEGER,
    Hospital_ID INTEGER,
    Family_ID INTEGER,

    Blood_Cell_Count FLOAT,
    White_Blood_Cell_Count FLOAT,
    Respiratory_Rate FLOAT,
    Heart_Rate FLOAT,

    FOREIGN KEY (Patient_ID)
    REFERENCES Dim_Patient(Patient_ID),

    FOREIGN KEY (Disease_ID)
    REFERENCES Dim_Disease(Disease_ID),

    FOREIGN KEY (Gene_ID)
    REFERENCES Dim_Gene(Gene_ID),

    FOREIGN KEY (Hospital_ID)
    REFERENCES Dim_Hospital(Hospital_ID),

    FOREIGN KEY (Family_ID)
    REFERENCES Dim_Family(Family_ID)

);
""")

#commit changes
conn.commit()

print("All warehouse tables created successfully.")

#Validate table created

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table';
""")

tables = cursor.fetchall()

print("Tables in Warehouse:")

for table in tables:
    print(table[0])

"""##  **LOADING DATA CSV TO DATAWARE HOUSE**"""

import pandas as pd
import sqlite3

# Load cleaned CSV file
df = pd.read_csv("/content/genetic_disease_dataset_cleaned.csv")

# Connect to SQLite warehouse
conn = sqlite3.connect("genetic_disease_warehouse.db")

cursor = conn.cursor()

print("Cleaned CSV loaded and warehouse connected.")

display(df.head())

#Load Dim_Patient
patient_df = df[[
    "Patient Id",
    "Patient Age",
    "Gender",
    "Place of birth"
]].copy()


patient_df.columns = [
    "Patient_ID",
    "Patient_Age",
    "Gender",
    "Place_of_Birth"
]


patient_df = patient_df.drop_duplicates()


patient_df.to_sql(
    "Dim_Patient",
    conn,
    if_exists="append",
    index=False
)


print("Dim_Patient loaded.")

#Load Dim_Gene
gene_df = df[[
    "Genes in mother's side",
    "Inherited from father",
    "Maternal gene",
    "Paternal gene"
]].copy()


gene_df = gene_df.drop_duplicates()


gene_df.insert(
    0,
    "Gene_ID",
    range(1, len(gene_df)+1)
)


gene_df.columns = [
    "Gene_ID",
    "Mother_Side_Gene",
    "Inherited_From_Father",
    "Maternal_Gene",
    "Paternal_Gene"
]


gene_df.to_sql(
    "Dim_Gene",
    conn,
    if_exists="replace", # Changed from 'append' to 'replace'
    index=False
)


print("Dim_Gene loaded.")

#Load Dim Hospital
hospital_df = df[[
    "Institute Name",
    "Location of Institute",
    "Status"
]].copy()


hospital_df = hospital_df.drop_duplicates()


hospital_df.insert(
    0,
    "Hospital_ID",
    range(1, len(hospital_df)+1)
)


hospital_df.columns = [
    "Hospital_ID",
    "Institute_Name",
    "Location",
    "Status"
]


hospital_df.to_sql(
    "Dim_Hospital",
    conn,
    if_exists="replace", # Changed from 'append' to 'replace'
    index=False
)


print("Dim_Hospital loaded.")

#Load DIM Family
family_df = df[[
    "H/O serious maternal illness",
    "H/O radiation exposure (x-ray)",
    "H/O substance abuse",
    "No. of previous abortion"
]].copy()


family_df = family_df.drop_duplicates()


family_df.insert(
    0,
    "Family_ID",
    range(1, len(family_df)+1)
)


family_df.columns = [
    "Family_ID",
    "Maternal_Illness",
    "Radiation_Exposure",
    "Substance_Abuse",
    "Previous_Abortions"
]


family_df.to_sql(
    "Dim_Family",
    conn,
    if_exists="append",
    index=False
)


print("Dim_Family loaded.")

disease_df = df[[
    "Genetic Disorder",
    "Disorder Subclass"
]].copy()

disease_df.columns = [
    "Genetic_Disorder",
    "Disorder_Subclass"
]

disease_df = disease_df.drop_duplicates().reset_index(drop=True)
disease_df["Disease_ID"] = disease_df.index + 1 # Assign a simple integer ID

# Explicitly drop the table to ensure a clean slate before inserting
# This resolves cases where 'if_exists="replace"' might not fully clear the table
cursor.execute("DROP TABLE IF EXISTS Dim_Disease;")

disease_df.to_sql(
    "Dim_Disease",
    conn,
    if_exists="replace", # Changed from 'append' to 'replace'
    index=False
)

print("Dim_Disease loaded successfully.")

#LOAD FACTTABLE

fact_df = pd.DataFrame()

# Patient ID
fact_df["Patient_ID"] = df["Patient Id"]


# Disease ID mapping
disease_map = dict(
    zip(
        disease_df["Genetic_Disorder"],
        disease_df["Disease_ID"]
    )
)

fact_df["Disease_ID"] = df["Genetic Disorder"].map(disease_map)



# Gene ID mapping
gene_map = gene_df.set_index(
    [
        "Mother_Side_Gene",
        "Inherited_From_Father",
        "Maternal_Gene",
        "Paternal_Gene"
    ]
)["Gene_ID"].to_dict()


fact_df["Gene_ID"] = df.apply(
    lambda row: gene_map.get(
        (
            row["Genes in mother's side"],
            row["Inherited from father"],
            row["Maternal gene"],
            row["Paternal gene"]
        )
    ),
    axis=1
)



# Hospital ID mapping
hospital_map = hospital_df.set_index(
    [
        "Institute_Name",
        "Location",
        "Status"
    ]
)["Hospital_ID"].to_dict()


fact_df["Hospital_ID"] = df.apply(
    lambda row: hospital_map.get(
        (
            row["Institute Name"],
            row["Location of Institute"],
            row["Status"]
        )
    ),
    axis=1
)



# Family ID mapping
family_map = family_df.set_index(
    [
        "Maternal_Illness",
        "Radiation_Exposure",
        "Substance_Abuse",
        "Previous_Abortions"
    ]
)["Family_ID"].to_dict()


fact_df["Family_ID"] = df.apply(
    lambda row: family_map.get(
        (
            row["H/O serious maternal illness"],
            row["H/O radiation exposure (x-ray)"],
            row["H/O substance abuse"],
            row["No. of previous abortion"]
        )
    ),
    axis=1
)



# Measures
fact_df["Blood_Cell_Count"] = df[
    "Blood cell count (mcL)"
]


fact_df["White_Blood_Cell_Count"] = df[
    "White Blood cell count (thousand per microliter)"
]


fact_df["Respiratory_Rate"] = df[
    "Respiratory Rate (breaths/min)"
]

# Convert 'Heart Rate (rates/min' to numeric before loading to database
def convert_heart_rate(hr_str):
    if isinstance(hr_str, (int, float)):
        return hr_str
    hr_str = str(hr_str).strip().lower()
    if 'normal (30-60)' in hr_str:
        return 45.0 # Average of 30 and 60
    elif 'normal' in hr_str:
        return 70.0 # A typical normal heart rate
    elif 'tachypnea' in hr_str or 'tachycardia' in hr_str:
        return 90.0 # A typical higher heart rate
    else:
        try:
            # Attempt direct conversion for any remaining numeric strings
            return float(hr_str)
        except ValueError:
            return np.nan # Mark as NaN if cannot convert

fact_df["Heart_Rate"] = df["Heart Rate (rates/min"].apply(convert_heart_rate)

# Before loading, drop the existing Fact_Genetic_Disorder table
# to ensure the new, cleaned data is loaded completely.
cursor.execute("DROP TABLE IF EXISTS Fact_Genetic_Disorder;")

# Load fact table
fact_df.to_sql(
    "Fact_Genetic_Disorder",
    conn,
    if_exists="append",
    index=False
)


print("Fact table loaded successfully.")

display(fact_df.head())

pd.read_sql(
    "SELECT COUNT(*) FROM Fact_Genetic_Disorder;",
    conn
)

"""# **Online Analytical Processing (OLAP)**"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

"""Load Warehouse Data"""

query = """

SELECT

P.Patient_Age,
P.Gender,

D.Genetic_Disorder,
D.Disorder_Subclass,

G.Maternal_Gene,
G.Paternal_Gene,

H.Location,

F.Maternal_Illness,
F.Previous_Abortions,

FD.Blood_Cell_Count,
FD.White_Blood_Cell_Count,
FD.Respiratory_Rate,
FD.Heart_Rate


FROM Fact_Genetic_Disorder FD

JOIN Dim_Patient P
ON FD.Patient_ID = P.Patient_ID

JOIN Dim_Disease D
ON FD.Disease_ID = D.Disease_ID

JOIN Dim_Gene G
ON FD.Gene_ID = G.Gene_ID

JOIN Dim_Hospital H
ON FD.Hospital_ID = H.Hospital_ID

JOIN Dim_Family F
ON FD.Family_ID = F.Family_ID

"""

warehouse_df = pd.read_sql(query, conn)

display(warehouse_df.head())

"""## **OLAP Operation**"""

#Roll-up (Patient Age → Disease Count)
rollup = warehouse_df.groupby(
    "Genetic_Disorder"
).size().reset_index(
    name="Patient_Count"
)

display(rollup)

#VISUALIZATION OF ROLLUP
plt.figure(figsize=(8,5))

sns.barplot(
    data=rollup,
    x="Genetic_Disorder",
    y="Patient_Count"
)

plt.title("Roll-up: Disease Distribution")

plt.xticks(rotation=45)

plt.show()

#Drill-down (Disease → Disease Subclass)
drill_down = warehouse_df.groupby(

[

"Genetic_Disorder",

"Disorder_Subclass"

]

).size().reset_index(

name="Count"

)



display(drill_down)

#Slice (Analyze only Male patients)
male_patients = warehouse_df[
    warehouse_df["Gender"]=="Male"
]

display(male_patients)

#Dice (Male patients with genetic disorders from a specific location.)

dice = warehouse_df[
    (warehouse_df["Gender"]=="Male") &
    (warehouse_df["Genetic_Disorder"].notnull()) &
    (warehouse_df["Location"].notnull())
]


display(dice)

"""**BUSINESS REPORT GENERATION**"""

disease_report = warehouse_df.groupby(
"Genetic_Disorder"
).agg(

Total_Patients=("Genetic_Disorder","count"),

Average_Age=("Patient_Age","mean"),

Average_Heart_Rate=("Heart_Rate","mean")

).reset_index()

display(disease_report)



"""# **Machine Learning and Prediction**"""

#importing Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

#load Dataset

df = pd.read_csv(
    "/content/genetic_disease_dataset_cleaned.csv"
)

print("Dataset Loaded")

display(df.head())

"""## Select Features
Selected medical and genetic features
"""

features = [

    "Patient Age",
    "Gender",
    "Genes in mother's side",
    "Inherited from father",
    "Maternal gene",
    "Paternal gene",
    "Blood cell count (mcL)",
    "Respiratory Rate (breaths/min)",
    "Heart Rate (rates/min",
    "Birth defects",
    "Symptom 1",
    "Symptom 2",
    "Symptom 3",
    "Symptom 4",
    "Symptom 5"

]


X = df[features]

y = df["Genetic Disorder"]


print(X.shape)
print(y.shape)

#MISSING VALUE HANDLING
# Fill missing values

for col in X.columns:

    if X[col].dtype == "object":

        X[col] = X[col].fillna(
            X[col].mode()[0]
        )

    else:

        X[col] = X[col].fillna(
            X[col].median()
        )


y = y.fillna(
    y.mode()[0]
)


print("Missing values handled")

#Encode Categorical Data
encoder = LabelEncoder()


for col in X.select_dtypes(include="object").columns:

    X[col] = encoder.fit_transform(
        X[col]
    )


# Encode target

y = encoder.fit_transform(y)


print("Encoding completed")



"""## Train-Test Split

\\80% training, 20% testing.
"""

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)


print("Training size:", X_train.shape)

print("Testing size:", X_test.shape)

"""## Decision Tree Model"""

decision_tree = DecisionTreeClassifier(

    criterion="entropy",

    random_state=42

)


decision_tree.fit(

    X_train,

    y_train

)


print("Decision Tree trained")

#Decision Tree Prediction

dt_prediction = decision_tree.predict(
    X_test
)

#plotting decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))


plot_tree(

    decision_tree,

    feature_names=X.columns,

    class_names=[
        str(c) for c in np.unique(y)
    ],

    filled=True,

    rounded=True,

    fontsize=10

)


plt.title("Decision Tree - Genetic Disorder Prediction")

plt.show()

#Decision Tree Performance
dt_accuracy = accuracy_score(
    y_test,
    dt_prediction
)


print(
    "Decision Tree Accuracy:",
    dt_accuracy
)


print(
    classification_report(
        y_test,
        dt_prediction
    )
)



"""# Cross Validation"""

#Decision Tree Cross Validation

dt_cv = cross_val_score(

    decision_tree,

    X,

    y,

    cv=5,

    scoring="accuracy"

)


print(
    "Decision Tree CV Scores:"
)

print(dt_cv)


print(
    "Average:",
    dt_cv.mean()
)

"""# Evaluation"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#generate prediction

dt_pred = decision_tree.predict(X_test)

dt_prob = decision_tree.predict_proba(X_test)

#Accuracy
accuracy = accuracy_score(
    y_test,
    dt_prediction
)

print("Accuracy:", accuracy)

#Precison
precision = precision_score(
    y_test,
    dt_prediction,
    average="weighted"
)

print("Precision:", precision)

#Recal
recall = recall_score(
    y_test,
    dt_prediction,
    average="weighted"
)

print("Recall:", recall)

#F1 Score

f1 = f1_score(
    y_test,
    dt_prediction,
    average="weighted"
)

print("F1 Score:", f1)

"""# Confusion Matrix"""

plt.figure(figsize=(7,5))

sns.heatmap(

    confusion_matrix(
        y_test,
        dt_prediction # Corrected variable name from y_pred
    ),

    annot=True,

    fmt="d",

    cmap="Blues"

)


plt.title(
    "Decision Tree Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.show()

print("Enter Patient Information\n")

patient_data = {}

# Numeric features (Patient Age and Blood cell count are float)
patient_data["Patient Age"] = float(input("Patient Age: "))
patient_data["Blood cell count (mcL)"] = float(input("Blood Cell Count (mcL): "))

# Categorical features (Gender, Genes, Respiratory Rate, Heart Rate, Birth defects are objects/strings)
patient_data["Gender"] = input("Gender (Male/Female): ")
patient_data["Genes in mother's side"] = input("Genes in mother's side (Yes/No): ")
patient_data["Inherited from father"] = input("Inherited from father (Yes/No): ")
patient_data["Maternal gene"] = input("Maternal gene (Yes/No): ")
patient_data["Paternal gene"] = input("Paternal gene (Yes/No): ")
patient_data["Respiratory Rate (breaths/min)"] = input("Respiratory Rate (Normal/Tachypnea): ")
patient_data["Heart Rate (rates/min)"] = input("Heart Rate (Normal/Tachycardia): ") # Corrected column name
patient_data["Birth defects"] = input("Birth defects (Yes/No): ")

# Symptom features (Symptom 1-5 are floats in the original dataset)
for i in range(1, 6):
    while True:
        try:
            patient_data[f"Symptom {i}"] = float(input(f"Symptom {i}: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for the symptom.")


# Create DataFrame, ensuring column order matches the 'features' list used during training
# The 'features' variable is available from the kernel state and used to train the model.
patient_df = pd.DataFrame([patient_data], columns=features)

# Encode categorical columns using the same method as training
# Identify categorical features from the original 'features' list that were part of X
categorical_features_in_X = df[features].select_dtypes(include="object").columns.tolist()

for col in categorical_features_in_X:
    if col in patient_df.columns: # Ensure the column exists in the new patient_df
        le = LabelEncoder()
        # Fit on the original dataset column to get all possible classes
        le.fit(df[col].astype(str))

        # Handle unseen values by mapping them to the first class, then transform
        patient_df[col] = patient_df[col].astype(str).apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        patient_df[col] = le.transform(patient_df[col])
    # No 'else' needed here, as patient_df should have all features from 'features'

# Predict
prediction = decision_tree.predict(patient_df)

# Decode prediction
target_encoder = LabelEncoder()
target_encoder.fit(df["Genetic Disorder"])

predicted_disease = target_encoder.inverse_transform(prediction)

print("\n==============================")
print("Predicted Genetic Disorder:")
print(predicted_disease[0])
print("==============================")

# Probability
prob = decision_tree.predict_proba(patient_df)[0]

result = pd.DataFrame({
    "Disease": target_encoder.classes_,
    "Probability": prob
}).sort_values("Probability", ascending=False)

print("\nTop Predictions")
display(result)