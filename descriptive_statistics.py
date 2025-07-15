# **Data source**

https://www.kaggle.com/datasets/abdelazizsami/cdc-diabetes-health-indicators?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv

# **Data description**

Data description link: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
1.	Diabetes_binary: 0 = no diabetes 1 = prediabetes or diabetes
2.	HighBP: 0 = no high BP 1 = high BP
3.	HighChol: 0 = no high cholesterol 1 = high cholesterol
4.	CholCheck: 0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years
5.	BMI: Body Mass Index
6.	Smoker: Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes
7.	Stroke: (Ever told) you had a stroke. 0 = no 1 = yes
8.	HeartDiseaseorAttack: coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes
9.	PhysActivity: physical activity in past 30 days - not including job 0 = no 1 = yes
10.	Fruits: Consume Fruit 1 or more times per day 0 = no 1 = yes
11.	Veggies: Consume Vegetables 1 or more times per day 0 = no 1 = yes
12.	HvyAlcoholConsump: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no 1 = yes
13.	AnyHealthcare: Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes
14.	NoDocbcCost: Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes
15.	GenHlth: Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor
16.	MentHlth: Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days
17.	PhysHlth: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days
18.	DiffWalk: Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes
19.	Sex: 0 = female 1 = male
20.	Age: 13-level age category (_AGEG5YR see codebook)
1 Age 18 – 24, 2 Age 25 to 29, 3 Age 30 to 34, 4 Age 35 to 39, 5 Age 40 to 44, 6 Age 45 to 49, 7 Age 50 to 54, 8 Age 55 to 59, 9 Age 60 to 64, 10 Age 65 to 69, 11 Age 70 to 74, 12 Age 75 to 79, 13 Age 80 or older

21.	Education: Education level (EDUCA see codebook) scale 1-6
1 = Never attended school or only kindergarten, 2 = Grades 1 through 8 (Elementary), 3 = Grades 9 through 11 (Some high school), 4 = Grade 12 or GED (High school graduate), 5 = College 1 year to 3 years (Some college or technical school), 6 = College 4 years or more (College graduate)
22.	Income: Income scale (INCOME2 see codebook) scale 1-8
1 = less than 10,000, 2 = Less than $15,000 , 3 = Less than $20,000, 4 = Less than $25,000, 5 = Less than $35,000, 6 = Less than $50,000, 7 =  Less than $75,000, 8 = $75,000 or more

# **Data import**

**Importing libraries**
"""

import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

from nbconvert import HTMLExporter
import nbformat

"""**Load the data file**


"""

# Mount google drive to access the file
from google.colab import drive
drive.mount('/content/drive')

# Import the data file
file_path = '/content/drive/MyDrive/Colab Notebooks/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
os.chdir(os.path.dirname(file_path))

"""# **Descriptive statistics**"""

# Load the data file
data = pd.read_csv(file_path)
print(data)

# Summary of data frame
data.info()

# Check for missing value
data.isna().sum()

"""`No missing value in the dataset`"""

# Check the unique valuse in the columns
for col in data.columns:
    print(col, ":", data[col].nunique())

# Convert numerical to categorical variable
data['Diabetes_binary'] = data['Diabetes_binary'].astype('category')
data['HighBP'] = data['HighBP'].astype('category')
data['HighChol'] = data['HighChol'].astype('category')
data['CholCheck'] = data['CholCheck'].astype('category')
data['Smoker'] = data['Smoker'].astype('category')
data['Stroke'] = data['Stroke'].astype('category')
data['HeartDiseaseorAttack'] = data['HeartDiseaseorAttack'].astype('category')
data['PhysActivity'] = data['PhysActivity'].astype('category')
data['Fruits'] = data['Fruits'].astype('category')
data['Veggies'] = data['Veggies'].astype('category')
data['HvyAlcoholConsump'] = data['HvyAlcoholConsump'].astype('category')
data['AnyHealthcare'] = data['AnyHealthcare'].astype('category')
data['NoDocbcCost'] = data['NoDocbcCost'].astype('category')
data['GenHlth'] = data['GenHlth'].astype('category')
data['DiffWalk'] = data['DiffWalk'].astype('category')
data['Sex'] = data['Sex'].astype('category')
data['Age'] = data['Age'].astype('category')
data['Education'] = data['Education'].astype('category')
data['Income'] = data['Income'].astype('category')

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# Overall Summary for numerical variables

data.describe().T

# Categorical Feature Distribution

# HighBP
highbp_percent = data['HighBP'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with High Blood Pressure: {highbp_percent:.2f}%")

# HighChol
highchol_percent = data['HighChol'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with High Cholesterol: {highchol_percent:.2f}%")

# Cholestorol check
CholCheck_percent = data['CholCheck'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with Cholesterol Check: {CholCheck_percent:.2f}%")

# Smoker
Smoker_percent = data['Smoker'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with Smoker: {Smoker_percent:.2f}%")

# Stroke
Stroke_percent = data['Stroke'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with stroke history: {Stroke_percent:.2f}%")

# Heart Disease or Attack
HeartDiseaseorAttack_percent = data['HeartDiseaseorAttack'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with Heart Disease or Attack history: {HeartDiseaseorAttack_percent:.2f}%")

# PhysActivity
PhysActivity_percent = data['PhysActivity'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with Physical Activity: {PhysActivity_percent:.2f}%")

# Fruits
Fruits_percent = data['Fruits'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with people Fruits consumption: {Fruits_percent:.2f}%")

# Veggies
Veggies_percent = data['Veggies'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with people Veggies consumption: {Veggies_percent:.2f}%")

# HvyAlcoholConsump
HvyAlcoholConsump_percent = data['HvyAlcoholConsump'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with people consume heavy alcohol: {HvyAlcoholConsump_percent:.2f}%")

# AnyHealthcare
AnyHealthcare_percent = data['AnyHealthcare'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with people have health care coverage : {AnyHealthcare_percent:.2f}%")

# NoDocbcCost
NoDocbcCost_percent = data['NoDocbcCost'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with people need doctor but don't have any coverage: {NoDocbcCost_percent:.2f}%")

# DiffWalk
DiffWalk_percent = data['DiffWalk'].value_counts(normalize=True).get(1, 0) * 100
print(f"Percentage with people have diffculty walking: {DiffWalk_percent:.2f}%")

print()
print("Percentage with general health by level",data['GenHlth'].value_counts(normalize=True))

print()
print("Percentage by sex 0=Female, 1=Male", data['Sex'].value_counts(normalize=True))

print()
print("Percentage with education by level",data['Education'].value_counts(normalize=True))

print()
print("percentage with income by level",data['Income'].value_counts(normalize=True))

"""**Variables exploration to find out the associateion with Diabetes**"""

# Visualization
# Target variable distribution

sns.countplot(x='Diabetes_binary', data=data)
plt.title('Diabetes Binary Distribution')
plt.xticks([0,1], ['No Diabetes', 'Diabetes'])
plt.xlabel('')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

"""`Equal distrbution of diabetes and no diabetes. The dataset is balanced.`"""

# BMI distribution - Histogram
sns.histplot(data['BMI'], bins=30, kde=True)
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# BMI distribution - Boxplot
sns.boxplot(x='Diabetes_binary', y='BMI', data=data)
plt.xticks([0,1], ['No Diabetes', 'Diabetes'])
plt.title('BMI Distribution by Diabetes Status')
plt.xlabel('')
plt.tight_layout()
plt.show()

"""`The graph illustrates how BMI is distributed across a population, showing that a substantial number of individuals are in the overweight category, and there's a smaller but present group with obesity, while fewer individuals are underweight or in the lower end of the healthy weight range. The most frequent BMI values (the peak of the distribution) are somewhere between 25 and 30, suggesting that a significant portion of the population in this dataset falls into the "overweight" category (BMI 25-29.9 is generally considered overweight).`

`The boxplot is designed to visually demonstrate whether there's a difference in BMI levels between people who have diabetes and those who do not, and it appears to suggest that individuals with diabetes generally have higher BMIs. The median line in the right box (likely representing individuals with diabetes) appears to be higher than the median line in the left box (likely representing individuals without diabetes). This suggests that, on average, individuals with diabetes tend to have a higher BMI. Both groups appear to have a significant number of outliers, especially on the higher end of the BMI scale. This indicates that there are individuals in both groups with very high BMI values, some of which are considered extreme relative to their respective groups.`

`Next part is exploratory data analysis (EDA) code is avaialble in Exploratory data analysis (EDA) file`
"""