# -*- coding: utf-8 -*-
"""Exploratory data analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t1hBWptN2ZSayQ80rqG1Uw_tXoTtVX-2
"""

import os
import pandas as pd
import numpy as np

# Mount google drive to access the file
from google.colab import drive
drive.mount('/content/drive')

# Import the data file
file_path = '/content/drive/MyDrive/Colab Notebooks/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
os.chdir(os.path.dirname(file_path))

data = pd.read_csv(file_path)

"""# **Correlation Analysis**"""

# import libraries
import seaborn as sns
import matplotlib.pyplot as plt

"""**Heat Map for correlation testing**"""

# Create the bin columns if they don't exist
if 'BMI_Bin' not in data.columns:
    bins_bmi = [0, 18.4, 24.9, 29.9, 34.9, 39.9, 40]
    labels_bmi = ['0–18.4', '18.5–24.9', '25–29.9', '30-34.9', '35-39.9', '40-98']
    data['BMI_Bin'] = pd.cut(data['BMI'], bins=bins_bmi, labels=labels_bmi, right=True, include_lowest=True)

if 'MentHlth_Bin' not in data.columns:
    bins_menthlth = [0, 5, 10, 15, 20, 25, 30]
    labels_menthlth = ['0–5', '6–10', '11–15', '16–20', '21–25', '26–30']
    data['MentHlth_Bin'] = pd.cut(data['MentHlth'], bins=bins_menthlth, labels=labels_menthlth, right=True, include_lowest=True)

if 'PhysHlth_Bin' not in data.columns:
    bins_physhlth = [0, 5, 10, 15, 20, 25, 30]
    labels_physhlth = ['0–5', '6–10', '11–15', '16–20', '21–25', '26–30']
    data['PhysHlth_Bin'] = pd.cut(data['PhysHlth'], bins=bins_physhlth, labels=labels_physhlth, right=True, include_lowest=True)

# Compute correlation matrix
# Drop non-numeric columns created for visualization
data_numeric = data.drop(columns=['BMI_Bin', 'MentHlth_Bin', 'PhysHlth_Bin'])
corr = data_numeric.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Variables")
plt.show()

"""`**Strongest Correlations with Diabetes:**`

`GenHlth (-0.41): A negative correlation means poorer health is associated with higher diabetes prevalence. This is intuitive.`

`HighBP (-0.38): Remains a strong negative correlation. `

`Age (0.34): A moderate positive correlation. This means that as age increases, the likelihood of having diabetes tends to increase. `

`**Moderate Correlations with Diabetes:**`

`HighChol (-0.29): Moderate negative correlation. `

`BMI (-0.29): Moderate negative correlation. Obesity (higher BMI) is a major risk factor for diabetes, so a positive correlation is expected.`

`HeartDiseaseorAttack (-0.21): Moderate negative correlation. Heart disease is a common comorbidity or consequence of diabetes, or they share common risk factors, so a positive correlation is expected.`

`PhysActivity (-0.16): Weak negative correlation.`

`Education (-0.17): Weak negative correlation.`

`Income (-0.22): Weak negative correlation.`


`**Weak to Very Weak Correlations with Diabetes:**`

`MentHlth (-0.09), Smoker (-0.09), Stroke (-0.13), Fruits (-0.05), Veggies (-0.08), AlcoholConsump (-0.03), AnyHealthcare (-0.02), NoDocbcCost (-0.04).`

I want to investigate by doing some more correlation test because for some variables I found negative cor-relation where I expected it will be a positive correlation.

** for numerical variables**
"""

#Point-Biserial Correlation
from scipy.stats import pointbiserialr

corr, p_value = pointbiserialr(data["Diabetes_binary"], data["BMI"])
print(f"Point-Biserial Correlation BMI: {corr:.2f}, P-value: {p_value:.4f}")

corr, p_value = pointbiserialr(data["Diabetes_binary"], data["MentHlth"])
print(f"Point-Biserial Correlation Mental health {corr:.2f}, P-value: {p_value:.4f}")

corr, p_value = pointbiserialr(data["Diabetes_binary"], data["PhysHlth"])
print(f"Point-Biserial Correlation Physical health {corr:.2f}, P-value: {p_value:.4f}")

"""**Chi-square correlation test for categorical variable**"""

from scipy.stats import chi2_contingency
#import warnings

def analyze_categorical_association(data, var1_name, var2_name, alpha=0.05):
    # Create a Contingency Table
    contingency_table = pd.crosstab(data[var1_name], data[var2_name])
    print(f"Contingency Table for '{var1_name}' vs '{var2_name}':\n", contingency_table)

    # Perform Chi-Square Test of Independence
    chi2_statistic, p_value, degrees_freedom, expected_frequencies = chi2_contingency(contingency_table)

    print(f"\n--- Chi-Square Test Results for {var1_name} vs {var2_name} ---")
    print(f"Chi-Square Statistic: {chi2_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {degrees_freedom}")

    # Interpretation of P-value
    interpretation_p_value = ""
    if p_value < alpha:
        interpretation_p_value = (
            f"Since p-value ({p_value:.4f}) < alpha ({alpha}), we reject the null hypothesis.\n"
            f"There is a statistically significant association between {var1_name} and {var2_name}."
        )
    else:
        interpretation_p_value = (
            f"Since p-value ({p_value:.4f}) > alpha ({alpha}), we fail to reject the null hypothesis.\n"
            f"There is no statistically significant association between {var1_name} and {var2_name}."
        )
    print(interpretation_p_value)

    # Calculate Measures of Association
    N = contingency_table.sum().sum()
    k_rows = contingency_table.shape[0]
    k_cols = contingency_table.shape[1]

    # Cramer's V
    cramers_v = np.sqrt(chi2_statistic / (N * (min(k_rows, k_cols) - 1)))
    print(f"\nCramer's V: {cramers_v:.4f}")

    # For Phi (only for 2x2 tables)
    phi = None # Initialize phi to None
    if k_rows == 2 and k_cols == 2:
      phi = np.sqrt(chi2_statistic / N)
      print(f"Phi Coefficient (for 2x2 table): {phi:.4f}")


    results = {
        'contingency_table': contingency_table,
        'chi2_statistic': chi2_statistic,
        'p_value': p_value,
        'degrees_freedom': degrees_freedom,
        'cramers_v': cramers_v,
        'phi_coefficient': phi,
        'interpretation_p_value': interpretation_p_value,
        # 'interpretation_cramers_v': interpretation_cramers_v # This variable is not defined
    }

    return results


# Define the list of independent categorical variables you want to analyze
independent_variables = ['HighBP', 'HighChol', 'CholCheck','Smoker', 'Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex']

# Define your target variable
target_variable = 'Diabetes_binary'

# Iterate through the independent variables and call the function for each
for var in independent_variables:
    results = analyze_categorical_association(
        data=data,
        var1_name=var,
        var2_name=target_variable,
        alpha=0.05 # You can change the significance level if needed
    )
    print("\n" + "="*50 + "\n") # Separator for better readability

"""**Mann-Whitney U Test and Rank-biserial correlation test for ordinal variable**"""

from scipy import stats
def analyze_ordinal_association(data, var1_name, var2_name, alpha=0.05):

  # Separate the data into two groups based on Diabetes Status
  variable1_no_diabetes = data[data[var2_name] == 0][var1_name]
  Variable1_with_diabetes = data[data[var2_name] == 1][var1_name]

  # Get sample sizes
  n1 = len(variable1_no_diabetes)
  n2 = len(Variable1_with_diabetes)

  # Perform the Mann-Whitney U Test
  statistic_u, p_value = stats.mannwhitneyu(variable1_no_diabetes, Variable1_with_diabetes, alternative='two-sided')

  rank_biserial_corr = 1 - (2 * statistic_u) / (n1 * n2)

  print(f"Mann-Whitney U Statistic: {statistic_u:.4f}")
  print(f"P-value: {p_value:.4f}")
  #print(f"Sample size Group 0 (No Diabetes): n1 = {n1}")
  #print(f"Sample size Group 1 (With Diabetes): n2 = {n2}")
  print(f"\nCalculated Rank-Biserial Correlation (r_rb): {rank_biserial_corr:.4f}")

  # --- Interpretation of rank-biserial correlation ---
  # Guidelines for magnitude (Cohen's conventions are often adapted, though somewhat debated for non-parametric)
  # |r_rb| < 0.1: Very weak/trivial
  # |r_rb| approx 0.1 - 0.3: Small/weak
  # |r_rb| approx 0.3 - 0.5: Medium/moderate
  # |r_rb| > 0.5: Large/strong

  abs_r_rb = abs(rank_biserial_corr)
  if abs_r_rb < 0.1:
      strength = "very weak/trivial"
  elif abs_r_rb < 0.3:
      strength = "small/weak"
  elif abs_r_rb < 0.5:
      strength = "medium/moderate"
  else:
      strength = "large/strong"

  direction = "positive" if rank_biserial_corr > 0 else "negative" if rank_biserial_corr < 0 else "no"

  print(f"\nInterpretation:")
  print(f"The rank-biserial correlation is {rank_biserial_corr:.4f}, indicating a {strength} {direction} association.")
  print(f"This means that as Diabetes_binary status changes from 0 to 1, there is a {direction} tendency in {var1_name} ranks.")

# Define the list of independent ordinal variables
independent_variables = ['GenHlth', 'Age', 'Education','Income']

# Define your target variable
target_variable = 'Diabetes_binary'

# Iterate through the independent variables and call the function for each
for var in independent_variables:
    results = analyze_ordinal_association(
        data=data,
        var1_name=var,
        var2_name=target_variable,
        alpha=0.05 # You can change the significance level if needed
    )
    print("\n" + "="*50 + "\n") # Separator for better readability

"""`Variables Most Strongly Associated with Diabetes from the biserial correlation test, Chi-square test and rank biserial test`

`**Strongest Correlations**`

`GenHlth (0.46)`

`HighBP (0.38)`

`HighChol (0.29)`

`BMI (0.29)`

`Age (0.30)`

`DiffWalk (0.27)`

`**Negative Correlations (only for rank biserial correlation)**`

`Education (-0.18)`

`Income (-0.26)`

`**Weak Correlations**`

`Heavy Alcohol Consumption (0.09)`

`Fruits / Veggies (0.05 to 0.07)`

`Sex (0.04) Very small correlation`

`No Doctor Because of Cost (0.04) `

`AnyHealthcare (0.02)`

`**Moderate correlations**`

`HeartDiseaseorAttack (0.21)`

`Physhlth (0.21)`

`Physical Activity (0.15)`

`Stroke (0.12)`

`Cholcheck (0.11)`

`Smoker (0.8)`

`Menthlth (0.9)`

`I found the correlation between the variables and the diabetes binary to be more reasonable than the heat map. I have decided to continue using this correlation result for my further analysis.`

# **Differentiation of Risk Factors Among Subgroups**
"""

# Function to plot prevalence across subgroups
def plot_diabetes_by_group(col):
    data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
    prevalence = data.groupby(col)['Diabetes_binary'].mean().reset_index()
    sns.barplot(x=col, y='Diabetes_binary', data=prevalence)
    plt.title(f"Diabetes Prevalence by {col}")
    plt.ylabel("Diabetes Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Define bins and labels
bins = [0, 18.4, 24.9, 29.9, 34.9, 39.9, 40]
labels = ['0–18.4', '18.5–24.9', '25–29.9', '30-34.9', '35-39.9', '40-98']

# Create a new column for the bin
data['BMI_Bin'] = pd.cut(data['BMI'], bins=bins, labels=labels, right=True, include_lowest=True)

# Define bins and labels
bins = [0, 5, 10, 15, 20, 25, 30]
labels = ['0–5', '6-10', '11-15', '16-20', '21-25', '26-30']

# Create a new column for the bin
data['PhysHlth_Bin'] = pd.cut(data['PhysHlth'], bins=bins, labels=labels, right=True, include_lowest=True)
data['MentHlth_Bin'] = pd.cut(data['MentHlth'], bins=bins, labels=labels, right=True, include_lowest=True)

# Age, Gender, Income, Education etc.
plot_diabetes_by_group("GenHlth")
plot_diabetes_by_group("HighBP")
plot_diabetes_by_group("HighChol")
plot_diabetes_by_group("BMI_Bin")
plot_diabetes_by_group("Age")
plot_diabetes_by_group("DiffWalk")
plot_diabetes_by_group("PhysActivity")
plot_diabetes_by_group("Income")
plot_diabetes_by_group("Education")
plot_diabetes_by_group("HeartDiseaseorAttack")
plot_diabetes_by_group("PhysHlth_Bin")
plot_diabetes_by_group("Stroke")
plot_diabetes_by_group("Sex")
plot_diabetes_by_group("MentHlth_Bin")
plot_diabetes_by_group("Smoker")
plot_diabetes_by_group("HvyAlcoholConsump")

"""`**Insights**`

`- Lower general health is associated with a higher prevalence of diabetes.`

`- Individuals with high blood pressure (BP) tend to have a higher incidence of diabetes.`

`- High cholesterol levels are linked to increased diabetes risk.`

`- The rate of diabetes gradually increases with a higher Body Mass Index (BMI). An outlier is defined as a BMI greater than 30, which is categorized as obesity; the higher the BMI, the greater the risk of diabetes.`

`- As age increases, the prevalence of diabetes also gradually rises.`

`- Difficulty walking is related to a higher incidence of diabetes.`

`- More physical activity is associated with a lower risk of diabetes.`

`- Lower income levels may correlate with higher diabetes prevalence, potentially due to stress.`

`- Lower education levels are linked to higher diabetes rates, possibly because individuals may be less aware of health information.`

`- A history of heart disease or heart attacks is associated with a higher likelihood of diabetes.`

`- Poor physical health sustained over longer periods is connected to a greater risk of diabetes.`

`- A history of strokes is related to increased diabetes prevalence.`

`- Males have a higher incidence of diabetes compared to females.`

`- More days of poor mental health correlate with increased diabetes risk.`

`- Smokers are more likely to have diabetes.`

`- Lower alcohol consumption is associated with a reduced risk of diabetes.`

`As I mentioned earlier, several factors influence the development of diabetes, including genetics, lifestyle, age, and socioeconomic status. In light of this, I am examining the correlation between these factors and looking for specific risk factors that may contribute to diabetes, focusing on various combinations of variables. It's important to note that not all of these variables are highly correlated, and the relationships may vary depending on the risk factors involved.`

# **Define risky behaviours**
"""

# Define risky behaviours
# Convert categorical columns to numeric for calculation
data['Risk_Score'] = data['Smoker'].astype(int) + (1 - data['PhysActivity'].astype(int)) + (data['BMI'] > 30).astype(int)

# Diabetes prevalence by risk score
data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
risk_diabetes = data.groupby('Risk_Score')['Diabetes_binary'].mean().reset_index()

# Plot
sns.lineplot(x='Risk_Score', y='Diabetes_binary', data=risk_diabetes, marker='o')
plt.title("Diabetes Rate by Combined Risk Behaviors (Smoking, Inactivity, Obesity)")
plt.xlabel("Number of Risk Factors")
plt.ylabel("Diabetes Rate")
plt.xticks(risk_diabetes['Risk_Score'])
plt.show()

"""`The plot clearly demonstrates a strong, positive, and approximately linear relationship between the number of combined risk factors and the diabetes rate.`

`Individuals with no of these three risk factors have the lowest diabetes rate, approximately 0.31. This serves as a baseline.`

`The diabetes rate increases to approximately 0.49 for individuals with one risk factor. This is a significant jump from the baseline.`

`The rate continues to climb to approximately 0.63 for individuals possessing two of these risk factors. The increase from one to two risk factors is also substantial.`

`Individuals with all three risk factors exhibit the highest diabetes rate, reaching approximately 0.75. This indicates a very high likelihood of diabetes when all three behaviors are present.`

`This composite "Number of Risk Factors" appears to be a strong predictor of diabetes status, showing a clear gradient of risk.`
"""

# Convert categorical columns to appropriate types for calculations
data['HealthyDiet'] = (data['Fruits'].astype(bool)) & (data['Veggies'].astype(bool))

# Ensure other categorical columns used in calculations are treated as numeric
data['RiskScore_DietActivity'] = (1 - data['HealthyDiet'].astype(int)) + (1 - data['PhysActivity'].astype(int)) + (data['BMI'] > 30).astype(int)

# Diabetes rate by number of risk factors
combo_col = 'RiskScore_DietActivity'

data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
combo_plot = data.groupby(combo_col)['Diabetes_binary'].mean().reset_index()
sns.lineplot(x=combo_col, y='Diabetes_binary', data=combo_plot, marker='o')
plt.title(f"Diabetes rate by combined risk beaviour no healthy diet/inactivity/obestiy")
plt.ylabel("Diabetes Rate")
plt.xlabel("Number of Risk Factors")
plt.xticks(combo_plot[combo_col])
plt.show()

"""`This composite "RiskScore_DietActivity" appears to be a strong predictor of diabetes status, showing a clear gradient of risk.`"""

data['CardioRisk'] = data['HighBP'].astype(int) + data['HighChol'].astype(int) + data['HeartDiseaseorAttack'].astype(int)

# Diabetes rate by number of risk factors
combo_col = 'CardioRisk'

data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
combo_plot = data.groupby(combo_col)['Diabetes_binary'].mean().reset_index()
sns.lineplot(x=combo_col, y='Diabetes_binary', data=combo_plot, marker='o')
plt.title(f"Diabetes rate by combined risk beaviour High BP/High Cholestorol/Having a heart diease history")
plt.ylabel("Diabetes Rate")
plt.xlabel("Number of Risk Factors")
plt.xticks(combo_plot[combo_col])
plt.show()

"""`This composite "Cardio Risk" appears to be a strong predictor of diabetes status, showing a clear gradient of risk.`"""

data['StrokeRisk'] = data['HighBP'].astype(int) + data['HighChol'].astype(int) + data['Stroke'].astype(int)

# Diabetes rate by number of risk factors
combo_col = 'CardioRisk'

data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
combo_plot = data.groupby(combo_col)['Diabetes_binary'].mean().reset_index()
sns.lineplot(x=combo_col, y='Diabetes_binary', data=combo_plot, marker='o')
plt.title(f"Diabetes rate by combined risk beaviour High BP/High Cholestorol/Having a stroke history")
plt.ylabel("Diabetes Rate")
plt.xlabel("Number of Risk Factors")
plt.xticks(combo_plot[combo_col])
plt.show()

"""`This composite "Stroke Risk" appears to be a strong predictor of diabetes status, showing a clear gradient of risk.`"""

data['HighMentalStress'] = (data['MentHlth'].astype(int) >= 10).astype(int)
data['HighPhysicalIssues'] = (data['PhysHlth'].astype(int) >= 10).astype(int)
data['PoorGeneralHealth'] = (data['GenHlth'].astype(int) >= 4).astype(int)
data['RiskScore_HealthPerception'] = data['HighMentalStress'] + data['HighPhysicalIssues'] + data['PoorGeneralHealth']

# Diabetes rate by number of risk factors
combo_col = 'RiskScore_HealthPerception'

data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
combo_plot = data.groupby(combo_col)['Diabetes_binary'].mean().reset_index()
sns.lineplot(x=combo_col, y='Diabetes_binary', data=combo_plot, marker='o')
plt.title(f"Diabetes rate by combined risk beaviour High Mental stress/High Physical Illness/ Poor General Health")
plt.ylabel("Diabetes Rate")
plt.xlabel("Number of Risk Factors")
plt.xticks(combo_plot[combo_col])
plt.show()

"""`This composite of overall health appears to be a strong predictor of diabetes status, showing a clear gradient of risk.`"""