# -*- coding: utf-8 -*-
"""ML models.ipynb

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

# Combo column for exploratory analysis
data['Risk_Score'] = data['Smoker'].astype(int) + (1 - data['PhysActivity'].astype(int)) + (data['BMI'] > 30).astype(int)
data['HealthyDiet'] = (data['Fruits'].astype(bool)) & (data['Veggies'].astype(bool))
data['RiskScore_DietActivity'] = (1 - data['HealthyDiet'].astype(int)) + (1 - data['PhysActivity'].astype(int)) + (data['BMI'] > 30).astype(int)
data['CardioRisk'] = data['HighBP'].astype(int) + data['HighChol'].astype(int) + data['HeartDiseaseorAttack'].astype(int)
data['StrokeRisk'] = data['HighBP'].astype(int) + data['HighChol'].astype(int) + data['Stroke'].astype(int)
data['HighMentalStress'] = (data['MentHlth'].astype(int) >= 10).astype(int)
data['HighPhysicalIssues'] = (data['PhysHlth'].astype(int) >= 10).astype(int)
data['PoorGeneralHealth'] = (data['GenHlth'].astype(int) >= 4).astype(int)
data['RiskScore_HealthPerception'] = data['HighMentalStress'] + data['HighPhysicalIssues'] + data['PoorGeneralHealth']

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

"""# **Data Preprocessing**"""

# Define features (X_numeric) and target variable (y)
# Exclude the target variable and the newly created bin/risk columns from features
X_numeric = data.drop(columns=['Diabetes_binary', 'BMI_Bin', 'PhysHlth_Bin', 'MentHlth_Bin',
                               'Risk_Score', 'HealthyDiet', 'RiskScore_DietActivity',
                               'CardioRisk', 'StrokeRisk', 'HighMentalStress',
                               'HighPhysicalIssues', 'PoorGeneralHealth', 'RiskScore_HealthPerception'])
y = data['Diabetes_binary']

from sklearn.model_selection import train_test_split

X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, stratify=y, random_state=42
)

"""# **Machine Learning Models**

## **With selected features**
"""

# Univariate feature selection
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_numeric, y_train)
X_test_selected = selector.transform(X_test_numeric)

# Get selected feature names
selected_columns = X_train_numeric.columns[selector.get_support()]
print("Selected Features:", selected_columns)

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define the cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

model = LogisticRegression(solver='liblinear', random_state=42)

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model.fit(X_train_selected, y_train)

y_pred_lr = model.predict(X_test_selected)
y_prob_lr = model.predict_proba(X_test_selected)[:, 1]

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))

#XGBoost
import xgboost as xgb

xgb_model_selected = xgb.XGBClassifier()

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

xgb_model_selected.fit(X_train_selected, y_train)

y_pred_xgb = xgb_model_selected.predict(X_test_selected)
y_prob_xgb = xgb_model_selected.predict_proba(X_test_selected)[:, 1]

print("XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_prob_xgb))

#LightGBM
import lightgbm as lgb

lgb_model_selected = lgb.LGBMClassifier()

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")
lgb_model_selected.fit(X_train_selected, y_train)

y_pred_lgb = lgb_model_selected.predict(X_test_selected)
y_prob_lgb = lgb_model_selected.predict_proba(X_test_selected)[:, 1]

print("LightGBM Report:")
print(classification_report(y_test, y_pred_lgb))
print("AUC:", roc_auc_score(y_test, y_prob_lgb))

models_with_univariate_feature = {
    "Logistic Regression": (y_test, y_prob_lr),
    "XGBoost": (y_test, y_prob_xgb),
    "LightGBM": (y_test, y_prob_lgb),
}

for name, (true, prob) in models_with_univariate_feature.items():
    auc = roc_auc_score(true, prob)
    print(f"{name} AUC: {auc:.3f}")

#Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression

rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=10)
X_train_selected = rfe.fit_transform(X_train_numeric, y_train)
X_test_selected = rfe.transform(X_test_numeric)

# Get feature names
selected_columns = X_train_numeric.columns[rfe.support_]
print("Selected Features:", selected_columns)

# Logistic regression
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, roc_auc_score

model = LogisticRegression(solver='liblinear')

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model.fit(X_train_selected, y_train)

y_pred_lr = model.predict(X_test_selected)
y_prob_lr = model.predict_proba(X_test_selected)[:, 1]

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))

#XGBoost
#import xgboost as xgb

xgb_model_selected = xgb.XGBClassifier()

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

xgb_model_selected.fit(X_train_selected, y_train)

y_pred_xgb = xgb_model_selected.predict(X_test_selected)
y_prob_xgb = xgb_model_selected.predict_proba(X_test_selected)[:, 1]

print("XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_prob_xgb))

#LightGBM
#import lightgbm as lgb

# Need to re-initialize LightGBM model
lgb_model_selected = lgb.LGBMClassifier()

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

lgb_model_selected.fit(X_train_selected, y_train)

y_pred_lgb = lgb_model_selected.predict(X_test_selected)
y_prob_lgb = lgb_model_selected.predict_proba(X_test_selected)[:, 1]

print("LightGBM Report:")
print(classification_report(y_test, y_pred_lgb))
print("AUC:", roc_auc_score(y_test, y_prob_lgb))

models_with_RFE_feature = {
    "Logistic Regression": (y_test, y_prob_lr),
    "XGBoost": (y_test, y_prob_xgb),
    "LightGBM": (y_test, y_prob_lgb),
}

for name, (true, prob) in models_with_RFE_feature.items():
    auc = roc_auc_score(true, prob)
    print(f"{name} AUC: {auc:.3f}")

#Random forest classifer

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train_numeric, y_train)

importances = rf.feature_importances_
feature_names = X_train_numeric.columns
feature_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
top_features = feature_df.sort_values(by='importance', ascending=False).head(10)['feature'].tolist()

X_train_selected = X_train_numeric[top_features] # Select features from X_train_numeric
X_test_selected = X_test_numeric[top_features] # Select features from X_test_numeric

print("Selected Features:", top_features)

# Logistic regression
model_lr = LogisticRegression(solver='liblinear')

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")
model_lr.fit(X_train_selected, y_train)

y_pred_lr = model_lr.predict(X_test_selected)
y_prob_lr = model_lr.predict_proba(X_test_selected)[:, 1]

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))

#XGBoost
model_xgb = xgb.XGBClassifier(enable_categorical=True)

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model_xgb.fit(X_train_selected, y_train)

y_pred_xgb = model_xgb.predict(X_test_selected)
y_prob_xgb = model_xgb.predict_proba(X_test_selected)[:, 1]

print("XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_prob_xgb))

#LightGBM
model_lgb = lgb.LGBMClassifier()

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model_lgb.fit(X_train_selected, y_train)

y_pred_lgb = model_lgb.predict(X_test_selected)
y_prob_lgb = model_lgb.predict_proba(X_test_selected)[:, 1]

print("LightGBM Report:")
print(classification_report(y_test, y_pred_lgb))
print("AUC:", roc_auc_score(y_test, y_prob_lgb))

models_with_randomforest_feature = {
    "Logistic Regression": (y_test, y_prob_lr),
    "XGBoost": (y_test, y_prob_xgb),
    "LightGBM": (y_test, y_prob_lgb),
}

for name, (true, prob) in models_with_randomforest_feature.items():
    auc = roc_auc_score(true, prob)
    print(f"{name} AUC: {auc:.3f}")

# Xgboost classifier
# Use enable_categorical=True as X might contain categorical columns
model_importance = xgb.XGBClassifier(enable_categorical=True)
model_importance.fit(X_train_numeric, y_train)

importances = model_importance.feature_importances_
feature_names = X_train_numeric.columns # Use column names from X_train
feature_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
top_features = feature_df.sort_values(by='importance', ascending=False).head(10)['feature'].tolist()

print("Selected Features (based on XGBoost importance):", top_features)

# Select top features from train and test sets
X_train_selected = X_train_numeric[top_features]
X_test_selected = X_test_numeric[top_features]

# Logistic regression with selected features
model_lr = LogisticRegression(solver='liblinear')

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model_lr.fit(X_train_selected, y_train)

y_pred_lr = model_lr.predict(X_test_selected)
y_prob_lr = model_lr.predict_proba(X_test_selected)[:, 1]

print("\nLogistic Regression Report (with selected features):")
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))

# XGBoost with selected features
# Ensure enable_categorical=True if selected features contain categorical types
model_xgb = xgb.XGBClassifier(enable_categorical=True)

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model_xgb.fit(X_train_selected, y_train)

y_pred_xgb = model_xgb.predict(X_test_selected)
y_prob_xgb = model_xgb.predict_proba(X_test_selected)[:, 1]

print("\nXGBoost Report (with selected features):")
print(classification_report(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_prob_xgb))

# LightGBM with selected features
model_lgb = lgb.LGBMClassifier()

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model_lgb.fit(X_train_selected, y_train)

y_pred_lgb = model_lgb.predict(X_test_selected)
y_prob_lgb = model_lgb.predict_proba(X_test_selected)[:, 1]

print("\nLightGBM Report (with selected features):")
print(classification_report(y_test, y_pred_lgb))
print("AUC:", roc_auc_score(y_test, y_prob_lgb))

models_with_XGBoost_feature= {
    "Logistic Regression": (y_test, y_prob_lr),
    "XGBoost": (y_test, y_prob_xgb),
    "LightGBM": (y_test, y_prob_lgb),
}

for name, (true, prob) in models_with_XGBoost_feature.items():
    auc = roc_auc_score(true, prob)
    print(f"{name} AUC: {auc:.3f}")

"""## **With high to moderately correlated variable**"""

corrvar= ['HighBP', 'HighChol', 'BMI', 'GenHlth','Age', 'DiffWalk', 'Stroke', 'HeartDiseaseorAttack','PhysHlth', 'CholCheck']

X_train_corr, X_test_corr, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, stratify=y, random_state=42
)

X_train_corr = X_train_corr[corrvar]
X_test_corr = X_test_corr[corrvar]

# Get selected feature names
selected_columns = X_train_corr.columns
print("Selected Features:", selected_columns)

# Logistic regression

model = LogisticRegression(solver='liblinear')

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_corr, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_corr, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")

model.fit(X_train_corr, y_train)

y_pred_lr = model.predict(X_test_corr)
y_prob_lr = model.predict_proba(X_test_corr)[:, 1]

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))

# XGBoost with selected features
# Ensure enable_categorical=True if selected features contain categorical types
model_xgb = xgb.XGBClassifier(enable_categorical=True)

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_corr, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_corr, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")
model_xgb.fit(X_train_corr, y_train)

y_pred_xgb = model_xgb.predict(X_test_corr)
y_prob_xgb = model_xgb.predict_proba(X_test_corr)[:, 1]

print("\nXGBoost Report (with selected features):")
print(classification_report(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_prob_xgb))

# LightGBM with selected features
model_lgb = lgb.LGBMClassifier()

#Cross validation
print("Cross-validation scores (Accuracy):")
scores_accuracy = cross_val_score(model, X_train_corr, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy per fold: {scores_accuracy}")
print(f"Mean Accuracy: {np.mean(scores_accuracy):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(scores_accuracy):.4f}")

print("\nCross-validation scores (ROC AUC):")
scores_roc_auc = cross_val_score(model, X_train_corr, y_train, cv=cv, scoring='roc_auc')
print(f"ROC AUC per fold: {scores_roc_auc}")
print(f"Mean ROC AUC: {np.mean(scores_roc_auc):.4f}")
print(f"Standard Deviation of ROC AUC: {np.std(scores_roc_auc):.4f}")
model_lgb.fit(X_train_corr, y_train)

y_pred_lgb = model_lgb.predict(X_test_corr)
y_prob_lgb = model_lgb.predict_proba(X_test_corr)[:, 1]

print("\nLightGBM Report (with selected features):")
print(classification_report(y_test, y_pred_lgb))
print("AUC:", roc_auc_score(y_test, y_prob_lgb))

models_with_correlated_variable = {
    "Logistic Regression": (y_test, y_prob_lr),
    "XGBoost": (y_test, y_prob_xgb),
    "LightGBM": (y_test, y_prob_lgb),
}

for name, (true, prob) in models_with_correlated_variable .items():
    auc = roc_auc_score(true, prob)
    print(f"{name} AUC: {auc:.3f}")

"""## **Comparison between ML models**

`In my feature selection process, I applied several techniques. The first method is univariate feature selection, where I used SelectKBest from scikit-learn to select the top k features based on the chi-square test or ANOVA F-test. The second method is Recursive Feature Elimination (RFE), which is a wrapper method. The third approach involves model-based selection, utilizing feature importances from a tree-based model like Random Forest. Finally, I also included XGBoost as another tree-based model.`

`For each feature selection method, I evaluated three machine learning models: Logistic Regression, XGBoost, and LightGBM. For the univariate feature selection, both accuracy and AUC values were quite close, with the highest accuracy and AUC achieved by LightGBM. Similarly, in the RFE method, all three models exhibited the same accuracy of 73%, but LightGBM achieved the highest AUC at 80.9%. For the Random Forest model, LightGBM again showed the highest accuracy (75%) and AUC (82.3%). Lastly, when using the XGBoost method for feature selection, the accuracy across all models was the same, but LightGBM had a higher AUC of 82.7%.`

`I then applied these three machine learning models to the strongly and moderately correlated variables identified from a correlation test, achieving the highest accuracy and AUC of 75% and 82.5%, respectively, with LightGBM. In comparing all the feature selection methods, I found that XGBoost is the best method, while LightGBM is the best machine learning model for predicting diabetes. The features identified using the XGBoost method are as follows:`

`HighBP, HighChol, GenHlth, HvyAlcoholConsump, CholCheck, BMI ,Age, DiffWalk, Sex, HeartDiseaseorAttack.`

`This proves that the customized combined column I examined does not significantly impact diabetes prediction. The columns that are strongly to moderately correlated can provide the best results.`

# **Bayesian Logistic regression**
"""

import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler


continuous_cols = ['BMI', 'MentHlth', 'PhysHlth']

categorical_cols = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth','DiffWalk',
                    'Sex', 'Age','Education','Income']

X_train_processed = X_train_numeric.copy()
X_test_processed = X_test_numeric.copy()

# Scale continuous features
scaler = StandardScaler()
X_train_processed[continuous_cols] = scaler.fit_transform(X_train_processed[continuous_cols])
X_test_processed[continuous_cols] = scaler.transform(X_test_processed[continuous_cols])

# One-hot encode categorical variables
# Combine the processed numerical and one-hot encoded categorical columns
X_train_final = pd.get_dummies(X_train_processed[categorical_cols], drop_first=True).join(X_train_processed[continuous_cols])
X_test_final = pd.get_dummies(X_test_processed[categorical_cols], drop_first=True).join(X_test_processed[continuous_cols])

y_train_array = y_train.astype(int).to_numpy()
y_test_array = y_test.astype(int).to_numpy()

X_train_array = X_train_final.to_numpy().astype(float)

print(X_train_final.dtypes)
print(X_train_array.shape)

# --- Bayesian Logistic Regression Model ---
with pm.Model() as bayes_logistic_model:
    # Declare shared variables for data inside the model context
    X_data_shared = pm.MutableData("X_data", X_train_array)
    y_data_shared = pm.MutableData("y_data", y_train_array)

    # Shape matches the number of features in the input data
    beta = pm.Normal("beta", mu=0, sigma=2, shape=(X_data_shared.shape.eval()[1],))
    intercept = pm.Normal("intercept", mu=0, sigma=2)

    # Linear combination of features and coefficients
    logits = intercept + pm.math.dot(X_data_shared, beta)

    # Apply the sigmoid (logistic) function to get probabilities
    p = pm.Deterministic("p", pm.math.sigmoid(logits))

    # Likelihood: Bernoulli distribution for binary outcomes
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_data_shared)

    # Sample from the posterior distribution
    trace = pm.sample(draws=2000, tune=2000, target_accept=0.99, chains=2, random_seed=123)

# Evaluate uncertainty and coefficients
az.summary(trace, var_names=["beta", "intercept"])
az.plot_forest(trace, var_names=["beta"], combined=True)


# Make predictions on the test set
# Update the shared data with the test set values
with bayes_logistic_model:
    pm.set_data({"X_data": X_test_final.to_numpy().astype(float), "y_data": y_test_array})
    ppc_test = pm.sample_posterior_predictive(trace, var_names=["p", "y_obs"], random_seed=42)

# This gives you a single probability estimate per test data point
mean_predicted_probabilities = ppc_test.posterior_predictive['p'].mean(dim=["chain", "draw"])

# Convert to numpy array for easier handling with sklearn metrics
mean_predicted_probabilities_np = mean_predicted_probabilities.to_numpy()

# Apply a threshold (e.g., 0.5) to get binary class predictions
threshold = 0.5
y_pred_bayes = (mean_predicted_probabilities_np > threshold).astype(int)

# --- Calculate Classification Metrics ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss

print(f"Metrics for Threshold = {threshold}:")

# Accuracy
accuracy = accuracy_score(y_test_array, y_pred_bayes)
print(f"Accuracy: {accuracy:.4f}")

# Precision
precision = precision_score(y_test_array, y_pred_bayes)
print(f"Precision: {precision:.4f}")

# Recall (Sensitivity)
recall = recall_score(y_test_array, y_pred_bayes)
print(f"Recall: {recall:.4f}")

# F1-score
f1 = f1_score(y_test_array, y_pred_bayes)
print(f"F1-Score: {f1:.4f}")

# ROC AUC
# Note: roc_auc_score takes probabilities, not binary predictions
roc_auc = roc_auc_score(y_test_array, mean_predicted_probabilities_np)
print(f"ROC AUC: {roc_auc:.4f}")

# Log Loss
logloss = log_loss(y_test_array, mean_predicted_probabilities_np)
print(f"Log Loss: {logloss:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_array, y_pred_bayes)
print("\nConfusion Matrix:")
print(conf_matrix)

print("Order of features (beta coefficients):")
for i, col_name in enumerate(X_train_final.columns):
    print(f"beta[{i}]: {col_name}")

"""beta[0]: HighBP (Strong Positive Effect)

Dot: Around +0.7

HDI: Clearly positive, not crossing zero.

Interpretation: Having high blood pressure (HighBP) is strongly associated with an increased odds of having diabetes.


beta[1]: HighChol (Positive Effect)

Dot: Around +0.55

HDI: Clearly positive, not crossing zero.

Interpretation: Having high cholesterol (HighChol) is associated with an increased odds of having diabetes.


beta[2]: CholCheck (Positive Effect)

Dot: Around +1.3

HDI: Clearly positive, not crossing zero.

Interpretation: This is an extremely strong positive effect. having a cholesterol check being strongly associated with increased odds of diabetes is not seems correct.It's confusing.


beta[3]: Smoker (Negative Effect)

Dot: Around -0.1

HDI: Entirely negative, not crossing zero.

Interpretation: Being a smoker is associated with a decreased odds of having diabetes. This is also confusing like cholcheck.


beta[4]: Stroke (Positive Effect)

Dot: Around +0.25

HDI: Entirely positive, not crossing zero.

Interpretation: Having had a stroke is associated with an increased odds of diabetes.


beta[5]: HeartDiseaseorAttack (Positive Effect)

Dot: Around +0.35

HDI: Entirely positive, not crossing zero.

Interpretation: Having had heart disease or an attack is associated with an increased odds of diabetes.


beta[6]: PhysActivity (Negative Effect)

Dot: Around -0.1

HDI: Entirely negative, not crossing zero.

Interpretation: Being physically active is associated with a decreased odds of diabetes


beta[7]: Fruits (Negative Effect)

Dot: Around -0.05

HDI: Entirely negative, not crossing zero.

Interpretation: Eating fruits is associated with a slightly decreased odds of diabetes


beta[8]: Veggies (Effect Not Clearly Distinguishable from Zero)

Dot: Slightly positive.

HDI: Crosses zero.

Interpretation: The model doesn't provide a positive or negative association between eating vegetables and the odds of diabetes.


beta[9]: HvyAlcoholConsump (Strong Negative Effect)

Dot: Around -0.6

HDI: Entirely negative, not crossing zero.

Interpretation: Heavy alcohol consumption is associated with a decreased odds of diabetes. Also confusing.


beta[10]: AnyHealthcare (Effect Not Clearly Distinguishable from Zero)

Dot: Slightly positive.

HDI: Crosses zero.

Interpretation: No strong evidence of a consistent association.


beta[11]: NoDocbcCost (Effect Not Clearly Distinguishable from Zero)

Dot: Slightly positive.

HDI: Crosses zero.

Interpretation: No strong evidence of a consistent association.


beta[12]: GenHlth (Positive Effect)

Dot: Around +0.05

HDI: Entirely positive, not crossing zero.

Interpretation: Poorer perceived general health is associated with increased odds of diabetes As, GenHlth coded 5=extremely poor health and 1 means very good condition of health.


beta[13]: DiffWalk (Positive Effect)

Dot: Around +0.15

HDI: Entirely positive, not crossing zero.

Interpretation: Having difficulty walking is associated with increased odds of diabetes


beta[14]: Sex (Effect Not Clearly Distinguishable from Zero)

Dot: Slightly negative.

HDI: Crosses zero.

Interpretation: No strong evidence for a difference in diabetes odds based on sex


beta[15]: Age (Positive Effect)

Dot: Around +0.55

HDI: Entirely positive, not crossing zero.

Interpretation: Age is strongly associated with increased odds of diabetes.


beta[16]: Education (Effect Not Clearly Distinguishable from Zero)

Dot: Slightly negative.

HDI: Crosses zero.

Interpretation: No strong evidence for a consistent association between education level and diabetes odds.


beta[17]: Income (Effect Not Clearly Distinguishable from Zero)

Dot: Slightly negative.

HDI: Crosses zero.

Interpretation: No strong evidence for a consistent association between income level and diabetes odds.


beta[18]: BMI (Strong Positive Effect)

Dot: Around +0.55

HDI: Entirely positive, not crossing zero.

Interpretation: BMI is strongly associated with an increased odds of diabetes.


beta[19]: MentHlth (Effect Not Clearly Distinguishable from Zero)

Dot: Very close to zero.

HDI: Crosses zero.

Interpretation: No strong evidence of a consistent association between mental health and diabetes odds.


beta[20]: PhysHlth (Effect Not Clearly Distinguishable from Zero)

Dot: Very close to zero.

HDI: Crosses zero.

Interpretation: No strong evidence of a consistent association between physical health and diabetes odds.


Major Risk Factors: HighBP, HighChol, BMI, and Age are strongly associated with increased odds of diabetes.

Protective Factors: PhysActivity and Fruits show a consistent association with decreased odds of diabetes.

Confusing Findings: The strong positive effect for CholCheck and the negative effects for Smoker and HvyAlcoholConsump are particularly notable.

No Strong Evidence: Veggies, AnyHealthcare, NoDocbcCost, Sex, Education, Income, MentHlth, PhysHlth do not show strong evidence of a consistent association with diabetes odds, given the presence of the other predictors in the model. This doesn't mean they have no effect, but rather that the data doesn't strongly support one direction over the other within this multivariate model.

# **Comparison between LightGBM and Bayesian logistic regression**

`Bayesian Logistic Regression is a parametric, interpretable, probabilistic model that quantifies uncertainty in its coefficients.`

`LightGBM is a non-parametric, ensemble machine learning algorithm that is highly optimized for performance.`



`Performance Comparison:`


LightGBM Performance:
Accuracy: 0.75
Precision (Class 0): 0.78
Recall (Class 0): 0.70
Precision (Class 1): 0.73
Recall (Class 1): 0.80
F1-Score (Class 0): 0.74
F1-Score (Class 1): 0.76
AUC: 0.8276





`Interpreting LightGBM's Performance:`

`**Good Overall Performance:** The model achieves an accuracy of 75% and an AUC of 82.7%, indicating it is a reliable predictive tool. Generally, an AUC above 0.8 is considered strong for various applications.`

`**Balanced Recall:** The recall for class 1 (diabetes) is 0.80, which is slightly higher than the recall for class 0 (no diabetes) at 0.70. This means the model is somewhat better at identifying positive cases of diabetes among all actual positive instances.`

`**Higher Precision for Class 0:** When the model predicts no diabetes (class 0), it is more accurate, achieving a precision of 0.78. In contrast, when it predicts diabetes (class 1), the precision is slightly lower at 0.73.`

`Hypothesized Bayesian Logistic Regression Performance:`

`**Accuracy:** Bayesian Logistic Regression is likely to have slightly lower accuracy than LightGBM, which stands at 74%.`

`**AUC: **Likewise, the AUC for Bayesian Logistic Regression is expected to be slightly lower than LightGBM's AUC of 82.2%.`  

`**Metrics Balance:** It may exhibit a more balanced precision and recall across classes if the underlying relationships are primarily linear.`

`In this dataset LightGBM is highly effective at predicting individuals who are likely to have diabetes based on the available data. In contrast, Bayesian Logistic Regression is excellent at explaining the reasons behind a person's likelihood of having diabetes. It details the individual contributions of each risk factor and offers robust uncertainty estimates for these contributions.`

`In conclusion, using both methods can be highly effective: LightGBM for predictive benchmarking and Bayesian Logistic Regression for gaining deeper insights into the underlying relationships.`
"""