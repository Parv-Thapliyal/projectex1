import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Exploratory Data Analysis (EDA)

df = pd.read_csv('diabetes.csv')

print("Shape of the dataset:")
print(df.shape)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nData types and non-null values:")
df.info()

print("\nChecking for null values:")
print(df.isnull().sum())

print("\nStatistical summary of the dataset:")
print(df.describe())

print("\nDistribution of the Outcome variable:")
print(df['Outcome'].value_counts())

# Data Cleaning and Preprocessing

df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

print("Statistical summary after replacing zeros:")
print(df.describe())

# Data Visualization

plt.style.use('seaborn-v0_8-darkgrid')

# --- Figure 1: Combined Statistical Plots ---
# This figure will hold the countplot, heatmap, boxplots, and scatterplot.
fig1 = plt.figure(figsize=(20, 18)) # Adjust figure size for better visibility

# Subplot 1: Distribution of Outcome
ax1 = fig1.add_subplot(3, 2, 1) # 3 rows, 2 columns, 1st subplot
sns.countplot(x='Outcome', data=df, ax=ax1)
ax1.set_title('Distribution of Outcome (0: Non-Diabetic, 1: Diabetic)')

# Subplot 2: Correlation Heatmap
ax2 = fig1.add_subplot(3, 2, 2) # 3 rows, 2 columns, 2nd subplot
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f', ax=ax2)
ax2.set_title('Correlation Heatmap')

# Subplot 3: Glucose Levels by Outcome
ax3 = fig1.add_subplot(3, 2, 3) # 3 rows, 2 columns, 3rd subplot
sns.boxplot(x='Outcome', y='Glucose', data=df, ax=ax3)
ax3.set_title('Glucose Levels by Outcome')

# Subplot 4: BMI by Outcome
ax4 = fig1.add_subplot(3, 2, 4) # 3 rows, 2 columns, 4th subplot
sns.boxplot(x='Outcome', y='BMI', data=df, ax=ax4)
ax4.set_title('BMI by Outcome')

# Subplot 5: Age vs. Glucose Scatterplot (spanning both columns in the last row)
ax5 = fig1.add_subplot(3, 1, 3) # 3 rows, 1 column, 3rd subplot
sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=df, alpha=0.7, ax=ax5)
ax5.set_title('Age vs. Glucose by Outcome')

plt.tight_layout() # Adjust layout to prevent subplots from overlapping

# --- Figure 2: Histograms of All Features ---
# df.hist() is best used independently as it automatically creates a grid of subplots
# for all numerical columns. It returns an array of axes.
fig2 = df.hist(bins=20, figsize=(15, 10))
# Set a suptitle for this new figure of histograms
plt.suptitle('Histograms of All Features', y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap

# Display all generated figures. The script will pause until all plot windows are closed.
plt.show()

# Data Splitting and Scaling

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training & Evaluation

# Model 1: Logistic Regression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

print("Logistic Regression - Baseline Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("\nBaseline Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Hyperparameter Tuning with GridSearchCV

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}
grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train_scaled, y_train)

print("Best Parameters for Logistic Regression:", grid_lr.best_params_)

y_pred_lr_tuned = grid_lr.predict(X_test_scaled)
print("\nLogistic Regression - Tuned Accuracy:", accuracy_score(y_test, y_pred_lr_tuned))
print("\nTuned Classification Report:\n", classification_report(y_test, y_pred_lr_tuned))

# Model 2: Support Vector Classifier (SVC)

svc = SVC(random_state=42)
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test_scaled)

print("SVC - Baseline Accuracy:", accuracy_score(y_test, y_pred_svc))
print("\nBaseline Classification Report:\n", classification_report(y_test, y_pred_svc))

# Hyperparameter Tuning with GridSearchCV

param_grid_svc = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid_svc = GridSearchCV(SVC(random_state=42), param_grid_svc, cv=5, scoring='accuracy')
grid_svc.fit(X_train_scaled, y_train)

print("Best Parameters for SVC:", grid_svc.best_params_)

y_pred_svc_tuned = grid_svc.predict(X_test_scaled)
print("\nSVC - Tuned Accuracy:", accuracy_score(y_test, y_pred_svc_tuned))
print("\nTuned Classification Report:\n", classification_report(y_test, y_pred_svc_tuned))

# Model 3: Random Forest Classifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("Random Forest - Baseline Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nBaseline Classification Report:\n", classification_report(y_test, y_pred_rf))

# Hyperparameter Tuning with GridSearchCV

param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], 'min_samples_leaf': [1, 2, 4]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train_scaled, y_train)

print("Best Parameters for Random Forest:", grid_rf.best_params_)

y_pred_rf_tuned = grid_rf.predict(X_test_scaled)
print("\nRandom Forest - Tuned Accuracy:", accuracy_score(y_test, y_pred_rf_tuned))
print("\nTuned Classification Report:\n", classification_report(y_test, y_pred_rf_tuned))

# Model 4: Gradient Boosting Classifier

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)

print("Gradient Boosting - Baseline Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nBaseline Classification Report:\n", classification_report(y_test, y_pred_gb))

# Hyperparameter Tuning with GridSearchCV
param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5, scoring='accuracy')
grid_gb.fit(X_train_scaled, y_train)

print("Best Parameters for Gradient Boosting:", grid_gb.best_params_)

y_pred_gb_tuned = grid_gb.predict(X_test_scaled)
print("\nGradient Boosting - Tuned Accuracy:", accuracy_score(y_test, y_pred_gb_tuned))
print("\nTuned Classification Report:\n", classification_report(y_test, y_pred_gb_tuned))
import joblib

joblib.dump(grid_gb, 'diabetes_gb_model.pkl')

# Save the fitted scaler
joblib.dump(scaler, 'diabetes_scaler.pkl')

print("Model and scaler saved successfully.")
print("End")