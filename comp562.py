#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# # Set the kernel to use
# # Replace 'tf' with the name of your kernel
# import os
# os.environ['JUPYTER_KERNEL_NAME'] = 'tf'


# In[ ]:


df = pd.read_csv('stroke.csv')


# In[ ]:


df


# In[ ]:


df.drop('id',axis=1,inplace=True)


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


df_continuous = df[['age', 'avg_glucose_level', 'bmi']]


# In[ ]:


sns.heatmap(df_continuous.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


for column in df.columns:
    if column != 'stroke':
        if df[column].dtype != 'O':
            sns.displot(df[column], kde=False, bins=10)
            plt.title(f'Distribution of {column}')
            plt.show()
        else:
            sns.countplot(data=df, x=column)
            plt.title(f'Count of {column}')
            plt.show()


# In[ ]:


num_other_gender = df[df['gender'] == 'Other'].shape[0]

# Print the number of rows where the gender is 'Other'
print(f'The number of rows where the gender is Other is {num_other_gender}')

#remove gender = other 
df =df[df['gender'] != 'Other']


# In[ ]:


df_unknownsmoker = df[df['smoking_status'] == 'Unknown']
stroke_counts_unknownsmoker = df_unknownsmoker['stroke'].value_counts()
print(stroke_counts_unknownsmoker)


# In[ ]:


# Calculate the mean of non-NaN values in the bmi column
bmi_mean = df['bmi'].mean()
print(bmi_mean)

stroke_mean_bmi = df.loc[df['stroke'] == 1, 'bmi'].mean()
print(stroke_mean_bmi)

nonstroke_mean_bmi = df.loc[df['stroke'] == 0, 'bmi'].mean()
print(nonstroke_mean_bmi)

df.loc[(df['stroke'] == 1) & (df['bmi'].isnull()), 'bmi'] = stroke_mean_bmi
df.loc[(df['stroke'] == 0) & (df['bmi'].isnull()), 'bmi'] = nonstroke_mean_bmi

df.isnull().sum()

# Fill the NaN values in the bmi column with the mean value
#df['bmi'].fillna(bmi_mean, inplace=True)


# In[ ]:


# Plot a pie chart of stroke values before removing unknown smokers
stroke_counts = df['stroke'].value_counts()
print(stroke_counts)
plt.pie(stroke_counts, labels=['No Stroke', 'Stroke'], autopct='%1.1f%%')
plt.title('Stroke Value Counts')
plt.show()


# In[ ]:


# Removing unknown smokers
print((df['smoking_status'] == 'Unknown').sum())
df= df[df['smoking_status'] != 'Unknown']


# Plot a pie chart of stroke values
stroke_counts = df['stroke'].value_counts()
plt.pie(stroke_counts, labels=['No Stroke', 'Stroke'], autopct='%1.1f%%')
plt.title('Stroke Value Counts')
plt.show()


# In[ ]:


df


# In[ ]:


#Crossing age and bmi due to moderate correlation to capture any interactions
df['age_bmi'] = df['age'] * df['bmi']


# In[ ]:


df


# ## Chi-Squared Statistic
# We conduct the Pearson's Chi-Squared Statistic to test for independence between categorical variables. This is to conclude whether two variables (categorical and the target variable stroke) are related to each other. Null Hypothesis (H0): There is no relationship between the variables Alternative Hypothesis (H1): There is a statistically significant relationship between the variables.
# 

# In[ ]:


#pip install stats


# In[ ]:


from scipy import stats
chi_table = pd.DataFrame(columns=["Category", "P-Value",'Chi Square Test Stat', "Conclusion"])
def find_dep(p_value): 
    alpha = 0.05
    if p_value <= alpha: 
        return "Dependent (reject H0)"
    else: 
        return "Independent(Do not reject H0)"
## get the 
cat_variables = ["gender", "hypertension", "heart_disease", "ever_married", "work_type","Residence_type","smoking_status"]
chi_lists = []

for column in cat_variables:
    contigency = pd.crosstab(df[column], df['stroke'])
    stat, p_value, dof, expected = stats.chi2_contingency(contigency)
    conclusion = find_dep(p_value)
    each_col = [column, p_value, stat, conclusion]
    chi_lists.append(each_col)

for i in chi_lists:
    chi_table.loc[len(chi_table)] = i
chi_table


# In[ ]:


#pip install imbalanced-learn


# ## Point Biserial Correlation
# Point-biserial correlation is used to measure the relationship between a binary variable, x, and a continuous variable, y. We use this correlation strategy to see the level of correlation between the continuous variables and the target variable, stroke

# In[ ]:


biser_table = pd.DataFrame(columns=["Category", "Biserial Stats",'P_value', "Conclusion"])
cont_var = ["age", "bmi", "avg_glucose_level", "age_bmi"]
bi_lists = []
for var in cont_var:
   stat, p = stats.pointbiserialr(df[var], df["stroke"])
   each_val = [var, stat, p, find_dep(p)]
   bi_lists.append(each_val)

for i in bi_lists:
    biser_table.loc[len(biser_table)] = i
biser_table


# ## One Hot Encoding

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder()
# one_hot = encoder.fit_transform(df[['gender','ever_married','work_type','Residence_type','smoking_status']])


df = pd.get_dummies(df, columns = ['gender','ever_married','work_type','Residence_type','smoking_status']).astype(int)
df


# ## Scaling

# In[ ]:


#Performing standardization on continuous variables
from sklearn.preprocessing import StandardScaler

# Fit the StandardScaler to the training data
scaler = StandardScaler()
df[['age', 'avg_glucose_level', 'bmi', 'age_bmi']] = scaler.fit_transform(df[['age', 'avg_glucose_level', 'bmi', 'age_bmi']])


# ## Hypothesis Testing

# In[ ]:


numeric = df.select_dtypes(include=np.number).columns.tolist()
tstats_df = pd.DataFrame()
warnings.filterwarnings("ignore")

for eachvariable in numeric:
    tstats = stats.ttest_ind(df.loc[df["stroke"] == 1, eachvariable], df.loc[df["stroke"] == 0, eachvariable])
    temp = pd.DataFrame([eachvariable, tstats[0], tstats[1]]).T
    temp.columns = ["Variable Name", "T stats", " P-value"]
    tstats_df = pd.concat([tstats_df, temp], axis = 0, ignore_index= True)
tstats_df = tstats_df.sort_values(by=" P-value").reset_index(drop=True)
print(tstats_df)


# At the 0.05 significant level, all the variables are statistically significant since the p-value < 0.05

# In[ ]:


import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import  pyplot
from sklearn.model_selection import train_test_split


# In[ ]:


np.random.seed(120)
y_feature_selection = df['stroke'] # dependent variable
x_feature_selection = df.drop(columns=["stroke"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(x_feature_selection, y_feature_selection, test_size=0.33, random_state=7)
model = XGBClassifier()
model.fit(x_feature_selection, y_feature_selection)
# plot feature importance
plot_importance(model)
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# num_cols = df.select_dtypes(include=['float64', 'int64']).columns
# corr_matrix = df[num_cols].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

#roughly choose variables to check correlation based on feature importance from  xgboost above
variables_to_plot = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'age_bmi', 'gender_Female', 'gender_Male', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes', 'Residence_type_Rural', 'Residence_type_Urban', 'stroke']
corr_matrix2 = df[variables_to_plot].corr()

#plot after scaling for more accurate correlation
sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', annot_kws={"fontsize":6})
plt.show()


# # Removing Residence Type, Gender and BMI

# In[ ]:


#removing residence_type
#to_drop = ['Residence_type','gender','bmi']
to_drop = ['Residence_type_Rural', 'Residence_type_Urban', 'gender_Male', 'gender_Female', 'bmi']
df2= df.drop(to_drop, axis=1)
df2


# In[ ]:


df


# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

X_df2 = df2.drop('stroke', axis=1)
y_df2 = df2['stroke']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_df2, y_df2, test_size=0.3, random_state=109)

df


# In[ ]:


from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import  pyplot


# In[ ]:


#without resampling (unbalanced dataset)

# # Define model
# xgb_model_unsampled = XGBClassifier()

# # Train model
# xgb_model_unsampled.fit(X_train, y_train)

# # Evaluate model
# y_pred_unsampled = xgb_model_unsampled.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred_unsampled)
# f1 = f1_score(y_test, y_pred_unsampled)
# precision = precision_score(y_test, y_pred_unsampled)
# recall = recall_score(y_test, y_pred_unsampled)

# print('Accuracy Unsampled: %.2f%%' % (accuracy * 100.0))
# print('F1 Score Unsampled: %.2f%%' % (f1 * 100.0))
# print('Precision Unsampled: %.2f%%' % (precision * 100.0))
# print('Recall Unsampled: %.2f%%' % (recall * 100.0))

#Define hyperparameters to tune
param_grid = {
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [3, 5, 10, 15],
    'n_estimators': [50, 100, 150],
    'subsample': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 2, 3]
}

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Retrain model with best hyperparameters
unsampled_best = XGBClassifier(**grid_search.best_params_)
unsampled_best.fit(X_train, y_train)

# Evaluate best model
y_pred_unsampled2 = unsampled_best.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred_unsampled2)
f12 = f1_score(y_test, y_pred_unsampled2)
precision2 = precision_score(y_test, y_pred_unsampled2)
recall2 = recall_score(y_test, y_pred_unsampled2)

print('Accuracy with best hyperparameters: %.2f%%' % (accuracy2 * 100.0))
print('F1 Score with best hyperparameters: %.2f%%' % (f12 * 100.0))
print('Precision with best hyperparameters: %.2f%%' % (precision2 * 100.0))
print('Recall with best hyperparameters: %.2f%%' % (recall2 * 100.0))


# In[ ]:


#after resampling
from copy import deepcopy
from imblearn.over_sampling import SMOTE

# Instantiate SMOTE object
smote = SMOTE(random_state = 42, sampling_strategy = 0.5)

# Resample the data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print the number of samples in each class before and after resampling
print(f"Before resampling: \n{y_train.value_counts()}")
print(f"After resampling: \n{y_resampled.value_counts()}")


# In[ ]:


# for dataset after dropping selected columns
from copy import deepcopy
from imblearn.over_sampling import SMOTE


# Instantiate SMOTE object
smote = SMOTE(random_state = 42, sampling_strategy = 0.5)

# Resample the data
X_resampled2, y_resampled2 = smote.fit_resample(X_train2, y_train2)

# Print the number of samples in each class before and after resampling
print(f"Before resampling: \n{y_train2.value_counts()}")
print(f"After resampling: \n{y_resampled.value_counts()}")


# In[ ]:


from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report 

def get_model_results(classifier, model, x_train, y_train, x_test, y_test):
    # fit the model with data 
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Results for "+ classifier)
    print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
    print("Recall:", metrics.recall_score(y_test,y_pred))
    print("Precision:", metrics.precision_score(y_test,y_pred))
    print("F1 Score:", metrics.f1_score(y_test,y_pred))

    # confusion matrix 
    cfn_matrix = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])
    print("\nConfusion Matrix:\n")
    print(cfn_matrix)

    # Classification Report 
    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))
    
    accuracy = metrics.accuracy_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    f1 = metrics.f1_score(y_test,y_pred)
    results = [accuracy, recall, precision, f1]

    print(results)
    # print the auc curve and show auc score 
    #plot roc curve 
    import matplotlib.pyplot as plt 
    # predicted probabilities of class 1 
    by_pred_prob_model = model.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, by_pred_prob_model)
    auc = metrics.roc_auc_score(y_test, by_pred_prob_model)
    plt.plot(fpr,tpr,label="XGB_Model, auc ="+str("{:.3f}".format(auc)))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.legend(loc=4)
    plt.show()
    ab_auc = auc
    print("AUC Score:" , ab_auc)


    return


# # XGBOOST
# 

# In[ ]:


#xgb model before dropping variables
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb_model = XGBClassifier()
#Define hyperparameters to tune
param_grid = {
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [3, 5, 10, 15],
    'n_estimators': [50, 100, 150],
    'subsample': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 2, 3]
}

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring = 'f1')
grid_search.fit(X_resampled, y_resampled)

# Print best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Retrain model with best hyperparameters
xgb_best = XGBClassifier(**grid_search.best_params_)
xgb_best.fit(X_resampled, y_resampled)

# Evaluate best model
y_pred_xgb = xgb_best.predict(X_test)

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)

print('Accuracy with best hyperparameters: %.2f%%' % (xgb_accuracy * 100.0))
print('F1 Score with best hyperparameters: %.2f%%' % (xgb_f1 * 100.0))
print('Precision with best hyperparameters: %.2f%%' % (xgb_precision * 100.0))
print('Recall with best hyperparameters: %.2f%%' % (xgb_recall * 100.0))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

# Calculate ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgb2)
# roc_auc = auc(fpr, tpr)

# Plot ROC curve
# plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# confusion matrix
# from sklearn.metrics import confusion_matrix
# matrix = confusion_matrix(y_test, y_pred_xgb2)
# print(matrix)

get_model_results('XGB', xgb_best, X_resampled, y_resampled, X_test, y_test)


# In[ ]:


#xgb model after dropping residence_type,bmi,gender
xgb_model = XGBClassifier()

#Define hyperparameters to tune
param_grid = {
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [3, 5, 10, 15],
    'n_estimators': [50, 100, 150],
    'subsample': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 2, 3]
}

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5)
grid_search.fit(X_resampled2, y_resampled2)

# Print best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Retrain model with best hyperparameters
xgb_best2 = XGBClassifier(**grid_search.best_params_)
xgb_best2.fit(X_resampled2, y_resampled2)

# Evaluate best model
y_pred_xgb2 = xgb_best2.predict(X_test2)

xgb_accuracy2 = accuracy_score(y_test2, y_pred_xgb2)
xgb_f12 = f1_score(y_test2, y_pred_xgb2)
xgb_precision2 = precision_score(y_test2, y_pred_xgb2)
xgb_recall2 = recall_score(y_test2, y_pred_xgb2)

print('Accuracy with best hyperparameterss: %.2f%%' % (xgb_accuracy2 * 100.0))
print('F1 Score with best hyperparameters: %.2f%%' % (xgb_f12 * 100.0))
print('Precision with best hyperparameters: %.2f%%' % (xgb_precision2 * 100.0))
print('Recall with best hyperparameters: %.2f%%' % (xgb_recall2 * 100.0))


# In[ ]:


# # Calculate ROC curve and AUC
# fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_pred_best2)
# roc_auc2 = auc(fpr2, tpr2)
# print(roc_auc2)

# # Plot ROC curve
# plt.plot(fpr2, tpr2, label='AUC = %0.2f' % roc_auc2)
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

get_model_results('XGB', xgb_best2, X_resampled2, y_resampled2, X_test2, y_test2)


# # SVM

# In[ ]:


#hyperparameter tuning for svm1
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import svm
from sklearn.ensemble import BaggingClassifier

n_estimators = 10

param_grid = {
    'base_estimator__C': [0.1,1, 10, 100, 1000], 
    'base_estimator__gamma': [1,0.1,0.01,0.001, 0.0001],
    'base_estimator__kernel':  ['linear', 'poly', 'rbf', 'sigmoid']
}


svm = svm.SVC()

bagging_svm = BaggingClassifier(svm, random_state = 88,max_samples=1.0 / n_estimators, n_estimators=n_estimators)

grid_search = GridSearchCV(
    bagging_svm,
    param_grid=param_grid,
    scoring = 'f1'
)

grid_search.fit(X_resampled, y_resampled)


print('Best hyper parameters:', grid_search.best_params_, 'Score', grid_search.best_score_)


# In[ ]:


#svm before dropping variables
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report


svm1 = BaggingClassifier(svm.SVC(kernel=grid_search.best_params_['base_estimator__kernel'], gamma = grid_search.best_params_['base_estimator__gamma'], C = grid_search.best_params_['base_estimator__C'] ), random_state = 88,max_samples=1.0 / n_estimators, n_estimators=n_estimators )
svm1.fit(X_resampled, y_resampled)
y_pred_SVM = svm1.predict(X_test)
get_model_results('SVM', svm1, X_resampled, y_resampled, X_test, y_test)


# In[ ]:


#hyperparameter tuning for svm2
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import svm
from sklearn.ensemble import BaggingClassifier

n_estimators = 10

param_grid = {
    'base_estimator__C': [0.1,1, 10, 100, 1000], 
    'base_estimator__gamma': [1,0.1,0.01,0.001, 0.0001],
    'base_estimator__kernel':  ['linear', 'poly', 'rbf', 'sigmoid']
}


svm2 = svm.SVC()

bagging_svm = BaggingClassifier(svm, random_state = 88,max_samples=1.0 / n_estimators, n_estimators=n_estimators)

grid_search2 = GridSearchCV(
    bagging_svm,
    param_grid=param_grid,
    scoring = 'f1'
)

grid_search2.fit(X_resampled2, y_resampled2)


print('Best hyper parameters:', grid_search2.best_params_, 'Score', grid_search2.best_score_)


# In[ ]:


#svm after dropping variables
from sklearn import svm

svm2 = BaggingClassifier(svm.SVC(kernel=grid_search2.best_params_['base_estimator__kernel'], gamma = grid_search2.best_params_['base_estimator__gamma'], C = grid_search2.best_params_['base_estimator__C'] ), random_state = 88,max_samples=1.0 / n_estimators, n_estimators=n_estimators, )
svm2.fit(X_resampled2, y_resampled2)
y_pred_SVM2 = svm2.predict(X_test2)
get_model_results('SVM with dropped columns', svm2, X_resampled2, y_resampled2, X_test2, y_test2)


# # LOG REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5,scoring = 'f1')
grid_search.fit(X_resampled, y_resampled)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


# In[ ]:


logreg = LogisticRegression(C=10, penalty='l2')
# Fit the logistic regression model to the entire training set
logreg.fit(X_resampled, y_resampled)
y_pred_logreg= logreg.predict(X_test)


# In[ ]:


get_model_results("Logistic Regression ", logreg, X_resampled, y_resampled, X_test, y_test)


# In[ ]:


feature_names = X_resampled.columns
coefficients = logreg.coef_[0]
from prettytable import PrettyTable


table = PrettyTable()
table.field_names = ['Feature Name', 'Coefficient']
for feature, coef in zip(feature_names, coefficients):
    table.add_row([feature, coef])

# Print the table
print(table)

plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xticks(feature_names, rotation=45)
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Logistic Regression Coefficients')
plt.show()


# In[ ]:


param_grid2 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
grid_search2 = GridSearchCV(LogisticRegression(), param_grid2, cv=5, scoring = 'f1')
grid_search2.fit(X_resampled2, y_resampled2)
best_params2 = grid_search2.best_params_
print("Best Hyperparameters:", best_params2)


# In[ ]:


logreg2 = LogisticRegression(**best_params2)

# Fit the logistic regression model to the entire training set
logreg2.fit(X_resampled2, y_resampled2)
y_pred_logreg2= logreg2.predict(X_test2)


# In[ ]:


get_model_results("Logistic Regression ", logreg2, X_resampled2, y_resampled2, X_test2, y_test2)


# In[ ]:


feature_names2 = X_resampled2.columns
coefficients2 = logreg2.coef_[0]
from prettytable import PrettyTable


table2 = PrettyTable()
table2.field_names = ['Feature Name', 'Coefficient']
for feature2, coef2 in zip(feature_names2, coefficients2):
    table2.add_row([feature2, coef2])

# Print the table
print(table2)

plt.figure(figsize=(10, 6))
plt.bar(feature_names2, coefficients2)
plt.xticks(feature_names2, rotation=45)
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Logistic Regression Coefficients')
plt.show()


# # AdaBoost

# In[ ]:


# import AdaBoost Model 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report 


# In[ ]:


grid_params = {
    'n_estimators': [50,100,200,400],
    'algorithm': ['SAMME','SAMME.R'],
    'learning_rate': [0.001,0.05,0.1,0.2],
}


# In[ ]:


abModel = AdaBoostClassifier(random_state=23)
gridCV = GridSearchCV(abModel, param_grid=grid_params, verbose=False)
gridCV.fit(X_resampled, y_resampled)
print('Best hyper parameters:', gridCV.best_params_, 'Score', gridCV.best_score_)


# In[ ]:


good_AB = AdaBoostClassifier(n_estimators = 400, learning_rate = 0.2, algorithm = 'SAMME.R', random_state=23)


# In[ ]:


get_model_results("Ada Boosting", good_AB, X_resampled, y_resampled, X_test, y_test)


# # Ada Boost on dropped column dataset

# In[ ]:


grid_params = {
    'n_estimators': [50,100,200,400],
    'algorithm': ['SAMME','SAMME.R'],
    'learning_rate': [0.01,0.05,0.1,0.2],
}


# In[ ]:


abModel = AdaBoostClassifier(random_state=23)
gridCV = GridSearchCV(abModel, param_grid=grid_params, verbose=False)
gridCV.fit(X_resampled2, y_resampled2)
print('Best hyper parameters:', gridCV.best_params_, 'Score', gridCV.best_score_)


# In[ ]:


AB_dropped = AdaBoostClassifier(n_estimators = 400, learning_rate = 0.2, algorithm = 'SAMME.R', random_state=23)


# In[ ]:


get_model_results("Ada Boosting", AB_dropped, X_resampled2, y_resampled2, X_test2, y_test2)


# # Neural Network

# In[ ]:


import keras.layers as layers
import keras.models
import tensorflow as tf


# In[ ]:


pip install scikeras


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scikeras.wrappers import KerasClassifier

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = keras.models.Sequential()
    model.add(layers.Dense(11, activation='relu', input_dim=20))
    model.add(layers.Dense(7, activation='relu', input_dim=20))
    model.add(layers.Dense(5, activation='relu', input_dim=20))
    model.add(layers.Dense(1, activation='sigmoid', name='predictions'))
    # return model without compile
    return model

# fix random seed for reproducibility
tf.random.set_seed(109)

# hyperparameter tuning

# create model
model = KerasClassifier(model=create_model, loss="binary_crossentropy", verbose=False, optimizer = keras.optimizers.Adam(lr=1e-5))

# define the grid search parameters
epochs = [250, 500, 750, 1000]
batch_size = [10, 20, 40, 60, 80, 100]
param_grid = dict(epochs=epochs, batch_size=batch_size)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1')
grid_result = grid.fit(X_resampled, y_resampled)


# In[ ]:


# summarize results
print('Best hyper parameters:', grid.best_params_, 'Score', grid.best_score_)


# In[ ]:


model = keras.models.Sequential()
model.add(layers.Dense(11, activation='relu'))
model.add(layers.Dense(7, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid', name='predictions'))


# In[ ]:


model.compile(loss='binary_crossentropy', 
              optimizer=keras.optimizers.Adam(lr=1e-5), 
              metrics=['accuracy',
                       keras.metrics.Precision(name='precision'),
                       keras.metrics.Recall(name='recall')])


# In[ ]:


model.fit(X_resampled, y_resampled, epochs=1000, batch_size=10)


# In[ ]:


predictions = model.predict(X_test)
y_pred = [
    1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
]


# In[ ]:


def get_model_results_nn(classifier, predictions, y_pred, y_test):
    
    print("Results for "+ classifier)
    print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
    print("Recall:", metrics.recall_score(y_test,y_pred))
    print("Precision:", metrics.precision_score(y_test,y_pred))
    print("F1 Score:", metrics.f1_score(y_test,y_pred))

    # confusion matrix 
    cfn_matrix = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])
    print("\nConfusion Matrix:\n")
    print(cfn_matrix)

    # Classification Report 
    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))
    
    accuracy = metrics.accuracy_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    f1 = metrics.f1_score(y_test,y_pred)
    results = [accuracy, recall, precision, f1]

    print(results)
    # print the auc curve and show auc score 
    
    # predicted probabilities of class 1 
    by_pred_prob_model = predictions.ravel()
    fpr, tpr, _ = metrics.roc_curve(y_test, by_pred_prob_model)
    auc = metrics.roc_auc_score(y_test, by_pred_prob_model)
    
    # plot roc curve 
    plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


# In[ ]:


from sklearn import metrics
from sklearn.metrics import classification_report
get_model_results_nn("Neural Network", predictions, y_pred, y_test)


# ## Dataset without Residence Type

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scikeras.wrappers import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model2():
    # create model
    model = keras.models.Sequential()
    model.add(layers.Dense(11, activation='relu', input_dim=15))
    model.add(layers.Dense(7, activation='relu', input_dim=15))
    model.add(layers.Dense(5, activation='relu', input_dim=15))
    model.add(layers.Dense(1, activation='sigmoid', name='predictions'))
    # return model without compile
    return model

# fix random seed for reproducibility
tf.random.set_seed(109)

# hyperparameter tuning
# create model
model2 = KerasClassifier(model=create_model2, loss="binary_crossentropy", verbose=False, optimizer = keras.optimizers.Adam(lr=1e-5))

epochs = [500, 1000]
batch_size = [10, 40, 80]
param_grid = dict(epochs=epochs, batch_size=batch_size)

grid2 = GridSearchCV(estimator=model2, param_grid=param_grid, scoring='f1')
grid_result2 = grid2.fit(X_resampled2, y_resampled2)

# grid2 = GridSearchCV(estimator=model2, param_grid=param_grid, scoring='f1')
# grid_result2 = grid2.fit(X_resampled2, y_resampled2)


# In[ ]:


# summarize results
print('Best hyper parameters:', grid2.best_params_, 'Score', grid2.best_score_)


# In[ ]:


model2 = keras.models.Sequential()
model2.add(layers.Dense(11, activation='relu'))
model2.add(layers.Dense(7, activation='relu'))
model2.add(layers.Dense(5, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid', name='predictions'))


# In[ ]:


model2.compile(loss='binary_crossentropy', 
               optimizer=keras.optimizers.Adam(lr=1e-5), 
               metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall')])


# In[ ]:


# fit the model with data 
model2.fit(X_resampled2, y_resampled2, epochs=1000, batch_size=20)

predictions2 = model2.predict(X_test2)
y_pred2 = [
    1 if prob > 0.5 else 0 for prob in np.ravel(predictions2)
]


# In[ ]:


from sklearn import metrics
from sklearn.metrics import classification_report

get_model_results_nn("Neural Network 2", predictions2, y_pred2, y_test2)

