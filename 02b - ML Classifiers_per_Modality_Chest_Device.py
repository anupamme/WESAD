#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Subject Data

# In[2]:


BASE_PATH = 'segmented_data'


# In[76]:


# subjects_data = [pd.read_feather(os.path.join(BASE_PATH, subject)) for subject in os.listdir(BASE_PATH)]
# df = pd.concat(subjects_data)
# df.drop(columns=['index'], inplace=True)
# df.shape


# In[67]:


# df.to_csv('segmented_data/all_subjects.csv')


# In[3]:


df = pd.read_csv(f'{BASE_PATH}/all_subjects.csv', index_col=0)


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.columns


# ## Picking a modality
# 
# Prior executing the models, uncomment a modality from the following cells

# In[191]:


# modality = 'ACC'
# modality = 'ECG'
# modality = 'EDA'
# modality = 'EMG'
# modality = 'RESP'
modality = 'TEMP'


# In[30]:


df_report = pd.DataFrame()
df_report.loc['ACC', :] = []
df_report.loc['ECG', :] = []
df_report.loc['EDA', :] = []
df_report.loc['EMG', :] = []
df_report.loc['RESP', :] = []
df_report.loc['TEMP', :] = []


# In[192]:


mod_cols = df.columns[df.columns.str.startswith(modality)]


# ## Splitting data for CV

# In[193]:


X = df[mod_cols]
y = df['label']
y_b = y.copy().apply(lambda x: 1 if x == 2 else 0)  # stress vs non-stress
loso_cv = LeaveOneGroupOut()  # just to analyze training set sizes
for train, test in loso_cv.split(X, y, groups=df['subject']):
    print(train.shape)


# In[194]:


X.shape


# ## Hyperparameters

# In[195]:


n_estimators = 100
min_samples_split = 20
criterion = 'entropy'  # information gain
subjects = df['subject'].unique()


# In[196]:


def run_cv(clf, X, y, groups, cv, scoring=['accuracy', 'f1_macro'], return_train_score=True,
          return_estimator=True, n_jobs=-1):
    """
        More on cross validation: https://scikit-learn.org/stable/modules/cross_validation.html#
        More on scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """
    return cross_validate(clf, X, y, groups=groups, cv=cv, scoring=scoring, return_train_score=return_train_score,
                          return_estimator=return_estimator, n_jobs=n_jobs)
def raw_scores(cv):
    return np.mean(cv['test_accuracy']), np.mean(cv['test_f1_macro'])

def test_score(cv):
    return {
        'accuracy': f"{np.mean(cv['test_accuracy'])} +-{np.std(cv['test_accuracy'])}",
        'f1_macro': f"{np.mean(cv['test_f1_macro'])} +-{np.std(cv['test_f1_macro'])}"
    }

def min_max(acc_scores):
    print(np.argmax(acc_scores), np.amax(acc_scores))
    print(np.argmin(acc_scores), np.amin(acc_scores))


# # Random Forest

# In[197]:


def create_rf_pipeline(n_estimators=100, min_samples_split=20, criterion='entropy', n_jobs=-1):
    model_rf =  RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, 
                                  criterion=criterion, n_jobs=-1)
    return make_pipeline(StandardScaler(), model_rf)


# ### Baseline vs Stress vs Amusement

# In[198]:


clf_rf_all = create_rf_pipeline(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                criterion=criterion, n_jobs=-1)
cv_rf_all = run_cv(clf_rf_all, X, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_rf_all)


# In[199]:


sns.barplot(x=list(range(len(subjects))), y=cv_rf_all['test_accuracy']);


# In[200]:


min_max(cv_rf_all['test_accuracy'])


# ### Stress vs Non-Stress

# In[201]:


clf_b_rf_all = create_rf_pipeline(n_estimators=n_estimators, min_samples_split=min_samples_split,
                              criterion=criterion, n_jobs=-1)
cv_b_rf_all = run_cv(clf_b_rf_all, X, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
print(test_score(cv_b_rf_all))


# In[202]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_rf_all['test_accuracy']);


# In[203]:


min_max(cv_b_rf_all['test_accuracy'])


# In[204]:


rf_acc, rf_f1 = raw_scores(cv_rf_all)
rf_b_acc, rf_b_f1 = raw_scores(cv_b_rf_all)
df_report.loc[modality, 'rf_f1'] = rf_f1
df_report.loc[modality, 'rf_acc'] = rf_acc
df_report.loc[modality, 'rf_b_f1'] = rf_b_f1
df_report.loc[modality, 'rf_b_acc'] = rf_b_acc


# # AdaBoost

# In[205]:


def create_ab_pipeline(n_estimators=100, min_samples_split=20, criterion='entropy'):
    base_estimator = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split)
    model_ab = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
    return make_pipeline(StandardScaler(), model_ab)


# ### Baseline vs Stress vs Amusement

# In[206]:


clf_ab_all = create_ab_pipeline()
cv_ab_all = run_cv(clf_ab_all, X, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_ab_all)


# In[207]:


sns.barplot(x=list(range(len(subjects))), y=cv_ab_all['test_accuracy']);


# In[208]:


min_max(cv_ab_all['test_accuracy'])


# ### Stress vs Non-stress

# In[209]:


clf_b_ab_all = create_ab_pipeline()
cv_b_ab_all = run_cv(clf_b_ab_all, X, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_b_ab_all)


# In[210]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_ab_all['test_accuracy']);


# In[211]:


min_max(cv_b_ab_all['test_accuracy'])


# In[212]:


ab_acc, ab_f1 = raw_scores(cv_ab_all)
ab_b_acc, ab_b_f1 = raw_scores(cv_b_ab_all)
df_report.loc[modality, 'ab_f1'] = ab_f1
df_report.loc[modality, 'ab_acc'] = ab_acc
df_report.loc[modality, 'ab_b_f1'] = ab_b_f1
df_report.loc[modality, 'ab_b_acc'] = ab_b_acc


# # LDA

# In[213]:


def create_lda_pipeline():
    return make_pipeline(StandardScaler(), LDA())


# ### Baseline vs Stress vs Non-stress

# In[214]:


model_lda_all = create_lda_pipeline()
cv_lda_all = run_cv(model_lda_all, X, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_lda_all)


# In[215]:


sns.barplot(x=list(range(len(subjects))), y=cv_lda_all['test_accuracy']);


# In[216]:


min_max(cv_lda_all['test_accuracy'])


# ### Stress vs Non-Stress

# In[217]:


model_b_lda_all = create_lda_pipeline()
cv_b_lda_all = run_cv(model_b_lda_all, X, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_b_lda_all)


# In[218]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_lda_all['test_accuracy']);


# In[219]:


min_max(cv_b_lda_all['test_accuracy'])


# In[220]:


lda_acc, lda_f1 = raw_scores(cv_lda_all)
lda_b_acc, lda_b_f1 = raw_scores(cv_b_lda_all)
df_report.loc[modality, 'lda_f1'] = lda_f1
df_report.loc[modality, 'lda_acc'] = lda_acc
df_report.loc[modality, 'lda_b_f1'] = lda_b_f1
df_report.loc[modality, 'lda_b_acc'] = lda_b_acc


# ## Results

# In[221]:


df_report


# In[222]:


df_report.to_csv('chest_scores_per_mod.csv')


# In[1]:


# df_report.to_markdown()


# In[ ]:




