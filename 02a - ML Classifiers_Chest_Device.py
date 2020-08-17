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


# In[3]:


# subjects_data = [pd.read_feather(os.path.join(BASE_PATH, subject)) for subject in os.listdir(BASE_PATH)]
# df = pd.concat(subjects_data)
# df.drop(columns=['index'], inplace=True)
# df.shape


# In[4]:


# df.to_csv('segmented_data/all_subjects.csv')


# In[5]:


df = pd.read_csv(f'{BASE_PATH}/all_subjects.csv', index_col=0)


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.columns


# In[9]:


with pd.option_context("display.max_rows", df.columns.shape[0]): 
        display(df.head().T)


# In[10]:


with pd.option_context("display.max_rows", df.columns.shape[0], "display.max_columns", 100):
    display(df.describe().T)


# ## Splitting data for CV

# In[11]:


X = df.drop(columns=['label', 'subject'])
filter_cols = df.columns[df.columns.str.startswith('ACC')]
X_p = X.drop(columns=df.columns[df.columns.str.startswith('ACC')].values)  # Physiological modalities only
y = df['label']
y_b = y.copy().apply(lambda x: 1 if x == 2 else 0)  # stress vs non-stress
loso_cv = LeaveOneGroupOut()  # just to analyze training set sizes
for x_idxs, y_idxs in loso_cv.split(X, y, groups=df['subject']):
    print(x_idxs.shape)


# ## Hyperparameters

# In[12]:


n_estimators = 100
min_samples_split = 20
criterion = 'entropy'  # information gain
subjects = df['subject'].unique()


# In[26]:


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
        'accuracy': f"{np.mean(cv['test_accuracy']) * 100} +-{np.std(cv['test_accuracy']) * 100}",
        'f1_macro': f"{np.mean(cv['test_f1_macro']) * 100} +-{np.std(cv['test_f1_macro']) * 100}"
    }

def min_max(acc_scores):
    print(np.argmax(acc_scores), np.amax(acc_scores))
    print(np.argmin(acc_scores), np.amin(acc_scores))


# ## Experimental setup

# In[53]:


df_report_all = pd.DataFrame()
df_report_phy = pd.DataFrame()


# In[365]:


experiment_idx = df_report_all.shape[0]
# df_report_all.loc[experiment_idx, :] = []
# df_report_phy.loc[experiment_idx, :] = []
# df_report_all = df_report_all.append([[]])
# df_report_phy = df_report_phy.append([[]])
f'Experiment No {experiment_idx + 1}'


# # Random Forest

# In[366]:


def create_rf_pipeline(n_estimators=100, min_samples_split=20, criterion='entropy', n_jobs=-1):
    model_rf =  RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, 
                                  criterion=criterion, n_jobs=-1)
    return make_pipeline(StandardScaler(), model_rf)


# ### Baseline vs Stress vs Amusement

# #### All modalities

# In[367]:


clf_rf_all = create_rf_pipeline(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                criterion=criterion, n_jobs=-1)
cv_rf_all = run_cv(clf_rf_all, X, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_rf_all)


# In[368]:


sns.barplot(x=list(range(len(subjects))), y=cv_rf_all['test_accuracy']);


# In[369]:


min_max(cv_rf_all['test_accuracy'])


# In[370]:


sns.barplot(x=list(range(len(subjects))), y=cv_rf_all['test_f1_macro']);


# In[371]:


min_max(cv_rf_all['test_f1_macro'])


# #### Physiological modalities

# In[372]:


clf_rf_phy = create_rf_pipeline(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                criterion=criterion, n_jobs=-1)
cv_rf_phy = run_cv(clf_rf_phy, X_p, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_rf_phy)


# In[373]:


sns.barplot(x=list(range(len(subjects))), y=cv_rf_phy['test_accuracy']);


# In[374]:


sns.barplot(x=list(range(len(subjects))), y=cv_rf_phy['test_f1_macro']);


# ### Stress vs Non-Stress

# In[375]:


y_b.value_counts()


# #### All modalities

# In[376]:


clf_b_rf_all = create_rf_pipeline(n_estimators=n_estimators, min_samples_split=min_samples_split,
                              criterion=criterion, n_jobs=-1)
cv_b_rf_all = run_cv(clf_b_rf_all, X, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
print(test_score(cv_b_rf_all))


# In[377]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_rf_all['test_accuracy']);


# In[378]:


min_max(cv_b_rf_all['test_accuracy'])


# In[379]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_rf_all['test_f1_macro']);


# In[380]:


min_max(cv_b_rf_all['test_f1_macro'])


# #### Physiological modalities

# In[381]:


clf_b_rf_phy = create_rf_pipeline(n_estimators=n_estimators, min_samples_split=min_samples_split,
                              criterion=criterion, n_jobs=-1)
cv_b_rf_phy = run_cv(clf_b_rf_phy, X_p, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
print(test_score(cv_b_rf_phy))


# In[382]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_rf_phy['test_accuracy']);


# In[383]:


min_max(cv_b_rf_phy['test_accuracy'])


# In[384]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_rf_phy['test_f1_macro']);


# In[385]:


min_max(cv_b_rf_phy['test_f1_macro'])


# In[386]:


## Reporting results
rf_acc, rf_f1 = raw_scores(cv_rf_all)
rf_b_acc, rf_b_f1 = raw_scores(cv_b_rf_all)
df_report_all.loc[experiment_idx, 'rf_f1'] = rf_f1
df_report_all.loc[experiment_idx, 'rf_acc'] = rf_acc
df_report_all.loc[experiment_idx, 'rf_b_f1'] = rf_b_f1
df_report_all.loc[experiment_idx, 'rf_b_acc'] = rf_b_acc

rf_acc, rf_f1 = raw_scores(cv_rf_phy)
rf_b_acc, rf_b_f1 = raw_scores(cv_b_rf_phy)
df_report_phy.loc[experiment_idx, 'rf_f1'] = rf_f1
df_report_phy.loc[experiment_idx, 'rf_acc'] = rf_acc
df_report_phy.loc[experiment_idx, 'rf_b_f1'] = rf_b_f1
df_report_phy.loc[experiment_idx, 'rf_b_acc'] = rf_b_acc


# # AdaBoost

# In[387]:


def create_ab_pipeline(n_estimators=100, min_samples_split=20, criterion='entropy'):
    base_estimator = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split)
    model_ab = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
    return make_pipeline(StandardScaler(), model_ab)


# ### Baseline vs Stress vs Amusement

# #### All modalities

# In[388]:


clf_ab_all = create_ab_pipeline()
cv_ab_all = run_cv(clf_ab_all, X, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_ab_all)


# In[389]:


sns.barplot(x=list(range(len(subjects))), y=cv_ab_all['test_accuracy']);


# In[390]:


min_max(cv_ab_all['test_accuracy'])


# In[391]:


sns.barplot(x=list(range(len(subjects))), y=cv_ab_all['test_f1_macro']);


# In[392]:


min_max(cv_ab_all['test_f1_macro'])


# #### Physiological modalities

# In[393]:


clf_ab_phy = create_ab_pipeline()
cv_ab_phy = run_cv(clf_ab_phy, X_p, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_ab_phy)


# In[394]:


sns.barplot(x=list(range(len(subjects))), y=cv_ab_phy['test_accuracy']);


# In[395]:


min_max(cv_ab_phy['test_accuracy'])


# In[396]:


sns.barplot(x=list(range(len(subjects))), y=cv_ab_phy['test_f1_macro']);


# In[397]:


min_max(cv_ab_phy['test_f1_macro'])


# ### Stress vs Non-stress

# #### All modalities

# In[398]:


clf_b_ab_all = create_ab_pipeline()
cv_b_ab_all = run_cv(clf_b_ab_all, X, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_b_ab_all)


# In[399]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_ab_all['test_accuracy']);


# In[400]:


min_max(cv_b_ab_all['test_accuracy'])


# In[401]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_ab_all['test_f1_macro']);


# In[402]:


min_max(cv_b_ab_all['test_f1_macro'])


# #### Physiological modalities

# In[403]:


clf_b_ab_phy = create_ab_pipeline()
cv_b_ab_phy = run_cv(clf_b_ab_phy, X_p, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_b_ab_phy)


# In[404]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_ab_phy['test_accuracy']);


# In[405]:


min_max(cv_b_ab_phy['test_accuracy'])


# In[406]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_ab_phy['test_f1_macro']);


# In[407]:


min_max(cv_b_ab_phy['test_f1_macro'])


# In[408]:


## Reporting results
ab_acc, ab_f1 = raw_scores(cv_ab_all)
ab_b_acc, ab_b_f1 = raw_scores(cv_b_ab_all)
df_report_all.loc[experiment_idx, 'ab_f1'] = ab_f1
df_report_all.loc[experiment_idx, 'ab_acc'] = ab_acc
df_report_all.loc[experiment_idx, 'ab_b_f1'] = ab_b_f1
df_report_all.loc[experiment_idx, 'ab_b_acc'] = ab_b_acc

ab_acc, ab_f1 = raw_scores(cv_ab_phy)
ab_b_acc, ab_b_f1 = raw_scores(cv_b_ab_phy)
df_report_phy.loc[experiment_idx, 'ab_f1'] = ab_f1
df_report_phy.loc[experiment_idx, 'ab_acc'] = ab_acc
df_report_phy.loc[experiment_idx, 'ab_b_f1'] = ab_b_f1
df_report_phy.loc[experiment_idx, 'ab_b_acc'] = ab_b_acc


# # LDA

# In[409]:


def create_lda_pipeline():
    return make_pipeline(StandardScaler(), LDA())


# ### Baseline vs Stress vs Non-stress

# #### All modalities

# In[410]:


model_lda_all = create_lda_pipeline()
cv_lda_all = run_cv(model_lda_all, X, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_lda_all)


# In[411]:


sns.barplot(x=list(range(len(subjects))), y=cv_lda_all['test_accuracy']);


# In[412]:


min_max(cv_lda_all['test_accuracy'])


# In[413]:


sns.barplot(x=list(range(len(subjects))), y=cv_lda_all['test_f1_macro']);


# In[414]:


min_max(cv_lda_all['test_f1_macro'])


# #### Physiological modalities

# In[415]:


model_lda_phy = create_lda_pipeline()
cv_lda_phy = run_cv(model_lda_phy, X_p, y, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_lda_phy)


# In[416]:


sns.barplot(x=list(range(len(subjects))), y=cv_lda_phy['test_accuracy']);


# In[417]:


min_max(cv_lda_phy['test_accuracy'])


# In[418]:


sns.barplot(x=list(range(len(subjects))), y=cv_lda_phy['test_f1_macro']);


# In[419]:


min_max(cv_lda_phy['test_f1_macro'])


# ### Stress vs Non-Stress

# #### All modalities

# In[420]:


model_b_lda_all = create_lda_pipeline()
cv_b_lda_all = run_cv(model_b_lda_all, X, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_b_lda_all)


# In[421]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_lda_all['test_accuracy']);


# In[422]:


min_max(cv_b_lda_all['test_accuracy'])


# In[423]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_lda_all['test_f1_macro']);


# In[424]:


min_max(cv_b_lda_all['test_f1_macro'])


# #### Physiological modalities

# In[425]:


model_b_lda_phy = create_lda_pipeline()
cv_b_lda_phy = run_cv(model_b_lda_phy, X_p, y_b, groups=df['subject'], cv=LeaveOneGroupOut())
test_score(cv_b_lda_phy)


# In[426]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_lda_phy['test_accuracy']);


# In[427]:


min_max(cv_b_lda_phy['test_accuracy'])


# In[428]:


sns.barplot(x=list(range(len(subjects))), y=cv_b_lda_phy['test_f1_macro']);


# In[429]:


min_max(cv_b_lda_phy['test_f1_macro'])


# In[430]:


## Reporting results
lda_acc, lda_f1 = raw_scores(cv_lda_all)
lda_b_acc, lda_b_f1 = raw_scores(cv_b_lda_all)
df_report_all.loc[experiment_idx, 'lda_f1'] = lda_f1
df_report_all.loc[experiment_idx, 'lda_acc'] = lda_acc
df_report_all.loc[experiment_idx, 'lda_b_f1'] = lda_b_f1
df_report_all.loc[experiment_idx, 'lda_b_acc'] = lda_b_acc

lda_acc, lda_f1 = raw_scores(cv_lda_phy)
lda_b_acc, lda_b_f1 = raw_scores(cv_b_lda_phy)
df_report_phy.loc[experiment_idx, 'lda_f1'] = lda_f1
df_report_phy.loc[experiment_idx, 'lda_acc'] = lda_acc
df_report_phy.loc[experiment_idx, 'lda_b_f1'] = lda_b_f1
df_report_phy.loc[experiment_idx, 'lda_b_acc'] = lda_b_acc


# ## Experiment results

# In[431]:


df_report_all


# In[450]:


eval_all = df_report_all.describe().T[['mean', 'std']].apply(lambda x: x * 100).round(decimals=2)
eval_all


# In[463]:


# eval_all.sort_values(by='mean', ascending=False).to_markdown()
# eval_all.to_markdown()


# In[432]:


df_report_phy


# In[456]:


eval_phy = df_report_phy.describe().T[['mean', 'std']].apply(lambda x: x * 100).round(decimals=2)
eval_phy


# In[464]:


# eval_phy.to_markdown()


# # Feature Importance

# ### Baseline vs Stress vs Amusement

# In[458]:


fi_3 = DecisionTreeClassifier()
fi_3.fit(X, y)
df_3_fi = pd.DataFrame({'cols': X.columns, 'imp': fi_3.feature_importances_}).sort_values('imp', ascending=False)


# In[459]:


df_3_fi[:10]


# In[435]:


# df_3_fi.iloc[:10].to_markdown()


# ### Stress vs Non-stress

# In[460]:


fi_b = DecisionTreeClassifier()
fi_b.fit(X, y_b)
df_b_fi = pd.DataFrame({'cols': X.columns, 'imp': fi_b.feature_importances_}).sort_values('imp', ascending=False)
df_b_fi[:10]


# In[437]:


# df_b_fi.iloc[:10].to_markdown()


# In[ ]:




