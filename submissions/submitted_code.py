
# coding: utf-8

# # Import libraries

# In[220]:


# basics :
import pandas as pd
import numpy as np
from numpy import*
import datetime

# plot
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# impute and scale data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
# evaluation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
# metrics
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
# save models
from sklearn.externals import joblib


# # Read data

# In[221]:


dfm = pd.read_csv('train.csv', sep=",")
dfp = pd.read_csv('test.csv', sep=",")
df = pd.concat([dfm, dfp]) # I do not reset_index 
vardescr = pd.read_excel('variabledescr.xlsx')

vardescr.columns = vardescr.columns.str.lower()

dataframes = [dfm, dfp, df]
for d in dataframes:
    d.columns = d.columns.str.lower()

df['y'] = df['approved']
dfm['y'] = dfm['approved']
df.drop('approved', inplace=True, axis=1)
dfm.drop('approved', inplace=True, axis=1)


# In[222]:


df.shape, dfm.shape, dfp.shape


# In[223]:


set(df.columns) - set(dfp.columns)


# # Inspect and clean

# ```
# input:
# print(vardescr.loc[:, ['column', 'type']])
# 
# output:
# 0                                    id       Object
# 1                                gender      Boolean
# 2                                   dob         Date
# 3                    lead_creation_date         Date
# 4                             city_code  Categorical
# 5                         city_category  Categorical
# 6                         employer_code  Categorical
# 7                    employer_category1  Categorical
# 8                    employer_category2    Categorical
# 9                        monthly_income    Numerical
# 10  customer_existing_primary_bank_code  Categorical
# 11                    primary_bank_type      Boolean
# 12                            contacted      Boolean
# 13                               source  Categorical
# 14                      source_category  Categorical
# 15                         existing_emi    Numerical
# 16                          loan_amount    Numerical
# 17                          loan_period    Numerical
# 18                        interest_rate    Numerical
# 19                                  emi    Numerical
# 20                                 var1  Categorical
# 21                             approved            y
# ```

# In[224]:


# infer dtype
def getcoltype(type):
    '''Input string of type. Output list of colnames with that type. 
    Types can be Numerical, Categorical, Boolean, Object
    '''
    return vardescr.loc[vardescr.type == type, 'column'].tolist()

# save to list based on dtype
cols_datetime = getcoltype('Date')
cols_numeric = getcoltype('Numerical')
cols_category = getcoltype('Categorical')
cols_object = getcoltype('Object')
cols_boolean = getcoltype('Boolean')


# In[225]:


# convert booleans
df['female'] = (df.gender == 'Female')
df['primary_bank_type_G'] = (df.primary_bank_type == 'G')
df['contacted'] = (df.contacted == 'Y')
df.drop(['gender', 'primary_bank_type'], inplace = True, axis=1)
cols_boolean = ['female', 'primary_bank_type_G', 'contacted']


# In[226]:


# convert categorical, numerical, datetime
df[cols_numeric] = df[cols_numeric].apply(lambda x: pd.to_numeric(x))
df[cols_datetime] = df[cols_datetime].apply(lambda x: pd.to_datetime(x))
df[cols_category] = df[cols_category].apply(lambda x: x.astype('category'))


# In[227]:


# create age 
df['age'] = 2018 - df.dob.dt.year
cols_numeric.append('age')


# In[228]:


# based on descripe, fix age
df.loc[df.age < 0, 'age'] = np.nan


# In[229]:


# rows 
rows_train_csv = df['y'].notnull()
rows_test_csv = ~rows_train_csv


# In[230]:


# impute NA  for dfm and dfp
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df.loc[rows_train_csv, cols_numeric] = imp.fit_transform(df.loc[rows_train_csv, cols_numeric])
df.loc[rows_test_csv, cols_numeric] = imp.fit_transform(df.loc[rows_test_csv, cols_numeric])


# In[231]:


# standardize for dfm and dfp
scaler = StandardScaler()
df.loc[rows_train_csv, cols_numeric] = scaler.fit_transform(df.loc[rows_train_csv, cols_numeric])
df.loc[rows_test_csv, cols_numeric] = scaler.fit_transform(df.loc[rows_test_csv, cols_numeric])


# # Fit and predict 

# In[232]:


scoring = 'roc_auc'


# ## Choose X cols

# List some options:

# In[233]:


# ,'source',  and  'customer_existing_primary_bank_code' # has issues

cols_0 = ['monthly_income', 'contacted',
          'source_category',
          'var1', 'female', 'primary_bank_type_G']

# add a row
cols_1 = ['monthly_income', 'contacted',
          'source_category', 'var1', 'female', 'primary_bank_type_G',
          'age', 'employer_category1', 'employer_category2'
         ]

# like 1 but add financial
cols_1fin = ['monthly_income', 'contacted',
          'source_category', 'var1', 'female', 'primary_bank_type_G',
          'age', 'employer_category1', 'employer_category2'
          , 'existing_emi', 'loan_amount', 'loan_period', 'interest_rate'
         ]

# based on DecisionTree() .feature_importances_ in another .py file:
cols_vip = ['monthly_income', 'existing_emi', 'interest_rate', 'loan_amount', 'age',
       'var1_10', 'employer_category1_C', 'source_category_F',
       'employer_category1_B', 'interest_rate']
# cols_vip is sorted!


# df[cols_numeric].describe().round(2) # qq monthly income and loan amount does not have unit std in the entire set => must be large outliers in test-data.


# In[234]:


# choose cols !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cols_X = cols_1


# ## Convert X categoricals to dummies

# In[235]:


X = pd.get_dummies(df[cols_X], drop_first = True)


# In[236]:


# with one hote encode nr of cols is same or larger 
assert df[cols_X].shape[1] <= X.shape[1]


# In[237]:


# X variables for dfm dfp
Xm = X[rows_train_csv]
Xp = X[rows_test_csv]

# y variables for dfm dfp and df
ym = df.loc[rows_train_csv, 'y']
yp = df.loc[rows_test_csv, 'y']
y = df['y']

# assert 
assert X.shape[0] == y.shape[0]
assert Xm.shape[0] + Xp.shape[0] == X.shape[0] # rows add upp 
assert Xm.shape[1] == Xp.shape[1] == X.shape[1] # same nr of cols


# ## Fit on X_train predict on X_test get roc auc score

# In[238]:


# split 
X_train, X_test, y_train, y_test = train_test_split(Xm, ym, test_size=0.20, random_state=9)


# In[239]:


# fit 
clf = GradientBoostingClassifier() # good roc auc


# In[240]:


clf.fit(X_train, y_train)


# In[241]:


# predict
y_pred_proba = clf.predict_proba(X_test)[:, 1]
#y_pred_class = clf.predict(X_test)


# In[242]:


roc_auc_score(y_test, y_pred_proba)
# 0.81778214617737444 with fit on X_train and test on y_test


# ## Fit on Xm and predict on Xp
# 
# Refit on entire Xm rather than only X_train improves estimation

# In[218]:


clf.fit(Xm, ym)
y_pred_proba_sumbit = clf.predict_proba(Xp)[:, 1]


# # Submit 

# In[219]:


# rule: no NA in submission
# rule: range is 0 to 1: min >= 0 max <= 1

test = pd.read_csv('test.csv')
pd.DataFrame({'ID':test['ID'], 'Approved':y_pred_proba_sumbit}).set_index('ID').to_csv('submit.csv')


# # Todo in the future

# install and use xgboost 
# 
# gridsearchcv did not improve roc_auc in Xp or mck solution checker
# 
# votingclassifier (see docs /ensemble.html)
# 
# handle NA differently? 
# 
# use these variables? ,'source',  and  'customer_existing_primary_bank_code'
# 
# explore how to use the date column, I hav not thought about it for now
