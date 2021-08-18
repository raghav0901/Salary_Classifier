# Importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier

# Reading in the dataset using pandas:
df=pd.read_csv("adult.csv")

# Having a look at the data::
df.head()

# EXPLORATORY DATA ANALYSIS::
print(df.columns)

## CHECKING for duplicte rows::
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape[0])

df.drop_duplicates(inplace=True)

## CHECKING for duplicte rows again::
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape[0])

##checking for missing values with different ways:
missing=df.isna().sum()
print(missing)

## Handling the '?' signs to replace them with np.nan values
def checkMiss(s):
    if '?' in str(s):
        return None
    elif type(s)==str:
        return s.strip()
    else:
        return s
df=df.applymap(checkMiss).copy()
print(df['occupation'].value_counts())

##checking for missing values again:
missing_cols=[]
missing=df.isna().sum()
for x in missing.index:
    if missing[x]>0:
        missing_cols.append(x)
print(missing_cols)



## Since all the columns which actually have missing values have very few missing values comapred to the number of rows
## Replacement of the missing values is suitable here.
## Using Random Forest Classifier to predict missing values:
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')

## creating te nomial variables and categorical variales:

categories=[]
for x in df.dtypes.index:
    if df.dtypes[x]=='object':
        categories.append(x)
print(categories)

## categories with ordinal relation:
nominal=['education']
for x in nominal:
  categories.remove(x)
print(categories)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()



## To predict each coulmn's missing values ,were only gonna use the columns with no missing values:
for x in missing_cols:
    del_list=list(missing_cols)
    
    use=list(categories)
    for k in del_list:
        use.remove(k)
        
    del_list.remove(x)
    df2=df.copy()
    df2.drop(del_list,axis=1,inplace=True)
    df2.dropna(subset=[x],inplace=True)
    
    enc_df = pd.get_dummies(df2,columns=use)
    ## handling the nominal variables:
    enc_df['education']=lb.fit_transform(enc_df['education'])
    
    ## fitting the model:
    clf.fit(enc_df.drop(x,axis=1),enc_df[x])
    
    # predicting the missing values from the model:
    df3=df.copy()
    df3.drop(del_list,axis=1,inplace=True)
    df3=pd.get_dummies(df3,columns=use)
    df3['education']=lb.fit_transform(df3['education'])
    df3=df3.loc[df3[x].isna()]
    predictions=clf.predict(df3.drop(x,axis=1))
    target=(df[df[x].isna()][x])
    i=0
    for t in target.index:
        df.at[t,x]=predictions[i]
        i+=1


missing=df.isna().sum()
print(missing)

# MISSING VALUES HAVE NOW BEEN FIXED

# EXPLORATORY DATA ANALYSIS:
## Checking all the different types of data available::
numerical_=[]
categorical_=[]
for x in df.dtypes.index:
    if df.dtypes[x]=='object':
        categorical_.append(x)
    else:
        numerical_.append(x)
print(categorical_)



## Feature selection for categorical features:
bf=df.copy()
for x in categorical_:
    dik={k:i for i,k in enumerate(bf[x].unique())}
    bf[x]=bf[x].map(dik)
print(bf.head())

## Chi square:
from sklearn.feature_selection import chi2
kf=bf[categorical_]
kf=kf.drop('salary',axis=1)
f_p_vals=chi2(kf,bf['salary'])

## checking p-values at 0.05 significance:
remove_cols=[]
ser=pd.Series(f_p_vals[1])
ser.index=kf.columns
for x in ser.index:
    if ser[x]>0.05:
        remove_cols.append(x)
        print('Remove the feature ::',x)
        
## removing the unimportant columns from DF:
df.drop(remove_cols,axis=1,inplace=True)


# FEATURE ENGINNERING:

## converting ordinal categorical data to numerical:
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['education']=lb.fit_transform(df['education'])
print(lb.classes_)
print(lb.transform(lb.classes_))

categorical_.remove('marital-status')
## features with high cardinality:
high_card=['country','occupation','salary']
low_card=list(set(categorical_)-set(high_card))
low_card.remove('education')
## converting non-ordinal categorical(low_cardinaility) data to numerical:
df=pd.get_dummies(df,columns=low_card)

print(df.head())

## converting ordinal categorical(low_cardinaility) data to numerical:
val_count1=df.occupation.value_counts().to_dict()
val_count2=df.country.value_counts().to_dict()
print(val_count1)
print(val_count2)
df.occupation=df.occupation.map(val_count1)
df.country=df.country.map(val_count2)
print(df.head())

## Using xgboost with hyper-parameter tuning::
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
params={
    "learning_rate": [0.05,0.10,0.15,0.2],
    "max_depth":[3,4,5,6],
    "min_child_weight":[1,3,5,7],
    "gamma":[0.0,0.1,0.2,0.3]
}

xtrain,xtest,ytrain,ytest=train_test_split(df.drop('salary',axis=1),df['salary'],test_size=0.33)

model = xgboost.XGBClassifier()
rs=RandomizedSearchCV(model,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
rs.fit(xtrain,ytrain)

rs.best_estimator_

classy=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.15, max_delta_step=0, max_depth=5,
              min_child_weight=5, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None,missing=None)


classy.fit(xtrain,ytrain)

pred=classy.predict(xtest)

## Printing accuracy score::
print(accuracy_score(ytest,pred))
print(df.columns)
# Saving model to disk
pickle.dump(classy, open('model.pkl','wb'))