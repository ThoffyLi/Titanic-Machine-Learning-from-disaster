# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:07:49 2019

@author: Thoffy
"""

'''
Stacking
 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
 

# PART 2 - preprocessing  
# concatenate train and test and preprocess them together- to have uniform format
train_set = pd.read_csv('titanic_train.csv')
train_set_y =train_set.iloc[:,1] 
train_set = train_set.drop('Survived',axis=1)
test_set = pd.read_csv('titanic_test.csv')
dataset = train_set.append(test_set)

#reset index
dataset.reset_index(inplace = True)
dataset.drop('index',axis=1,inplace = True)

dataset.drop('PassengerId',axis = 1, inplace = True)
 

#Embarked missing value : looking into observations with missing Embarked port
emb_missing = dataset[dataset['Embarked'].isnull()]
# these two observations share variables: Pclass and Ticket, so use these variables to speculate Embarked port:
# Tickets starts with 113:
emb_missing_pattern = '^113.+'
emb_missing_ticket = dataset['Ticket'].apply(lambda x:bool(re.search(emb_missing_pattern,x)))
ticket_113 = dataset[emb_missing_ticket]  #all observations with tickets starting with 113
ticket_113['Ticket'] = ticket_113['Ticket'].apply(int)
ticket_113.sort_values('Ticket',inplace = True)
ticket_113 = ticket_113[['Ticket','Embarked']] 
ticket_113['Embarked'].value_counts()
# 52 S vs 10 C, and around 1135XX, most of them are S, so here fill missing values with S
dataset['Embarked'].fillna('C',inplace = True)
    
# name : extract the title from name using regex
pattern = re.compile('\w+, (.+?)[.]')
dataset['Title'] = 'None'
for i in range(0,len(dataset['Name'])):
    try:    
        dataset['Title'][i] = re.search(pattern,list(dataset['Name'].values)[i]).group(1) 
    except:
        continue
    
dataset.drop('Name',axis=1,inplace = True)


# FamilySize = Sibsp + Parch + 1, further classified as 0(familysize>7),1(familysize=1 or 5,6,7) ,2(2<=familysize<=4)
dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1
dataset['FamilySize'][dataset['FamilySize']>7]=0
dataset['FamilySize'][(dataset['FamilySize']==1)|((dataset['FamilySize']>=5)&(dataset['FamilySize']<=7))]=1
dataset['FamilySize'][(dataset['FamilySize']>=2)&(dataset['FamilySize']<=4)]=2
dataset.drop(['SibSp','Parch'],axis=1,inplace = True)


# Fare has 1 missing value - look into the dataset:
missing_fare_ob = dataset[dataset['Fare'].isnull()] 
#this person's ticket is 3701, then check the similar ticket number's fare pattern:
ticket_37_pattern = r'^3\d\d\d$'
ticket_37 = dataset[dataset['Ticket'].apply(lambda x:bool(re.search(ticket_37_pattern,x)))]
# all the fare are similar, it's reasonable to use mean to fill
mean_fare = np.mean(ticket_37[ticket_37['Fare'].notnull()]['Fare'])
dataset['Fare'].fillna(mean_fare,inplace = True)

# Cabin
#two groups: 0(Unknown),1(known)
dataset['Cabin'][dataset['Cabin'].isnull()] ='U'
dataset['Cabin']=dataset['Cabin'].str.get(0)
dataset['Cabin'] = dataset['Cabin'].apply(lambda x: 0 if x=='U' else 2 if x in ('E','D','B') else 1)



# Title: Unify to English format and implement dummy encoding
title_dict = dict()
title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
dataset['Title'] = dataset['Title'].map(title_dict)

dataset['Title'][dataset['Title'].isin(('Mr','Officer'))] = 0
dataset['Title'][dataset['Title'] =='Master'] = 1
dataset['Title'][dataset['Title'].isin(('Mrs','Miss','Royalty'))]=2
dataset['Title'] = dataset['Title'].apply(int)


 
#Ticket: same pattern as family size, classify them into groups
ticket_counts = dataset['Ticket'].value_counts()
ticket_unique =list(dataset['Ticket'].unique())
ticket_dict = dict()
for k in range(0,len(ticket_unique)):
    ticket_dict.update(dict.fromkeys([ticket_unique[k]],ticket_counts.loc[ticket_unique[k]]))
dataset['Ticket'] = dataset['Ticket'].map(ticket_dict)
dataset['Ticket_nums'] = dataset['Ticket']
# divide Tickets into three classes 
dataset['Ticket_nums'][(dataset['Ticket_nums']==5)|(dataset['Ticket_nums']==6)|(dataset['Ticket_nums']>=8)]=0 # 5,6,more than 8 -- group 0
dataset['Ticket_nums'][(dataset['Ticket_nums']==1)|(dataset['Ticket_nums']==7)]=1 #1,7 --group 1
dataset['Ticket_nums'][(dataset['Ticket_nums']>=2)&(dataset['Ticket_nums']<=5)]=2 #2,3,4 --group 2
dataset.drop('Ticket',axis=1,inplace = True)

# Age: choose relevant variables to predict and fill missing values
'''
# pclass:
age_pclass = dataset[['Age','Pclass']][dataset['Age'].notnull()]
age_pclass['Pclass'] = age_pclass['Pclass'].apply(str)
a_p_facet = sns.FacetGrid(data =age_pclass,hue = 'Pclass',size=7 )
a_p_facet.map(sns.kdeplot,'Age',shade = True)
a_p_facet.add_legend()
plt.show()
# different distribution, so pclass is relevant

#Title:
age_title = dataset[['Age','Title']][dataset['Age'].notnull()]
a_t_facet = sns.FacetGrid(data =age_title,hue = 'Title',size=7 )
a_t_facet.map(sns.kdeplot,'Age',shade = True)
a_t_facet.add_legend()
plt.show()
# different distribution, so title is relevant

#Sex:
age_sex = dataset[['Age','Sex']][dataset['Age'].notnull()]
a_s_facet = sns.FacetGrid(data =age_sex,hue = 'Sex',size=7 )
a_s_facet.map(sns.kdeplot,'Age',shade = True)
a_s_facet.add_legend()
plt.show()
# different distribution, so sex is relevant
'''
#use Sex,Title,Pclass to predict Age

age_set = dataset[['Age','Pclass','Title','Sex']]

# preprocessing: Pclass Sex and Title -> dummy variables
age_set = pd.get_dummies(age_set)

age_train_x = age_set[age_set['Age'].notnull()].iloc[:,1:]
age_train_y = age_set[age_set['Age'].notnull()].iloc[:,0]
age_test_x = age_set[age_set['Age'].isnull()].iloc[:,1:]

 #Random Forest
 #Grid Search to find optimal parameters
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
'''
regressor_age_rf = RandomForestRegressor()
age_rf_params = [{'n_estimators':[100,500,1000,2000],'max_depth':[4,5,6,7],'min_samples_split':[6]}]
age_rf_grid_search = GridSearchCV(estimator=regressor_age_rf,param_grid = age_rf_params,scoring = 'neg_mean_squared_error',cv=10,n_jobs=-1)
age_rf_gs = age_rf_grid_search.fit(age_train_x,age_train_y)
age_rf_gs.best_params_
age_rf_gs.best_score_
# best params:{'max_depth': 6, 'min_samples_split': 7, 'n_estimators': 100}
# best score:-151.83
'''
regressor_age_rf = RandomForestRegressor(n_estimators = 100,max_depth = 6, min_samples_split = 5, random_state=0)
regressor_age_rf.fit(age_train_x,age_train_y)
'''
features = list(age_train_x.columns)
importance = list(regressor_age_rf.feature_importances_)
fea_imp = {'feature':features,'importance':importance}
imp = pd.DataFrame(data =fea_imp )
imp.sort_values('importance',inplace = True,ascending = False)
sns.barplot(x = 'feature',y='importance',data = imp)
plt.show()
'''
age_pred_y = regressor_age_rf.predict(age_test_x)
dataset['Age_RF']=dataset['Age']
dataset['Age_RF'][dataset['Age'].isnull()] = age_pred_y
 

#Gradient Boosting
#Grid Search to find optimal parameters

from sklearn.ensemble import GradientBoostingRegressor
'''
regressor_gb = GradientBoostingRegressor()
gb_params = [{'n_estimators':[100,1000,2000],'max_depth':[3],'min_samples_split':[2,3,4]}]
gb_grid_search = GridSearchCV(estimator=regressor_gb,param_grid = gb_params,scoring = 'neg_mean_squared_error',cv=10,n_jobs=-1)
gb_gs = gb_grid_search.fit(age_train_x,age_train_y)
gb_gs.best_params_
gb_gs.best_score_'''
# best params: {'max_depth': 3, 'min_samples_split': 3, 'n_estimators': 100}
# best score: -141.40474456016958
regressor_gb = GradientBoostingRegressor(n_estimators=100,max_depth=3, min_samples_split=3,random_state=0)
regressor_gb.fit(age_train_x,age_train_y)
'''
#Feature Selection
regressor_gb.feature_importances_
features = list(age_train_x.columns)
importance = list(regressor_gb.feature_importances_)
fea_imp = {'feature':features,'importance':importance}
imp = pd.DataFrame(data =fea_imp )
imp.sort_values('importance',inplace = True,ascending = False)
sns.barplot(x = 'feature',y='importance',data = imp)
plt.show()
'''
age_pred_y_gb = regressor_gb.predict(age_test_x)
dataset['Age_GB']=dataset['Age']
dataset['Age_GB'][dataset['Age'].isnull()] = age_pred_y_gb

# Merging two models - mean
dataset['Age'] =(dataset['Age_GB']+dataset['Age_RF'])/2
dataset.drop(['Age_GB','Age_RF'],axis=1,inplace = True)

# divide people into child and adult(>15) groups
dataset['Child']='None'
dataset['Child'][dataset['Age']<=15]='Child'
dataset['Child'][dataset['Age']>15]='Adult'
dataset.drop('Age',axis=1,inplace = True)


# dummy encoding
dataset = pd.get_dummies(dataset)


# test_train split
train_set_x = dataset.iloc[:891,:]
test_set_x = dataset.iloc[891:,:]
#train_set_y has been defined at the beginning

X_train = train_set_x.values
y_train = train_set_y.values
X_test = test_set_x.values


# Classification: 
# use K-best feature and grid search to find optimal parameters
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

#1. Random Forest
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('rf', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'kb__k':[9],
              'kb__score_func':[chi2],
              'rf__n_estimators':[554], 
              'rf__max_depth':[7],
              'rf__min_samples_split':[2],
              'rf__min_samples_leaf':[2],
              'rf__criterion':['entropy']
              }


gs_rf = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10,n_jobs=-1)
gs_rf.fit(X_train,y_train)
print(gs_rf.best_params_, gs_rf.best_score_)
'''
#best params:{'kb__k': 9, 'kb__score_func': chi2, 'rf__criterion': 'entropy',
  #'rf__max_depth': 7, 'rf__min_samples_leaf': 2, 'rf__min_samples_split': 2, 'rf__n_estimators': 554}
#best score: roc_auc = 0.8945287854510803

# Learning curve on selected model
'''
from sklearn.model_selection import learning_curve
kb_rf = SelectKBest(chi2,k = 9)
clf_rf = RandomForestClassifier(random_state = 10, 
                                  n_estimators = 554,
                                  max_depth = 7, 
                                  criterion= 'entropy',
                                  min_samples_leaf = 2,
                                  min_samples_split = 2,
                                  max_features = 'sqrt')
pipeline_rf = make_pipeline(kb_rf, clf_rf)
train_sizes,train_scores,test_scores = learning_curve(pipeline_rf,X_train,y_train,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.65,1])
plt.legend()
plt.title('Random Forest Learning Curve')
plt.show()
'''
# RF predict 
kb_rf = SelectKBest(chi2,k = 9)
clf_rf = RandomForestClassifier(random_state = 10, 
                                  n_estimators = 554,
                                  max_depth = 7, 
                                  criterion= 'entropy',
                                  min_samples_leaf = 2,
                                  min_samples_split = 2,
                                  max_features = 'sqrt')
rf_pipeline = make_pipeline(kb_rf, clf_rf)




# 2. Extra Tree
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('et', ExtraTreesClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'kb__k':[9],
              'kb__score_func':[chi2],
              'et__n_estimators':[941], 
              'et__max_depth':[7],
              'et__min_samples_split':[2],
              'et__min_samples_leaf':[2],
              'et__criterion':['entropy']
              }


gs_et = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10,n_jobs=-1)
gs_et.fit(X_train,y_train)
print(gs_et.best_params_, gs_et.best_score_)
#best params: {'et__criterion': 'entropy', 'et__max_depth': 7, 'et__min_samples_leaf': 2,
   #'et__min_samples_split': 2, 'et__n_estimators': 941, 'kb__k': 9, 'kb__score_func': <function chi2 at 0x000001A6C9135400>}
#best score: accuracy = 0.8383838383838383
'''
# Learning curve on selected model
'''
from sklearn.model_selection import learning_curve
kb_et = SelectKBest(chi2,k = 10)
clf_et = ExtraTreesClassifier(random_state = 10, 
                                  n_estimators = 941,
                                  max_depth = 7, 
                                  criterion= 'entropy',
                                  min_samples_leaf = 2,
                                  min_samples_split = 2,
                                  max_features = 'sqrt')
               
pipeline_et = make_pipeline(kb_et, clf_et)
train_sizes,train_scores,test_scores = learning_curve(pipeline_et,X_train,y_train,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.84,1])
plt.legend()
plt.title('Extra Tree Learning Curve')
plt.show()
'''
# ET predict 
kb_et = SelectKBest(chi2,k = 10)
clf_et = ExtraTreesClassifier(random_state = 10, 
                                  n_estimators = 941,
                                  max_depth = 7, 
                                  criterion= 'entropy',
                                  min_samples_leaf = 2,
                                  min_samples_split = 2,
                                  max_features = 'sqrt')
et_pipeline = make_pipeline(kb_et, clf_et)




# Feature scaling:Fare,Age,FamilySize
dataset_scaled = dataset.copy()
s_fare = StandardScaler()
fare_scaled = s_fare.fit_transform(dataset_scaled['Fare'].values.reshape(-1,1))
dataset_scaled['Fare'] = fare_scaled

s_pc = StandardScaler()
pc_scaled = s_pc.fit_transform(dataset_scaled['Pclass'].values.reshape(-1,1))
dataset_scaled['Pclass'] = pc_scaled

s_cb = StandardScaler()
cb_scaled = s_cb.fit_transform(dataset_scaled['Cabin'].values.reshape(-1,1))
dataset_scaled['Cabin'] = cb_scaled

s_tt = StandardScaler()
tt_scaled = s_tt.fit_transform(dataset_scaled['Title'].values.reshape(-1,1))
dataset_scaled['Title'] = tt_scaled

s_fs = StandardScaler()
fs_scaled = s_fs.fit_transform(dataset_scaled['FamilySize'].values.reshape(-1,1))
dataset_scaled['FamilySize'] = fs_scaled


s_tn = StandardScaler()
tn_scaled = s_tn.fit_transform(dataset_scaled['Ticket_nums'].values.reshape(-1,1))
dataset_scaled['Ticket_nums'] = tn_scaled


train_set_x_s = dataset_scaled.iloc[:891,:]
test_set_x_s = dataset_scaled.iloc[891:,:]
 
# train/test set after scaling
X_train_s = train_set_x_s.values
y_train_s = train_set_y.values
X_test_s = test_set_x_s.values


#  SVM
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('svm', SVC())])

param_test = {'kb__k':['all'],
              'svm__kernel':['rbf'], 
              'svm__C':[1.41],
              'svm__gamma':[0.1]
              }
gs_svm = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10,n_jobs=-1)
gs_svm.fit(X_train_s,y_train_s)
print(gs_svm.best_params_, gs_svm.best_score_)

#best params: {'kb__k': 'all', 'svm__C': 1.41, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}
#best score: accuracy = 0.8316498316498316
'''
# Learning curve on selected model
'''
from sklearn.model_selection import learning_curve
kb_svm = SelectKBest(f_classif,k=12)
clf_svm = SVC(C = 1.41, gamma = 0.1, kernel = 'rbf')
 
pipeline_svm = make_pipeline(kb_svm, clf_svm)
train_sizes,train_scores,test_scores = learning_curve(pipeline_svm,X_train_s,y_train_s,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.83,1])
plt.legend()
plt.title('SVM Learning Curve')
plt.show()
'''
# SVM predict 
kb_svm = SelectKBest(f_classif,k=12)
clf_svm = SVC(C = 1.41, gamma = 0.1, kernel = 'rbf')
svm_pipeline = make_pipeline(kb_svm, clf_svm)



#Stacking: 
 #LEVEL 1:  bagging RF+ET+SVM
  # k-fold on un-scaled training set
def get_out_of_kfold(model,fold,train_x,train_y,test_x):
   
    kfold_uns = KFold(n_splits = fold,shuffle = False)
    train_pred_results =np.array([])
    test_pred_results =np.zeros((X_test.shape[0],))
    for kf_data in kfold_uns.split(train_x):
        train_index = kf_data[0]
        test_index = kf_data[1]
        # training set(folds-1),test set(1)
        kf_train_x = train_x[train_index]
        kf_train_y = train_y[train_index]
        kf_test = train_x[test_index]
       
        model.fit(kf_train_x,kf_train_y)
        train_pred_results = np.hstack((train_pred_results,model.predict(kf_test)))
        test_pred_results =test_pred_results + model.predict(test_x) 
    test_pred_results = np.round(test_pred_results/fold) 
    result_list = list()
    result_list.append(train_pred_results)
    result_list.append(test_pred_results)
    return result_list

NFold = 9  # 9 fold
rf_list = get_out_of_kfold(rf_pipeline,NFold,X_train,y_train,X_test)
et_list = get_out_of_kfold(et_pipeline,NFold,X_train,y_train,X_test)
#for svm, use scaled dataset
svm_list = get_out_of_kfold(svm_pipeline,NFold,X_train_s,y_train_s,X_test_s)

# Create train/test set for level2
bagging_dict_train = {'RF':rf_list[0],'ET':et_list[0],'SVM':svm_list[0]}
X_train_lv1 =pd.DataFrame(bagging_dict_train)
y_train_lv1 = y_train

bagging_dict_test = {'RF':rf_list[1],'ET':et_list[1],'SVM':svm_list[1]}
X_test_lv2 = pd.DataFrame(bagging_dict_test)



# lEVEL2: Meta classifier:
# use K-best feature and grid search to find optimal parameters

# meta - Logistic Regression
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('lr', LogisticRegression())])

param_test = {'kb__k':['all',2,1],
              'lr__penalty':['l1','l2'], 
              'lr__C':[0.01,0.1,1,10,100,1000,10000],
              'lr__solver':['saga','liblinear']
              }
gs_lr = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10,n_jobs=-1)
gs_lr.fit(X_train_lv1,y_train_lv1)
print(gs_lr.best_params_, gs_lr.best_score_)
#best params: shown above
#best score: roc_auc = 0.863100132410887

# Learning curve on selected model

from sklearn.model_selection import learning_curve
kb_lr = SelectKBest(f_classif,k=2)
clf_lr = LogisticRegression(C = 0.01, penalty='l2', solver='saga',random_state=10)
 
pipeline_lr = make_pipeline(kb_lr, clf_lr)
train_sizes,train_scores,test_scores = learning_curve(pipeline_lr,X_train_lv1,y_train_lv1,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0,1])
plt.legend()
plt.title('Logistic Regression Learning Curve')
plt.show()

# LR predict 
kb_lr = SelectKBest(f_classif,k='all')
clf_lr = LogisticRegression(C = 10, penalty='l2', solver='saga',random_state=10)
pipeline = make_pipeline(kb_lr, clf_lr)
pipeline.fit(X_train_s, y_train_s)
lr_pred = pipeline.predict(test_set_x_s.values)
'''


# meta - Random Forest
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('rf', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'kb__k':['all'],
              'kb__score_func':[chi2],
              'rf__n_estimators':[994], 
              'rf__max_depth':[7],
              'rf__min_samples_split':[2],
              'rf__min_samples_leaf':[2],
              'rf__criterion':['entropy']
              }

gs_rf = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10,n_jobs=-1)
gs_rf.fit(X_train_lv1,y_train_lv1)
print(gs_rf.best_params_, gs_rf.best_score_)

#best params: shown above
#best score: accuracy = 0.8372615039281706

# Learning curve on selected model

from sklearn.model_selection import learning_curve
kb_rf = SelectKBest(chi2,k ='all')
clf_rf = RandomForestClassifier(random_state = 10, 
                                  n_estimators = 994,
                                  max_depth = 7, 
                                  criterion= 'entropy',
                                  min_samples_leaf = 2,
                                  min_samples_split = 2,
                                  max_features = 'sqrt')
pipeline_rf = make_pipeline(kb_rf, clf_rf)
train_sizes,train_scores,test_scores = learning_curve(pipeline_rf,X_train_lv1,y_train_lv1,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.7,1])
plt.legend()
plt.title('Random Forest Learning Curve')
plt.show()
'''



# meta - Extra Tree
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('et', ExtraTreesClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'kb__k':['all'],
              'kb__score_func':[chi2],
              'et__n_estimators':[995], 
              'et__max_depth':[6],
              'et__min_samples_split':[2],
              'et__min_samples_leaf':[2],
              'et__criterion':['entropy','gini']
              }


gs_et = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10,n_jobs=-1)
gs_et.fit(X_train_lv1,y_train_lv1)
print(gs_et.best_params_, gs_et.best_score_)
#best params: shown above
#best score: accuracy = 0.8406285072951739

# Learning curve on selected model

from sklearn.model_selection import learning_curve
kb_et = SelectKBest(chi2,k = 'all')
clf_et = ExtraTreesClassifier(random_state = 10, 
                                  n_estimators = 995,
                                  max_depth = 6, 
                                  criterion= 'entropy',
                                  min_samples_leaf = 2,
                                  min_samples_split = 2,
                                  max_features = 'sqrt')
               
pipeline_et = make_pipeline(kb_et, clf_et)
train_sizes,train_scores,test_scores = learning_curve(pipeline_et,X_train_lv1,y_train_lv1,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.8,1])
plt.legend()
plt.title('Extra Tree Learning Curve')
plt.show()
'''


# meta -SVM
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('svm', SVC())])

param_test = {'kb__k':['all'],
              'svm__kernel':['rbf'], 
              'svm__C':list(range(1,10,1)),
              'svm__gamma':list(range(1,11,1))
              }
gs_svm = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10,n_jobs=-1)
gs_svm.fit(X_train_lv1,y_train_lv1)
print(gs_svm.best_params_, gs_svm.best_score_)

#best params: 
#best score: accuracy =  0.8439955106621774
'''
# Learning curve on selected model
'''
from sklearn.model_selection import learning_curve
kb_svm = SelectKBest(f_classif,k='all')
clf_svm = SVC(C = 1, gamma = 1, kernel = 'rbf')
 
pipeline_svm = make_pipeline(kb_svm, clf_svm)
train_sizes,train_scores,test_scores = learning_curve(pipeline_svm,X_train_lv1,y_train_lv1,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.83,1])
plt.legend()
plt.title('SVM Learning Curve')
plt.show()
'''


# meta - XGBoost
'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('xgb', XGBClassifier())])

param_test = {'kb__k':['all'],
              'kb__score_func':[chi2],
              'xgb__n_estimators':[200,500,1000], 
              'xgb__max_depth':[7],
              'xgb__min_child_weight':[4,5,6,7],
              'xgb__gamma':[0.0001,0.001,0.01,0.05,0.1,0.5,1],
              'xgb__subsample':[0.65],
              'xgb__colsample_bytree':[0.7]
              }
gs_xgb = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10,n_jobs=-1)
gs_xgb.fit(X_train_lv1,y_train_lv1)
print(gs_xgb.best_params_, gs_xgb.best_score_)

#best params: {'kb__k': 11, 'kb__score_func': chi2 , 'xgb__colsample_bytree': 0.7, 'xgb__gamma': 0.001, 
  #'xgb__max_depth': 7, 'xgb__min_child_weight': 7, 'xgb__n_estimators': 991, 'xgb__subsample': 0.65}
#best score: roc_auc = 

# Learning curve on selected model

from sklearn.model_selection import learning_curve
kb_xgb = SelectKBest(chi2,k='all')
clf_xgb = XGBClassifier(n_estimators = 900, colsample_bytree = 0.7, gamma = 0.001, 
                        max_depth = 4, min_child_weight = 7,subsample =0.65) 
 
pipeline_xgb = make_pipeline(kb_xgb, clf_xgb)
train_sizes,train_scores,test_scores = learning_curve(pipeline_xgb,X_train,y_train,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.65,1])
plt.legend()
plt.title('XGBoost Learning Curve')
plt.show()
'''

# meta -KNN
# K-Nearest Neighbors

'''
pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 
               ('knn', KNeighborsClassifier())])

param_test = {'kb__k':['all'],
              'knn__n_neighbors':list(range(5,30)), 
              'knn__weights':['distance'],
              'knn__algorithm':['auto']
              }
gs_knn = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10,n_jobs=-1)
gs_knn.fit(X_train_lv1,y_train_lv1)
print(gs_knn.best_params_, gs_knn.best_score_)

#best params: {'kb__k': 'all', 'knn__algorithm': 'auto', 'knn__n_neighbors': 28, 'knn__weights': 'distance'}
#best score: acc =  0.8422326329436648

'''
# Learning curve on selected model
'''
from sklearn.model_selection import learning_curve
kb_knn = SelectKBest(f_classif,k='all' )
clf_knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors=28, weights='uniform')
 
pipeline_knn = make_pipeline(kb_knn, clf_knn)
train_sizes,train_scores,test_scores = learning_curve(pipeline_knn,X_train_lv1,y_train_lv1,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')
plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')
plt.ylim([0.7,1])
plt.legend()
plt.title('K-NN Learning Curve')
plt.show()
'''

# Use Extra Tree, SVM and K-NN as meta-classifier one by one

# K-NN
kb_knn = SelectKBest(f_classif,k='all' )
clf_knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors=28, weights='uniform')
pipeline_knn = make_pipeline(kb_knn, clf_knn)
pipeline_knn.fit(X_train_lv1,y_train_lv1)
knn_pred = pipeline_knn.predict(X_test_lv2)


#SVM
kb_svm = SelectKBest(f_classif,k='all')
clf_svm = SVC(C = 1, gamma = 1, kernel = 'rbf')
pipeline_svm = make_pipeline(kb_svm, clf_svm)
pipeline_svm.fit(X_train_lv1,y_train_lv1)
svm_pred = pipeline_svm.predict(X_test_lv2)


#EXT
kb_et = SelectKBest(chi2,k = 'all')
clf_et = ExtraTreesClassifier(random_state = 10, 
                                  n_estimators = 995,
                                  max_depth = 6, 
                                  criterion= 'entropy',
                                  min_samples_leaf = 2,
                                  min_samples_split = 2,
                                  max_features = 'sqrt')           
pipeline_et = make_pipeline(kb_et, clf_et)
pipeline_et.fit(X_train_lv1,y_train_lv1)
et_pred = pipeline_et.predict(X_test_lv2)


test_set_x.reset_index(inplace = True)
test_set_x.drop('index',axis = 1, inplace = True)

 
test_set_x['ET_pred']=et_pred
test_set_x['SVM_pred']=svm_pred
test_set_x['KNN_pred']=knn_pred
 

# final result output
final_result = pd.DataFrame()
final_result['PassengerId'] = test_set['PassengerId']
final_result['Survived'] = test_set_x['KNN_pred']
final_result.to_csv('Thoffy-submission-v7 knn.csv')
