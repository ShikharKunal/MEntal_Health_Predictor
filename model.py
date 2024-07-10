import numpy as np
import pandas as pd
import warnings
import pickle
warnings.simplefilter('ignore')

df = pd.read_csv('dataset/survey.csv')

df.columns = df.columns.str.lower()

df = df.drop(['country','state','timestamp','comments'], axis = 1)

df['gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

df['gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'Female', inplace = True)

df["gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'Other', inplace = True)


df.loc[df.age<12,'age']=15
df.loc[df.age>75,'age']=75


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df,test_size=0.15,stratify=df['treatment'],random_state=42 )

health = train_data.copy()

#Self employed column contains as low as 2% null values , so it is okay to replace it with mode.
se_mode = train_data['self_employed'].mode().values[0] 
train_data['self_employed'].fillna(se_mode,inplace=True)
# Work_interfere contains almost 20% null values which is significant as we have less data.Let us once see the null values to find any pattern
train_data[train_data['work_interfere'].isna()]['treatment'].value_counts()


train_data['work_interfere'].fillna('Never',inplace = True)

X_train = train_data.drop('treatment',axis=1)
y_train = train_data['treatment'].copy()

gender_cols = ['Female','Male','Other']
self_employed_cols = ['No','Yes']
family_history_cols = ['No','Yes']
work_interfere_cols = ['Never','Rarely','Sometimes','Often']
no_employees_cols = ['1-5','6-25','26-100','100-500','500-1000','More than 1000']
remote_work_cols = ['No','Yes']
tech_company_cols = ['No','Yes']
benefits_cols = ['No','Don\'t know','Yes'] 
care_options_cols = ['No','Not sure','Yes']
wellness_program_cols  =['No','Don\'t know','Yes']
seek_help_cols = ['No','Don\'t know','Yes']
anonymity_cols = ['No','Don\'t know','Yes']
leave_cols = [ 'Very easy', 'Somewhat easy',"Don't know" ,'Somewhat difficult','Very difficult']
mental_health_consequence_cols = ['No','Maybe','Yes']
phys_health_consequence_cols = ['No','Maybe','Yes']
coworkers_col = ['No','Some of them','Yes']
supervisor_cols = ['No','Some of them','Yes']
mental_health_interview_cols = ['No','Maybe','Yes']
phys_health_interview_cols = ['No','Maybe','Yes']
mental_vs_physical_cols = ["Don't know",'No','Yes']
obs_consequence_cols = ['No','Yes']

columns_for_encoder = [gender_cols,self_employed_cols,family_history_cols,work_interfere_cols,no_employees_cols,remote_work_cols,
                            tech_company_cols,benefits_cols,care_options_cols,wellness_program_cols,seek_help_cols,anonymity_cols,leave_cols,
                            mental_health_consequence_cols,phys_health_consequence_cols,coworkers_col,supervisor_cols,mental_health_interview_cols,
                            phys_health_interview_cols,mental_vs_physical_cols,obs_consequence_cols]

features = list(X_train.columns)

from sklearn.preprocessing import OrdinalEncoder
ord_encoder = OrdinalEncoder(categories=list(columns_for_encoder))
X_train[features[1:]] = ord_encoder.fit_transform(X_train.iloc[:,1:])
#save the encoder
with open('ord_encoder.pkl', 'wb') as f:
    pickle.dump(ord_encoder, f)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train[features] = std_scaler.fit_transform(X_train)

#save the scaler
with open('std_scaler.pkl', 'wb') as f:
    pickle.dump(std_scaler, f)


from sklearn.preprocessing import LabelEncoder
lb_encoder = LabelEncoder()
y_train = lb_encoder.fit_transform(y_train)

#save the label encoder
with open('lb_encoder.pkl', 'wb') as f:
    pickle.dump(lb_encoder, f)


from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
def train_evaluate(model,X_train,y_train,name):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    f1_train = f1_score(y_train,y_pred)

    #Cross validation
    f1_val = cross_val_score(model,X_train,y_train,scoring='f1',cv=10)
    
    # returning the scores
    score = pd.DataFrame({'Name' : name ,'F1_score_trainset' : [f1_train], 'F1_score_validationset' : [f1_val.mean()]})
    return score

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='l1',solver='liblinear')
train_evaluate(log_reg,X_train,y_train,'Logistic Regression')

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(max_leaf_nodes=4,random_state=42)
train_evaluate(dt_clf,X_train,y_train,'DecisionTreeClassifier')

from sklearn.svm import SVC
svc_clf = SVC()
train_evaluate(svc_clf,X_train,y_train,'Support Vector Classifier')

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(random_state=42)
train_evaluate(rnd_clf,X_train,y_train,'RandomForestClassifier')


from sklearn.ensemble import GradientBoostingClassifier
gdb_clf = GradientBoostingClassifier(random_state=42,subsample=0.8)

train_evaluate(gdb_clf,X_train,y_train,"GradientBoosting CLASSIFIER")

from xgboost import XGBClassifier
xgb_clf = XGBClassifier(verbosity=0)
train_evaluate(xgb_clf,X_train,y_train,"XG Boost CLASSIFIER")

from sklearn.model_selection import GridSearchCV
param_distribs = {
        'kernel': ['linear', 'rbf','polynomial'],
        'C': [0.01,0.01,0.1,0.15,0.2,0.25,0.5,0.75,1,2,10,100],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    }
svm_clf = SVC()
grid_cv = GridSearchCV(svm_clf , param_grid = param_distribs,
                              cv=5,scoring='f1',
                              verbose=1)
grid_cv.fit(X_train,y_train)

train_evaluate(grid_cv.best_estimator_,X_train,y_train,"SVC Tuned")

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30,50,100],'max_features':[2,4,6,8],'max_depth' : [1,2,3,4]}
]



forest_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='f1',
                           return_train_score=True)
grid_search.fit(X_train, y_train)

train_evaluate(grid_search.best_estimator_,X_train,y_train,"RandomForest Tuned")

param_grid = [
    {'n_estimators':[3,10,30,50,100],
    'max_features':[2,4,6,8,10],
    'max_depth' : [1,2,3,4],
    'subsample': [0.25,0.5,0.75]}
]

gdb_clf2 = GradientBoostingClassifier(random_state=42)
grid_search2 = GridSearchCV(gdb_clf2, param_grid, cv=5,
                           scoring='f1',
                           return_train_score=True)
grid_search2.fit(X_train, y_train)


train_evaluate(grid_search2.best_estimator_,X_train,y_train,"GradientBoosting Tuned")

param_grid = [
    {'n_estimators':[3,10,30,50,100],
    'eta' : [0.01,0.025, 0.05, 0.1],
    'max_features':[2,4,6,8],
    'max_depth' : [1,2,3,4],
    'subsample': [0.5,0.75],
    'booster':['gblinear','gbtree']}
]

xgb_clf = XGBClassifier(verbosity = 0)
grid_search3 = GridSearchCV(xgb_clf, param_grid, cv=5,
                           scoring='f1',
                           return_train_score=True)
grid_search3.fit(X_train, y_train)

train_evaluate(grid_search3.best_estimator_,X_train,y_train,"XGBoost Finetuned")

XGBoost_final = grid_search3.best_estimator_

# Save the model
with open('XGBoost_final.pkl', 'wb') as f:
    pickle.dump(XGBoost_final, f)

X_test = test_data.drop('treatment',axis=1)
y_test = test_data['treatment'].copy()

X_test['self_employed'].fillna(se_mode,inplace=True)
X_test['work_interfere'].fillna('Never',inplace = True)

from sklearn.preprocessing import OrdinalEncoder
# We should only transform using the learned encoder from the training set
X_test[features[1:]] = ord_encoder.transform(X_test.iloc[:,1:])

X_test[features] = std_scaler.transform(X_test)

# Encoding the target column
y_test = lb_encoder.transform(y_test)

# Evaluating the model on test set with our finalized model
y_test_pred = XGBoost_final.predict(X_test)
print(f'F1_score on Test Set : {f1_score(y_test,y_test_pred)}')