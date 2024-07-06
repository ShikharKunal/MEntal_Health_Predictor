import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
import pickle

path = 'dataset/survey.csv'
data_file = pd.read_csv(path)

dropped_columns = ['comments', 'Timestamp', 'state', 'self_employed', 'work_interfere', 'Country']
data_file.drop(dropped_columns, axis=1, inplace=True)
data_file.dropna(inplace=True)

# Process gender column
def process_gender(gender):
    gender = gender.lower()
    if 'female' not in gender and ('male' in gender or gender == 'm'):
        return 'male'
    elif 'female' in gender or gender == 'f':
        return 'female'
    else:
        return 'others'

data_file['Gender'] = data_file['Gender'].apply(process_gender)

# Filter age column
data_file = data_file[(data_file['Age'] > 16) & (data_file['Age'] < 100)]

# Define features and target
y = data_file['treatment'].replace(['Yes', 'No'], [1, 0])
X = data_file.drop(columns=['treatment'])

# Define numeric and categorical features
numeric_features = ['Age']
categorical_features = ['Gender', 'family_history', 'no_employees',
                        'remote_work', 'tech_company', 'benefits', 'care_options',
                        'wellness_program', 'seek_help', 'anonymity', 'leave',
                        'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                        'supervisor', 'mental_health_interview', 'phys_health_interview',
                        'mental_vs_physical', 'obs_consequence']

# Feature engineering
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)  # drop='first' to handle multicollinearity
    ])

# Fit the preprocessor
preprocessor.fit(X)

# Transform the features
X_preprocessed = preprocessor.transform(X)

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Define models and parameter grids
models = {
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'MLPClassifier': MLPClassifier(),
    'SVC': SVC()
}

param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'AdaBoost': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5]
    },
    'MLPClassifier': {
        'classifier__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'classifier__activation': ['tanh', 'relu'],
        'classifier__solver': ['sgd', 'adam'],
        'classifier__alpha': [0.0001, 0.05],
        'classifier__learning_rate': ['constant','adaptive']
    },
    'SVC': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': [1, 0.1, 0.01, 0.001],
        'classifier__kernel': ['rbf', 'poly', 'sigmoid']
    }
}

best_models = {}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('classifier', model)
    ])
    
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    
    # Save the model with the highest accuracy
    if model_name == max(best_models, key=lambda k: grid_search.best_score_):
        with open(f'{model_name}_best_model.pkl', 'wb') as f:
            pickle.dump(best_models[model_name], f)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_}\n")

# Optionally, load the best model later
# with open(f'{model_name}_best_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f'{model_name} results:')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\n')

pipeline_rf = Pipeline(steps=[
    ('classifier', best_models['RandomForest'])
])

pipeline_ab = Pipeline(steps=[
    ('classifier', best_models['AdaBoost'])
])

pipeline_gb = Pipeline(steps=[
    ('classifier', best_models['GradientBoosting'])
])

pipeline_mlp = Pipeline(steps=[
    ('classifier', best_models['MLPClassifier'])
])

pipeline_svc = Pipeline(steps=[
    ('classifier', best_models['SVC'])
])

# Create the StackingClassifier with pipelines
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', pipeline_rf),
        ('ab', pipeline_ab),
        ('gb', pipeline_gb),
        ('mlp', pipeline_mlp),
        ('svc', pipeline_svc)
    ],
    final_estimator=RandomForestClassifier(),
    cv=5
)

# Fit the StackingClassifier
stacking_clf.fit(X_train, y_train)

# Evaluate the StackingClassifier
y_pred_stacking = stacking_clf.predict(X_test)
print("Stacking Classifier:")
print(f"Classification report:\n{classification_report(y_test, y_pred_stacking)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred_stacking)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_stacking)}\n")

# Save the StackingClassifier
with open('stacking_classifier.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f)