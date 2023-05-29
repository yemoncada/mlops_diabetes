from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

MODELS = [
    {
        'name': 'LogisticRegression',
        'model': LogisticRegression,
        'params': [{'C': 0.1}, {'C': 1}, {'C': 10}]
    },
    {
        'name': 'RandomForest',
        'model': RandomForestClassifier,
        'params': [{'n_estimators': 10}, {'n_estimators': 50}, {'n_estimators': 100}]
    },
    {
        'name': 'KNN',
        'model': KNeighborsClassifier,
        'params': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}]
    },
    {
        'name': 'DecisionTree',
        'model': DecisionTreeClassifier,
        'params': [{'max_depth': 3}, {'max_depth': 5}, {'max_depth': 7}]
    },
    {
        'name': 'GradientBoosting',
        'model': GradientBoostingClassifier,
        'params': [{'n_estimators': 50, 'learning_rate': 0.1}, {'n_estimators': 100, 'learning_rate': 0.1}, {'n_estimators': 200, 'learning_rate': 0.1}]
    },
    {
        'name': 'AdaBoost',
        'model': AdaBoostClassifier,
        'params': [{'n_estimators': 50, 'learning_rate': 0.5}, {'n_estimators': 100, 'learning_rate': 0.5}, {'n_estimators': 200, 'learning_rate': 0.5}]
    },
    {
        'name': 'GaussianNB',
        'model': GaussianNB,
        'params': [{}]
    },
    {
        'name': 'MLP',
        'model': MLPClassifier,
        'params': [{'hidden_layer_sizes': (50,)}, {'hidden_layer_sizes': (100,)}, {'hidden_layer_sizes': (200,)}]
    },
]