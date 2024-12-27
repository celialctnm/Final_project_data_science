import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import component

warnings.filterwarnings('ignore')

def svm_model():
    print("_____ SVM _____")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = SVC(kernel='rbf', C=10, gamma='auto')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    component.evaluation_model(y_test, y_pred)

    # Confusion Matrix
    component.draw_confusion_matrix("SVM", y, y_test, y_pred)

    return model


def random_forest_model():
    print("_____ RANDOM FOREST _____")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    weight_configs = [
        {'Happy': 1.5, 'Sad': 0.8, 'Energetic': 1.2, 'Calm': 0.9},
        {'Happy': 2.0, 'Sad': 0.7, 'Energetic': 1.5, 'Calm': 0.8},
        {'Happy': 2.5, 'Sad': 0.6, 'Energetic': 1.8, 'Calm': 0.7},
        {'Happy': 1.5, 'Sad': 0.7, 'Energetic': 1.5, 'Calm': 0.8},
    ]

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 10, None],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5],
    }

    grid = ParameterGrid({'class_weight': weight_configs, **param_grid})

    best_f1_score = 0
    best_params = None
    best_report = None

    for params in grid:

        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            min_samples_split=params['min_samples_split'],
            class_weight=params['class_weight'],
            random_state=42
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        avg_f1_score = report['macro avg']['f1-score']

        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_params = params
            best_report = classification_report(y_test, y_pred)

    print("\n Best configuration :")
    print(best_params)
    print("\n Best Classification Report :")
    print(best_report)

    best_rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        class_weight=best_params['class_weight'],
        random_state=42
    )

    best_rf.fit(X_train, y_train)
    y_pred_best = best_rf.predict(X_test)

    component.draw_confusion_matrix("Random Forest", y, y_test, y_pred_best)

    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

    return best_rf

def KNN_model():
    print("_____ KNN _____")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best hyperparamètres : ", grid_search.best_params_)
    print("Best score : ", grid_search.best_score_)

    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)

    component.evaluation_model(y_test, y_pred)
    component.draw_confusion_matrix("KNN", y, y_test, y_pred)

# loading of dataset (available on kaggle)
music_data = pd.read_csv("../data/final_dataset/data_moods.csv")
print(music_data.head(5))

X = music_data.drop(columns=['mood', 'name', 'album', 'artist', 'id', 'release_date'])
y = music_data['mood']
feature_names = X.columns

component.graph_dataset(1, music_data,X,y)

# SVM MODEL
svm_model()

# RANDOM FOREST
random_forest_model()

# KNN
KNN_model()