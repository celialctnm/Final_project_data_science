import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
import component

warnings.filterwarnings('ignore')


# SVM Model
def SVM_model():
    print("_____ SVM _____")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Training
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', C=10, gamma='scale')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    component.evaluation_model(y_test, y_pred)

    # Confusion Matrix
    component.draw_confusion_matrix("SVM", y, y_test, y_pred)

    return model


def randomForest_model():

    print("_____ RANDOM FOREST _____")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    component.evaluation_model(y_test, y_pred)

    # Confusion Matrix
    component.draw_confusion_matrix("Random Forest", y, y_test, y_pred)

    return model


# loading of our custom data
music_data = pd.read_csv("../data/custom_dataset/custom_data.csv")
music_data.drop(columns=['music_name', 'artist(s)_name'], inplace=True)

# conversion in float/int to avoid constraint
numeric_cols = music_data.select_dtypes(include=['float64', 'int64']).columns
music_data[numeric_cols] = music_data[numeric_cols].fillna(music_data[numeric_cols].mean())

categorical_cols = music_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    music_data[col].fillna(music_data[col].mode()[0], inplace=True)

if 'key' in music_data.columns:
    music_data['key'] = music_data['key'].astype('category').cat.codes

if 'mode' in music_data.columns:
    music_data['mode'] = music_data['mode'].astype('category').cat.codes

print("Shape : ", music_data.shape)

X = music_data.drop(columns=['emotion'])
y = music_data['emotion']

component.graph_dataset(0, music_data, X, y)
SVM_model()
randomForest_model()