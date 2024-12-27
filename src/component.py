import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def graph_dataset(subplot_value, music_data, X, y):
    sns.set(style="whitegrid")
    emotion_counts = y.value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("deep", len(emotion_counts)))
    plt.title('Distribution of emotion in the dataset \n')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(15, 10))

    sub = 4
    if subplot_value == 0:
        sub = 3
    elif subplot_value == 1:
        sub = 4

    for i, column in enumerate(X.columns, 1):
        plt.subplot(sub, sub, i)
        sns.violinplot(x=y, y=column, data=music_data, inner='quartile')
        plt.title(f'Distribution of {column} by emotion')
        plt.xlabel('emotion')
        plt.ylabel(column)

    plt.tight_layout()
    plt.show()


# Evaluation
def evaluation_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# Visualization
def draw_confusion_matrix(model, y, y_test, y_pred):
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result)

    class_names = y.unique()
    cm_display.display_labels = class_names

    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.grid(False)
    plt.title('Confusion Matrix ' + model)
    plt.show()
