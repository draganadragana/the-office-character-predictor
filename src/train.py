import pandas as pd
import time
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Loading preprocessed data with validation
X = pd.read_csv('D:/0 FAKS/2 letnik/machine learning and data mining/The Office - Character Predictor/data/tfidf_features.csv')
y = pd.read_csv('D:/0 FAKS/2 letnik/machine learning and data mining/The Office - Character Predictor/data/cleaned-dataset.csv')['Character']


# Splitting data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Define hyperparameter grids for tuning
param_grid_log_reg = {'C': [0.01, 0.1, 1, 10, 100]}  # Regularization strength
param_grid_nb = {'alpha': [0.01, 0.1, 1.0, 10.0]}     # Smoothing parameter
param_grid_svm = {'C': [0.01, 0.1, 1, 10, 100]}       # Regularization
param_grid_mlp = {
    'hidden_layer_sizes': [(100,), (100, 50), (50, 50)],  # Neuron configurations
    'max_iter': [300, 500, 700]                          # Maximum training iterations
}

# Dictionary of models with their parameter grids
models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=2000, class_weight='balanced'),
        param_grid_log_reg
    ),
    'Multinomial Naive Bayes': (
        MultinomialNB(),
        param_grid_nb
    ),
    'Support Vector Machine': (
        SVC(kernel='linear', class_weight='balanced'),
        param_grid_svm
    ),
    'Neural Network (MLP)': (
        MLPClassifier(random_state=42),
        param_grid_mlp
    )
}

# Function to train, evaluate, and visualize each model
def train_and_evaluate(model, param_grid, model_name):
    print(f"\nTuning and training {model_name}...")

    try:
        start_time = time.time()  # Start timing

        # Grid search with 3-fold cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds.")

        # Use of the best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculating and displaying accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {acc:.2f}")

        # Displaying and saving classification report
        report = classification_report(y_test, y_pred)
        print("Classification Report:\n", report)
        os.makedirs('../data', exist_ok=True)
        with open(f'../data/{model_name.lower().replace(" ", "_")}_report.txt', 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Best Parameters: {grid_search.best_params_}\n")
            f.write(f"Accuracy: {acc:.2f}\n")
            f.write("Classification Report:\n")
            f.write(report)

        # Generating and saving confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)  # Use of model classes
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=best_model.classes_, yticklabels=best_model.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'../data/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

        # Displaying best hyperparameters
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Training and evaluating all models
for name, (model, param_grid) in models.items():
    train_and_evaluate(model, param_grid, name)