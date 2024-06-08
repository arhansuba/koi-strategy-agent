import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import onnx
from onnx import optimizer
from onnx import shape_inference
import torch

class MachineLearningModel:
    def __init__(self, data, target_variable):
        """
        Initialize the machine learning model.

        :param data: The data to train the model on
        :param target_variable: The target variable to predict
        """
        self.data = data
        self.target_variable = target_variable

    def preprocess_data(self):
        """
        Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical variables.

        :return: The preprocessed data
        """
        # Handle missing values using mean imputation
        imputer = SimpleImputer(strategy='mean')
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numerical_features] = imputer.fit_transform(self.data[numerical_features])

        # Encode categorical variables using one-hot encoding
        categorical_features = self.data.select_dtypes(include=['object']).columns
        self.data = pd.get_dummies(self.data, columns=categorical_features)

        # Scale numerical variables using standard scaling
        scaler = StandardScaler()
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

        return self.data

    def feature_selection(self):
        """
        Select the most important features using recursive feature elimination.

        :return: The selected features
        """
        # Select the top 10 features using recursive feature elimination
        selector = SelectKBest(f_classif, k=10)
        X = self.data.drop(self.target_variable, axis=1)
        y = self.data[self.target_variable]
        X_selected = selector.fit_transform(X, y)

        # Get the selected feature names
        selected_features = X.columns[selector.get_support(indices=True)]

        return X_selected, selected_features

    def train_model(self, X, y):
        """
        Train a random forest classifier on the preprocessed data.

        :param X: The feature matrix
        :param y: The target variable
        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the pipeline with PCA and random forest classifier
        pipeline = Pipeline([
            ('pca', PCA(n_components=5)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Define the hyperparameter tuning space
        param_grid = {
            'pca__n_components': [3, 5, 7],
            'rf__n_estimators': [50, 100, 200]
        }

        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
        grid_search.fit(X_train, y_train)

        # Train the best model on the entire training set
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        return best_model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the performance of the trained model on the testing data.

        :param model: The trained model
        :param X_test: The testing feature matrix
        :param y_test: The testing target variable
        :return: The evaluation metrics
        """
        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Calculate the accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        return accuracy, report, matrix

    def run(self):
        """
        Run the machine learning pipeline.

        :return: The evaluation metrics
        """
        # Preprocess the data
        self.data = self.preprocess_data()

        # Select the most important features
        X_selected, selected_features = self.feature_selection()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_selected, self.data[self.target_variable], test_size=0.2, random_state=42)

        # Train the model
        model = self.train_model(X_train, y_train)

        # Evaluate the model
        accuracy, report, matrix= self.evaluate_model(model, X_test, y_test)

        return accuracy, report, matrix
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [6, 7, 8, 9, 10],
    'target': [0, 1, 0, 1, 0]  # Hedef değişken
})

# Hedef değişkenin adı
target_variable = 'target'

model = MachineLearningModel(data, target_variable)  
X_train, X_test, y_train, y_test = train_test_split(data, target_variable, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Eğitilmiş modeli kaydedin
torch.save(model, "machine_learning_model.pt")
# Eğitilmiş PyTorch modelinin yüklenmesi
model = torch.load("machine_learning_model.pt")

# Modeli ONNX formatına dönüştürme
input_sample = torch.randn(1, 3, 224, 224)
onnx_path = "machine_learning_model.onnx"
torch.onnx.export(model, input_sample, onnx_path, export_params=True)
