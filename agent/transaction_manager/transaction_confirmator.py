import asyncio
from zksync import Web3, ZkSync

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

class TransactionConfirmator:
    def __init__(self, data, target_variable):
        """
        Initialize the transaction confirmator.

        :param data: The data to confirm transactions from
        :param target_variable: The target variable to predict (transaction confirmation)
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
        self.data[['amount', 'balance']] = imputer.fit_transform(self.data[['amount', 'balance']])

        # Encode categorical variables using one-hot encoding
        self.data = pd.get_dummies(self.data, columns=['card_type', 'transaction_type'])

        # Scale numerical variables using standard scaling
        scaler = StandardScaler()
        self.data[['amount', 'balance']] = scaler.fit_transform(self.data[['amount', 'balance']])

        return self.data

    def feature_selection(self):
        """
        Select the most important features using recursive feature elimination.

        :return: The selected features
        """
        # Select the top 10 features using recursive feature elimination
        selector = SelectKBest(f_classif, k=10)
        self.data = selector.fit_transform(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable])

        return self.data

    def train_model(self):
        """
        Train a random forest classifier on the preprocessed data.

        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable], test_size=0.2, random_state=42)

        # Create a pipeline with PCA and random forest classifier
        pipeline = Pipeline([
            ('pca', PCA(n_components=5)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train the model on the training data
        pipeline.fit(X_train, y_train)

        return pipeline

    def evaluate_model(self, model):
        """
        Evaluate the performance of the trained model on the testing data.

        :param model: The trained model
        :return: The evaluation metrics
        """
        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Calculate the accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        return accuracy, report, matrix

    def confirm_transaction(self, transaction_data):
        """
        Confirm a transaction using the trained model.

        :param transaction_data: The transaction data to confirm
        :return: The confirmation result
        """
        # Preprocess the transaction data
        transaction_data = self.preprocess_data(transaction_data)

        # Make a prediction using the trained model
        prediction = self.model.predict(transaction_data)

        # Return the confirmation result
        return prediction

    def run(self):
        """
        Run the transaction confirmator.

        :return: The trained model and the evaluation metrics
        """
        # Preprocess the data
        self.data = self.preprocess_data()

        # Select the most important features
        self.data = self.feature_selection()

        # Train the model
        self.model = self.train_model()

        # Evaluate the model
        accuracy, report, matrix = self.evaluate_model(self.model)

        return self.model, accuracy, report, matrix