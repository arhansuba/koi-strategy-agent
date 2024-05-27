import asyncio
from zksync import Web3, ZkSync

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

class GasEstimator:
    def __init__(self, data, target_variable):
        """
        Initialize the gas estimator.

        :param data: The data to estimate gas consumption from
        :param target_variable: The target variable to predict (gas consumption)
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
        self.data[['temperature', 'humidity', 'pressure']] = imputer.fit_transform(self.data[['temperature', 'humidity', 'pressure']])

        # Encode categorical variables using one-hot encoding
        self.data = pd.get_dummies(self.data, columns=['season', 'day_of_week'])

        # Scale numerical variables using standard scaling
        scaler = StandardScaler()
        self.data[['temperature', 'humidity', 'pressure']] = scaler.fit_transform(self.data[['temperature', 'humidity', 'pressure']])

        return self.data

    def feature_selection(self):
        """
        Select the most important features using recursive feature elimination.

        :return: The selected features
        """
        # Select the top 10 features using recursive feature elimination
        selector = SelectKBest(f_regression, k=10)
        self.data = selector.fit_transform(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable])

        return self.data

    def train_model(self):
        """
        Train a random forest regressor on the preprocessed data.

        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable], test_size=0.2, random_state=42)

        # Create a pipeline with PCA and random forest regressor
        pipeline = Pipeline([
            ('pca', PCA(n_components=5)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
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

        # Calculate the mean squared error, mean absolute error, and R2 score
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, mae, r2

    def run(self):
        """
        Run the gas estimator.

        :return: The estimated gas consumption and the evaluation metrics
        """
        # Preprocess the data
        self.data = self.preprocess_data()

        # Select the most important features
        self.data = self.feature_selection()

        # Train the model
        model = self.train_model()

        # Evaluate the model
        mse, mae, r2 = self.evaluate_model(model)

        # Make predictions on the entire dataset
        y_pred = model.predict(self.data.drop(self.target_variable, axis=1))

        return y_pred, mse, mae, r2