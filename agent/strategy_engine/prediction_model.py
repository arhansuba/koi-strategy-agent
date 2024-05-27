

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class PredictionModel:
    def __init__(self, data, target_variable):
        """
        Initialize the prediction model.

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
        self.data[['numerical_feature1', 'numerical_feature2']] = imputer.fit_transform(self.data[['numerical_feature1', 'numerical_feature2']])

        # Encode categorical variables using one-hot encoding
        self.data = pd.get_dummies(self.data, columns=['categorical_feature1', 'categorical_feature2'])

        # Scale numerical variables using standard scaling
        scaler = StandardScaler()
        self.data[['numerical_feature1', 'numerical_feature2']] = scaler.fit_transform(self.data[['numerical_feature1', 'numerical_feature2']])

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
        Train a gradient boosting regressor on the preprocessed data.

        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable], test_size=0.2, random_state=42)

        # Create a pipeline with PCA and gradient boosting regressor
        pipeline = Pipeline([
            ('pca', PCA(n_components=5)),
            ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
        ])

        # Train the model on the training data
        pipeline.fit(X_train, y_train)

        return pipeline

    def train_xgb_model(self):
        """
        Train an XGBoost regressor on the preprocessed data.

        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable], test_size=0.2, random_state=42)

        # Train the XGBoost model on the training data
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)

        return xgb_model

    def train_lgbm_model(self):
        """
        Train a LightGBM regressor on the preprocessed data.

        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable], test_size=0.2, random_state=42)

        # Train the LightGBM model on the training data
        lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        lgbm_model.fit(X_train, y_train)

        return lgbm_model

    def train_catboost_model(self):
        """
        Train a CatBoost regressor on the preprocessed data.

        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.target_variable, axis=1), self.data[self.target_variable], test_size=0.2, random_state=42)

       # Train the CatBoost model on the training data
        catboost_model = CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        catboost_model.fit(X_train, y_train)

        return catboost_model

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
        Run the prediction pipeline.

        :return: The evaluation metrics
        """
        # Preprocess the data
        self.data = self.preprocess_data()

        # Select the most important features
        self.data = self.feature_selection()

        # Train the gradient boosting regressor
        model = self.train_model()

        # Evaluate the model
        mse, mae, r2 = self.evaluate_model(model)

        return mse, mae, r2

    def run_xgb(self):
        """
        Train and evaluate the XGBoost regressor.

        :return: The evaluation metrics
        """
        # Train the XGBoost model
        model = self.train_xgb_model()

        # Evaluate the model
        mse, mae, r2 = self.evaluate_model(model)

        return mse, mae, r2

    def run_lgbm(self):
        """
        Train and evaluate the LightGBM regressor.

        :return: The evaluation metrics
        """
        # Train the LightGBM model
        model = self.train_lgbm_model()

        # Evaluate the model
        mse, mae, r2 = self.evaluate_model(model)

        return mse, mae, r2

    def run_catboost(self):
        """
        Train and evaluate the CatBoost regressor.

        :return: The evaluation metrics
        """
        # Train the CatBoost model
        model = self.train_catboost_model()

        # Evaluate the model
        mse, mae, r2 = self.evaluate_model(model)

        return mse, mae, r2