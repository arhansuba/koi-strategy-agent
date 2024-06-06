import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

class StrategyOptimizer:
    def __init__(self, data, target_variable, strategy_params):
        """
        Initialize the strategy optimizer.

        :param data: The data to optimize the strategy on
        :param target_variable: The target variable to predict
        :param strategy_params: The parameters of the strategy to optimize
        """
        self.data = data
        self.target_variable = target_variable
        self.strategy_params = strategy_params

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

    def train_model(self, X, y, params):
        """
        Train a random forest classifier on the preprocessed data with the given parameters.

        :param X: The feature matrix
        :param y: The target variable
        :param params: The parameters to use for training the model
        :return: The trained model
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a pipeline with PCA and random forest classifier
        pipeline = Pipeline([
            ('pca', PCA(n_components=5)),
            ('rf', RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42))
        ])

        # Train the model on the training data
        pipeline.fit(X_train, y_train)

        return pipeline

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

    def objective_function(self, params):
        """
        Define the objective function to minimize.

        :param params: The parameters to optimize
        :return: The objective function value
        """
        # Preprocess the data
        self.data = self.preprocess_data()

        # Select the most important features
        X_selected, selected_features = self.feature_selection()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_selected, self.data[self.target_variable], test_size=0.2, random_state=42)

        # Train the model with the given parameters
        model = self.train_model(X_train, y_train, params)

        # Evaluate the model
        accuracy, report, matrix = self.evaluate_model(model, X_test, y_test)

        # Return the negative accuracy as the objective function value
        return -accuracy

    def optimize_strategy(self):
        """
Optimize the strategy using the scipy minimize function.

        :return: The optimized parameters and the minimum objective function value
        """
        # Define the bounds for the parameters
        bounds = [(10, 100), (5, 20)]

        # Define the initial guess for the parameters
        initial_guess = [50, 10]

        # Minimize the objective function using the scipy minimize function
        result = minimize(self.objective_function, initial_guess, method="SLSQP", bounds=bounds)

        # Return the optimized parameters and the minimum objective function value
        return result.x, result.fun

    def run(self):
        """
        Run the strategy optimizer.

        :return: The optimized parameters and the minimum objective function value
        """
        # Optimize the strategy
        params, min_obj_func_val = self.optimize_strategy()

        return params, min_obj_func_val

