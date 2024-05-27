

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        """
        Initialize the data preprocessor.
        """
        pass

    def preprocess_data(self, data):
        """
        Preprocess the loaded data.

        :param data: The loaded data
        :return: The preprocessed data
        """
        data = self.handle_missing_values(data)
        data = self.encode_categorical_variables(data)
        data = self.scale_numerical_variables(data)
        return data

    def handle_missing_values(self, data):
        """
        Handle missing values in the data.

        :param data: The data
        :return: The data with missing values handled
        """
        imputer = SimpleImputer(strategy='mean')
        if isinstance(data, pd.DataFrame):
            data_imputed = imputer.fit_transform(data)
            data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
        else:
            data_imputed = imputer.fit_transform(data)
        return data_imputed

    def encode_categorical_variables(self, data):
        """
        Encode categorical variables in the data.

        :param data: The data
        :return: The data with categorical variables encoded
        """
        if isinstance(data, pd.DataFrame):
            categorical_columns = data.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
        return data

    def scale_numerical_variables(self, data):
        """
        Scale numerical variables in the data.

        :param data: The data
        :return: The data with numerical variables scaled
        """
        scaler = StandardScaler()
        if isinstance(data, pd.DataFrame):
            numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        else:
            data = scaler.fit_transform(data)
        return data