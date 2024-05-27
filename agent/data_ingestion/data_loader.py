

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_format):
        """
        Initialize the data loader with the desired data format.

        :param data_format: The desired format for the loaded data (e.g. DataFrame, NumPy array)
        """
        self.data_format = data_format

    def load_data(self, data):
        """
        Load the fetched data into the desired format.

        :param data: The fetched data
        :return: The loaded data
        """
        if self.data_format == 'DataFrame':
            return self.load_data_to_dataframe(data)
        elif self.data_format == 'NumPy array':
            return self.load_data_to_numpy_array(data)
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")

    def load_data_to_dataframe(self, data):
        """
        Load the fetched data into a pandas DataFrame.

        :param data: The fetched data
        :return: The loaded data
        """
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        elif isinstance(data, list):
            if isinstance(data[0], dict):
                data = [pd.DataFrame(d) for d in data]
                data = pd.concat(data, ignore_index=True)
            elif isinstance(data[0], (int, float)):
                data = np.array(data)
                data = pd.DataFrame(data, columns=[f'column_{i}' for i in range(len(data))])
        return data

    def load_data_to_numpy_array(self, data):
        """
        Load the fetched data into a NumPy array.

        :param data: The fetched data
        :return: The loaded data
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif isinstance(data, list):
            if isinstance(data[0], (int, float)):
                data = np.array(data)
            elif isinstance(data[0], dict):
                data = [np.array(list(d.values())) for d in data]
                data = np.vstack(data)
        return data