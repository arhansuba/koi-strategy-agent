import requests


import requests
import pandas as pd
from sqlalchemy import create_engine

class DataFetcher:
    def __init__(self, source_type, source_config):
        """
        Initialize the data fetcher with the source type and configuration.

        :param source_type: The type of source (e.g. API, database, file)
        :param source_config: The configuration for the source (e.g. API endpoint, database connection string, file path)
        """
        self.source_type = source_type
        self.source_config = source_config

    def fetch_data(self):
        """
        Fetch data from the source.

        :return: The fetched data
        """
        if self.source_type == 'API':
            return self.fetch_data_from_api()
        elif self.source_type == 'database':
            return self.fetch_data_from_database()
        elif self.source_type == 'file':
            return self.fetch_data_from_file()
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

    def fetch_data_from_api(self):
        """
        Fetch data from an API.

        :return: The fetched data
        """
        url = self.source_config['url']
        params = self.source_config.get('params', {})
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_data_from_database(self):
        """
        Fetch data from a database.

        :return: The fetched data
        """
        connection_string = self.source_config['connection_string']
        engine = create_engine(connection_string)
        query = self.source_config['query']
        data = pd.read_sql_query(query, engine)
        return data

    def fetch_data_from_file(self):
        """
        Fetch data from a file.

        :return: The fetched data
        """
        file_path = self.source_config['file_path']
        data = pd.read_csv(file_path)
        return data