import requests
from web3 import Web3

class DataFetcher:
    def __init__(self, koi_finance_api_url, web3_provider_url):
        """
        Initialize the data fetcher with the Koi Finance API URL and Web3 provider URL.

        :param koi_finance_api_url: The URL of the Koi Finance API
        :param web3_provider_url: The URL of the Web3 provider (e.g. Infura, Alchemy)
        """
        self.koi_finance_api_url = koi_finance_api_url
        self.web3_provider_url = web3_provider_url
        self.w3 = Web3(Web3.HTTPProvider(web3_provider_url))

    def fetch_user_portfolio(self, user_address):
        """
        Fetch a user's portfolio data from Koi Finance.

        :param user_address: The Ethereum address of the user
        :return: The user's portfolio data
        """
        # Call the Koi Finance API to get the user's portfolio data
        response = requests.get(f'{self.koi_finance_api_url}/users/{user_address}/portfolio')
        response.raise_for_status()
        return response.json()

    def fetch_token_prices(self, token_addresses):
        """
        Fetch the prices of a list of tokens from Koi Finance.

        :param token_addresses: A list of Ethereum addresses of the tokens
        :return: A dictionary with the token prices
        """
        # Call the Koi Finance API to get the token prices
        response = requests.get(f'{self.koi_finance_api_url}/tokens/prices', params={'addresses': token_addresses})
        response.raise_for_status()
        return response.json()

    def fetch_contract_data(self, contract_address, function_name, *args):
        """
        Fetch data from a Koi Finance smart contract.

        :param contract_address: The Ethereum address of the contract
        :param function_name: The name of the function to call
        :param args: The arguments to pass to the function
        :return: The result of the function call
        """
        # Get the contract instance
        contract_instance = self.w3.eth.contract(address=contract_address, abi=self.get_abi(contract_address))

        # Call the function on the contract
        result = contract_instance.functions[function_name](*args).call()

        return result

    def get_abi(self, contract_address):
        """
        Get the ABI (Application Binary Interface) for a Koi Finance smart contract.

        :param contract_address: The Ethereum address of the contract
        :return: The ABI of the contract
        """
        # Call the Koi Finance API to get the ABI
        response = requests.get(f'{self.koi_finance_api_url}/contracts/{contract_address}/abi')
        response.raise_for_status()
        return response.json()