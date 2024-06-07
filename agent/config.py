import os

class Config:
    def __init__(self):
        # Configuration parameters for the agent
        self.chain_id = 1
        self.network_id = 1
        # Network settings
        self.network = "zksync"  # Can be "mainnet", "ropsten", or "zksync"

        # Contract addresses
        self.koi_token_address = "0x8f3470A75a72Bb909cE66698E1A5BE9e325818D3"
        self.mute_amplifier_redux_address = "0x4772D618AD88b602a2ea76F2155D0356E6756b3e"

        # Gas settings
        self.gas_price = 1000000000
        self.gas_limit = 1000000

        # Farming pool settings
        self.pool_id = 1
        self.minimum_apy = 0.05
        self.maximum_apy = 0.15

        # Risk management settings
        self.maximum_position_size = 0.5
        self.maximum_drawdown = 0.1

        # Optionally, load environment variables if they are set
        self.load_env_variables()

    def load_env_variables(self):
        self.chain_id = int(os.getenv("CHAIN_ID", self.chain_id))
        self.network_id = int(os.getenv("NETWORK_ID", self.network_id))
        self.network = os.getenv("NETWORK", self.network)
        self.koi_token_address = os.getenv("KOI_TOKEN_ADDRESS", self.koi_token_address)
        self.mute_amplifier_redux_address = os.getenv("MUTE_AMPLIFIER_REDUX_ADDRESS", self.mute_amplifier_redux_address)
        self.gas_price = int(os.getenv("GAS_PRICE", self.gas_price))
        self.gas_limit = int(os.getenv("GAS_LIMIT", self.gas_limit))
        self.pool_id = int(os.getenv("POOL_ID", self.pool_id))
        self.minimum_apy = float(os.getenv("MINIMUM_APY", self.minimum_apy))
        self.maximum_apy = float(os.getenv("MAXIMUM_APY", self.maximum_apy))
        self.maximum_position_size = float(os.getenv("MAXIMUM_POSITION_SIZE", self.maximum_position_size))
        self.maximum_drawdown = float(os.getenv("MAXIMUM_DRAWDOWN", self.maximum_drawdown))


config = Config()
print(config.network)
print(config.koi_token_address)
print(config.gas_price)
