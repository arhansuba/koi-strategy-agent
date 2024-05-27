import os

class Config:
    def __init__(self):
        environment = os.getenv('ENVIRONMENT', 'development')
        if environment == 'development':
            from environments.development_config import DevelopmentConfig as EnvConfig
        elif environment == 'testing':
            from environments.testing_config import TestingConfig as EnvConfig
        elif environment == 'production':
            from environments.production_config import ProductionConfig as EnvConfig
        else:
            raise ValueError("Invalid environment name")
        
        self.load_config(EnvConfig)

    def load_config(self, EnvConfig):
        for key, value in EnvConfig.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)

# Example usage
config = Config()
print(config.rpc_url)

# Configuration parameters for the agent

# Network settings
NETWORK = "zksync"  # Can be "mainnet", "ropsten", or "zksync"

# Contract addresses
KOI_TOKEN_ADDRESS = "0x8f3470A75a72Bb909cE66698E1A5BE9e325818D3"
MUTE_AMPLIFIER_REDUX_ADDRESS = "0x4772D618AD88b602a2ea76F2155D0356E6756b3e"

# Gas settings
GAS_PRICE = 1000000000
GAS_LIMIT = 1000000

# Farming pool settings
POOL_ID = 1
MINIMUM_APY = 0.05
MAXIMUM_APY = 0.15

# Risk management settings
MAXIMUM_POSITION_SIZE = 0.5
MAXIMUM_DRAWDOWN = 0.1