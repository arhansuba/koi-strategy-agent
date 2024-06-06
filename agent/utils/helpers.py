

import json
import requests
from web3 import Web3
from .constants import MAINNET_RPC_URL, ROPSTEN_RPC_URL, ZKSYNC_RPC_URL
from agent.utils import constants

def get_contract_abi(contract_name: str) -> dict:
    """Loads the ABI for a given contract from the contracts/ directory."""
    with open(f"contracts/{contract_name}/{contract_name}.json", "r") as f:
        contract_abi = json.load(f)["abi"]
    return contract_abi

def get_web3_provider(network: str) -> Web3:
    """Returns a Web3 provider for the given network."""
    if network == "mainnet":
        rpc_url = MAINNET_RPC_URL
    elif network == "ropsten":
        rpc_url = ROPSTEN_RPC_URL
    elif network == "zksync":
        rpc_url = ZKSYNC_RPC_URL
    else:
        raise ValueError(f"Invalid network: {network}")
    return Web3(Web3.HTTPProvider(rpc_url))

def get_contract_instance(contract_name: str, network: str) -> Web3.Contract:
    """Returns a Web3 contract instance for the given contract and network."""
    contract_abi = get_contract_abi(contract_name)
    web3 = get_web3_provider(network)
    if network == "zksync":
        contract_address = web3.zksync.contract_address(contract_name)
    else:
        contract_address = getattr(constants, f"{contract_name.upper()}_ADDRESS")
    return web3.eth.contract(address=contract_address, abi=contract_abi)