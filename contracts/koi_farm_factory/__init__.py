# Initializes the koi_farm_factory module

import json
from agent.config import NETWORK
from agent.utils.helpers import get_contract_instance
#from .KoiFarmFactory import *

# JSON dosyasının tam yolunu belirtin
koi_farm_factory_abi_path = "/home/arhan/koi-strategy-agent/contracts/koi_farm_factory/KoiFarmFactory.json"

# JSON dosyasını açarak ABI'yi yükleyin
with open(koi_farm_factory_abi_path, 'r') as file:
    KoiFarmFactoryABI = json.load(file)

KOI_FARM_FACTORY_ADDRESS = "0x8f3470A75a72Bb909cE66698E1A5BE9e325818D3"

# Sözleşme örneğini alın
koi_farm_factory = get_contract_instance("KoiFarmFactory", NETWORK)
koi_farm_factory.address = KOI_FARM_FACTORY_ADDRESS
koi_farm_factory.abi = KoiFarmFactoryABI
