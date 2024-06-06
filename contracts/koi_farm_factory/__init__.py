# Initializes the koi_farm_factory module

from .KoiFarmFactory import *
from .KoiFarmFactory import abi as KoiFarmFactoryABI

KOI_FARM_FACTORY_ADDRESS = "0x8f3470A75a72Bb909cE66698E1A5BE9e325818D3"

koi_farm_factory = get_contract_instance("KoiFarmFactory", NETWORK)
koi_farm_factory.address = KOI_FARM_FACTORY_ADDRESS
koi_farm_factory.abi = KoiFarmFactoryABI