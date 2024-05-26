# Main logic of the agent

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import web3
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.providers.async_rpc import AsyncHTTPProvider

from utils.constants import *
from utils.helpers import *
from utils.logger import *
from transaction_manager.transaction_executor import TransactionExecutor
from transaction_manager.transaction_confirmator import TransactionConfirmator
from transaction_manager.gas_estimator import GasEstimator
from strategy_engine.machine_learning_model import MachineLearningModel
from strategy_engine.prediction_model import PredictionModel
from strategy_engine.strategy_optimizer import StrategyOptim