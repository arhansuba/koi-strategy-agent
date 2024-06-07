import asyncio
import logging
import json
import numpy as np
from typing import Dict
from risk_management.portfolio_monitor import PortfolioMonitor
from risk_management.risk_detector import RiskDetector
from risk_management.position_sizer import PositionSizer
from strategy_engine.machine_learning_model import MachineLearningModel
from strategy_engine.prediction_model import PredictionModel
from strategy_engine.strategy_optimizer import StrategyOptimizer
from transaction_manager.gas_estimator import GasEstimator
from transaction_manager.transaction_confirmator import TransactionConfirmator
from transaction_manager.transaction_executor import TransactionExecutor
from data_ingestion.data_fetcher import DataFetcher
from data_ingestion.data_loader import DataLoader
from data_ingestion.data_preprocessor import DataPreprocessor
from config import Config
from utils.logger import setup_logger

from giza.agents import AgentResult, GizaAgent
from giza.agents.task import task

# Set up logging
logger_name = "my_logger"
log_file = "agent.log"  # Boş bir değer
setup_logger(logger_name, log_file)





class Agent:
    def __init__(self, config: Config, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self.user_portfolios = self.data_loader.load_user_portfolios()

        # Risk Management
        data =  data_loader = DataLoader(data_file="data/user_portfolios.json")
        self.risk_detector = RiskDetector(data)
        self.position_sizer = PositionSizer(risk_tolerance=self.config.risk_threshold, account_size=self.user_portfolios)
        self.portfolio_monitor = PortfolioMonitor(self.user_portfolios)

        # Strategy Engine
        self.ml_model = MachineLearningModel(data=self.user_portfolios, target_variable=self.user_portfolios)
        self.prediction_model = PredictionModel(data=self.user_portfolios, target_variable=self.user_portfolios)
        self.strategy_optimizer = StrategyOptimizer(data=self.user_portfolios, target_variable=self.user_portfolios, strategy_params=self.user_portfolios)

        # Transaction Management
        self.gas_estimator = GasEstimator(data=self.user_portfolios, target_variable=self.user_portfolios)
        self.transaction_confirmator = TransactionConfirmator(data=self.user_portfolios, target_variable=self.user_portfolios)
        self.transaction_executor = TransactionExecutor(db_url="sqlite:///transactions.db", confirmator=self.transaction_confirmator)

        # Data Ingestion
        self.data_fetcher = DataFetcher(koi_finance_api_url='https://dapp.koi.finance/pool', web3_provider_url="http://127.0.0.1:8545" )
        self.data_preprocessor = DataPreprocessor()

    @task 
    async def create_agent(self):
        agent = GizaAgent(
            id=self.config.model_id,
            version_id=self.config.version_id,
            chain=f"ethereum:sepolia:{self.config.sepolia_rpc_url}",
            account=self.config.account,
        )
        return agent

    @task
    async def predict(self, agent: GizaAgent, X: np.ndarray):
        prediction = await Predict(input_feed={"val": X}, verifiable=True, job_size="XL")
        return prediction

    @task
    async def get_pred_val(self, prediction: AgentResult):
        return prediction.value[0][0]

    async def run(self):
        agent = await self.create_agent()
        X = np.array([[4.20, 0.1]])  # Replace with actual data
        prediction = await self.predict(agent, X)
        predicted_value = await self.get_pred_val(prediction)

        tasks = [
            self.portfolio_monitor_task(),
            self.risk_detector_task(),
            self.position_sizer_task(),
            self.trader_task(),
            self.data_fetch_task(),
            self.data_preprocess_task(),
            self.ml_model_task(),
            self.strategy_optimize_task(),
            self.gas_estimate_task(),
            self.transaction_confirm_task(),
            self.transaction_execute_task()
        ]

        await asyncio.gather(*tasks)

    async def portfolio_monitor_task(self):
        while True:
            logging.info("Monitoring user portfolios...")
            await asyncio.sleep(1)
            for user, portfolio in self.user_portfolios.items():
                logging.info(f"User {user} portfolio: {portfolio}")
            await asyncio.sleep(5)  # Monitor every 5 seconds

    async def risk_detector_task(self):
        while True:
            logging.info("Detecting risks in user portfolios...")
            await asyncio.sleep(2)
            for user, portfolio in self.user_portfolios.items():
                risk_level = self.risk_detector.calculate_risk_level(portfolio)
                if risk_level > self.config.risk_threshold:
                    logging.info(f"Risk detected for user {user}: {risk_level}")
            await asyncio.sleep(10)  # Detect risks every 10 seconds

    async def position_sizer_task(self):
        while True:
            logging.info("Sizing positions based on risk levels...")
            await asyncio.sleep(3)
            for user, portfolio in self.user_portfolios.items():
                risk_level = self.position_sizer.calculate_risk_level(portfolio)
                if risk_level > self.config.risk_threshold:
                    self.position_sizer.size_position(user, portfolio)
            await asyncio.sleep(15)  # Size positions every 15 seconds

    async def trader_task(self):
        while True:
            logging.info("Trading based on position sizes...")
            await asyncio.sleep(4)
            for user, portfolio in self.user_portfolios.items():
                position_size = self.position_sizer.get_position_size(user, portfolio)
                if position_size > 0:
                    self.trader.execute_trade(user, portfolio, position_size)
            await asyncio.sleep(20)  # Trade every 20 seconds

    async def data_fetch_task(self):
        while True:
            logging.info("Fetching data...")
            await asyncio.sleep(6)
            data = self.data_fetcher.fetch_data()
            logging.info(f"Fetched data: {data}")
            await asyncio.sleep(30)  # Fetch data every 30 seconds

    async def data_preprocess_task(self):
        while True:
            logging.info("Preprocessing data...")
            await asyncio.sleep(7)
            preprocessed_data = self.data_preprocessor.preprocess_data()
            logging.info(f"Preprocessed data: {preprocessed_data}")
            await asyncio.sleep(30)  # Preprocess data every 30 seconds

    async def ml_model_task(self):
        while True:
            logging.info("Running machine learning model...")
            await asyncio.sleep(8)
            ml_result = self.ml_model.run()
            logging.info(f"Machine learning model result: {ml_result}")
            await asyncio.sleep(30)  # Run ML model every 30 seconds

    async def strategy_optimize_task(self):
        while True:
            logging.info("Optimizing strategy...")
            await asyncio.sleep(9)
            optimized_strategy = self.strategy_optimizer.optimize()
            logging.info(f"Optimized strategy: {optimized_strategy}")
            await asyncio.sleep(30)  # Optimize strategy every 30 seconds

    async def gas_estimate_task(self):
        while True:
            logging.info("Estimating gas costs...")
            await asyncio.sleep(10)
            gas_estimate = self.gas_estimator.estimate_gas()
            logging.info(f"Gas estimate: {gas_estimate}")
            await asyncio.sleep(30)  # Estimate gas every 30 seconds

    async def transaction_confirm_task(self):
        while True:
            logging.info("Confirming transactions...")
            await asyncio.sleep(11)
            confirmed = self.transaction_confirmator.confirm_transactions()
            logging.info(f"Confirmed transactions: {confirmed}")
            await asyncio.sleep(30)  # Confirm transactions every 30 seconds

    async def transaction_execute_task(self):
        while True:
            logging.info("Executing transactions...")
            await asyncio.sleep(12)
            executed = self.transaction_executor.execute_transactions()
            logging.info(f"Executed transactions: {executed}")
            await asyncio.sleep(30)  # Execute transactions every 30 seconds

class Config:
    def __init__(self, risk_threshold: float, exchange: str, model_id: int, version_id: int, sepolia_rpc_url: str, account: str):
        self.risk_threshold = risk_threshold
        self.exchange = exchange
        self.model_id = model_id
        self.version_id = version_id
        self.sepolia_rpc_url = sepolia_rpc_url
        self.account = account

class DataLoader:
    def __init__(self, data_file: str):
        self.data_file = data_file

    def load_user_portfolios(self) -> Dict[str, Dict[str, float]]:
        # Load user portfolios from data file
        with open(self.data_file, "r") as f:
            data = json.load(f)
        return data

if __name__ == "__main__":
    config = Config(
        risk_threshold=10,
        exchange="Binance",
        model_id=1,
        version_id=1,
        sepolia_rpc_url="https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID",
        account="0xYourEthereumAddress"
    )
    data_loader = DataLoader(data_file="data/user_portfolios.json")

    agent = Agent(config, data_loader)
async def main():
   
    agent = Agent(config, data_loader)
    await agent.run()
    try:
        asyncio.run(asyncio.run(main()))
    except KeyboardInterrupt:
        logging.info("Shutting down...")
