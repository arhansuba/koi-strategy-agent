import asyncio
import logging
import random
from typing import Dict, List

from agent.risk_management.portfolio_monitor import PortfolioMonitor
from agent.risk_management.risk_detector import RiskDetector
from agent.risk_management.position_sizer import PositionSizer
from agent.trading.trader import Trader
from agent.data.data_loader import DataLoader
from agent.utils.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class Agent:
    def __init__(self, config: Config, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self.user_portfolios = self.data_loader.load_user_portfolios()
        self.risk_detector = RiskDetector(risk_threshold=self.config.risk_threshold)
        self.position_sizer = PositionSizer(risk_threshold=self.config.risk_threshold)
        self.trader = Trader(exchange=self.config.exchange, api_key=self.config.api_key, api_secret=self.config.api_secret)
        self.portfolio_monitor = PortfolioMonitor(self.user_portfolios)

    async def run(self):
        tasks = [
            self.portfolio_monitor.monitor(),
            self.risk_detector.detect_risk(self.user_portfolios),
            self.position_sizer.size_positions(self.user_portfolios),
            self.trader.trade(self.user_portfolios)
        ]

        await asyncio.gather(*tasks)

    async def portfolio_monitor(self):
        while True:
            logging.info("Monitoring user portfolios...")
            await asyncio.sleep(1)
            for user, portfolio in self.user_portfolios.items():
                logging.info(f"User {user} portfolio: {portfolio}")
            await asyncio.sleep(5)  # Monitor every 5 seconds

    async def risk_detector(self):
        while True:
            logging.info("Detecting risks in user portfolios...")
            await asyncio.sleep(2)
            for user, portfolio in self.user_portfolios.items():
                risk_level = self.risk_detector.calculate_risk_level(portfolio)
                if risk_level > self.config.risk_threshold:
                    logging.info(f"Risk detected for user {user}: {risk_level}")
            await asyncio.sleep(10)  # Detect risks every 10 seconds

    async def position_sizer(self):
        while True:
            logging.info("Sizing positions based on risk levels...")
            await asyncio.sleep(3)
            for user, portfolio in self.user_portfolios.items():
                risk_level = self.position_sizer.calculate_risk_level(portfolio)
                if risk_level > self.config.risk_threshold:
                    self.position_sizer.size_position(user, portfolio)
            await asyncio.sleep(15)  # Size positions every 15 seconds

    async def trader(self):
        while True:
            logging.info("Trading based on position sizes...")
            await asyncio.sleep(4)
            for user, portfolio in self.user_portfolios.items():
                position_size = self.position_sizer.get_position_size(user, portfolio)
                if position_size > 0:
                    self.trader.execute_trade(user, portfolio, position_size)
            await asyncio.sleep(20)  # Trade every 20 seconds

class Config:
    def __init__(self, risk_threshold: float, exchange: str, api_key: str, api_secret: str):
        self.risk_threshold = risk_threshold
        self.exchange = exchange
        self.api_key = api_key
        self.api_secret = api_secret

class DataLoader:
    def __init__(self, data_file: str):
        self.data_file = data_file

    def load_user_portfolios(self) -> Dict[str, Dict[str, float]]:
        # Load user portfolios from data file
        with open(self.data_file, "r") as f:
            data = json.load(f)
        return data

if __name__ == "__main__":
    config = Config(risk_threshold=10, exchange="Binance", api_key="my_api_key", api_secret="my_api_secret")
    data_loader = DataLoader(data_file="data/user_portfolios.json")

    agent = Agent(config, data_loader)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logging.info("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())