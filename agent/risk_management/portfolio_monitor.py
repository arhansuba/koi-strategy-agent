import asyncio
from typing import Dict

class PortfolioMonitor:
    def __init__(self, user_portfolios: Dict[str, Dict[str, float]]):
        self.user_portfolios = user_portfolios

    async def monitor(self):
        # Monitor user portfolios for risk
        for user, portfolio in self.user_portfolios.items():
            total_value = sum(portfolio.values())
            for token, value in portfolio.items():
                percentage = (value / total_value) * 100
                print(f"User {user} has {percentage:.2f}% of their portfolio in {token}")