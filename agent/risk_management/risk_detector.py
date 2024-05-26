import asyncio
from typing import Dict

class RiskDetector:
    def __init__(self, risk_threshold: float):
        self.risk_threshold = risk_threshold

    async def detect_risk(self, user_portfolios: Dict[str, Dict[str, float]]):
        # Detect potential risks in user portfolios
        for user, portfolio in user_portfolios.items():
            for token, value in portfolio.items():
                percentage = (value / sum(portfolio.values())) * 100
                if percentage > self.risk_threshold:
                    print(f"User {user} has {percentage:.2f}% of their portfolio in {token}, which is above the risk threshold of {self.risk_threshold:.2f}%")