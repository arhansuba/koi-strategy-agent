import asyncio
from agent.risk_management.portfolio_monitor import PortfolioMonitor
from agent.risk_management.risk_detector import RiskDetector
from agent.risk_management.position_sizer import PositionSizer
async def main():
    user_portfolios = {
        "Alice": {"ETH": 100, "BTC": 50},
        "Bob": {"ETH": 200, "BTC": 100},
        "Charlie": {"ETH": 150, "BTC": 100}
    }

    portfolio_monitor = PortfolioMonitor(user_portfolios)
    await portfolio_monitor.monitor()

    risk_detector = RiskDetector(risk_threshold=10)
    await risk_detector.detect_risk(user_portfolios)

    position_sizer = PositionSizer(risk_threshold=10)
    await position_sizer.size_positions(user_portfolios)

if __name__ == "__main__":
    asyncio.run(main())