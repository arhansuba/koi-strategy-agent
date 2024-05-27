

import numpy as np
import pandas as pd

class PortfolioMonitor:
    def __init__(self, portfolio):
        """
        Initialize the portfolio monitor.

        :param portfolio: The portfolio to monitor
        """
        self.portfolio = portfolio

    def monitor_portfolio(self):
        """
        Monitor the portfolio for risks.

        :return: A report of any risks detected
        """
        # Calculate the current value of the portfolio
        portfolio_value = self.calculate_portfolio_value()

        # Calculate the current risk of the portfolio
        portfolio_risk = self.calculate_portfolio_risk()

        # Check if the portfolio value is below a minimum threshold
        if portfolio_value < self.portfolio.minimum_value:
            report = "Portfolio value is below the minimum threshold."

        # Check if the portfolio risk is above a maximum threshold
        elif portfolio_risk > self.portfolio.maximum_risk:
            report = "Portfolio risk is above the maximum threshold."

        # If there are no issues, report that the portfolio is healthy
        else:
            report = "Portfolio is healthy."

        return report

    def calculate_portfolio_value(self):
        """
        Calculate the current value of the portfolio.

        :return: The current value of the portfolio
        """
        # Calculate the total value of the portfolio by summing the value of each position
        portfolio_value = sum(position.value for position in self.portfolio.positions)
        return portfolio_value

    def calculate_portfolio_risk(self):
        """
        Calculate the current risk of the portfolio.

        :return: The current risk of the portfolio
        """
        # Calculate the total risk of the portfolio by summing the risk of each position
        portfolio_risk = sum(position.risk for position in self.portfolio.positions)
        return portfolio_risk