

import pandas as pd
import numpy as np
from scipy.stats import norm

class RiskDetector:
    def __init__(self, data):
        """
        Initialize the risk detector.

        :param data: The data to analyze for risks
        """
        self.data = data

    def detect_risks(self):
        """
        Detect risks in the data.

        :return: A list of detected risks
        """
        risks = []

        # Calculate the daily returns of the stock
        daily_returns = self.data['Close'].pct_change()

        # Calculate the volatility of the stock
        volatility = daily_returns.std() * np.sqrt(252)

        # Calculate the Value-at-Risk (VaR) of the stock
        var = self.calculate_var(daily_returns, 0.95)

        # Check if the stock price is above the VaR
        if self.data['Close'][-1] > var:
            risks.append('Stock price is above the Value-at-Risk')

        # Calculate the Expected Shortfall (ES) of the stock
        es = self.calculate_es(daily_returns, 0.95)

        # Check if the stock price is above the ES
        if self.data['Close'][-1] > es:
            risks.append('Stock price is above the Expected Shortfall')

        return risks

    def calculate_risk_level(self, stock_price, volatility):
        """
        Calculate the risk level of a stock based on its price and volatility.

        :param stock_price: The current price of the stock
        :param volatility: The volatility of the stock
        :return: The risk level of the stock
        """
        # Calculate the z-score of the stock price
        z_score = (stock_price - self.data['Close'].mean()) / self.data['Close'].std()

        # Calculate the risk level based on the z-score and volatility
        risk_level = norm.cdf(z_score) * volatility

        return risk_level

    def calculate_var(self, daily_returns, confidence_level):
        """
        Calculate the Value-at-Risk (VaR) of a stock.

        :param daily_returns: The daily returns of the stock
        :param confidence_level: The confidence level for the VaR
        :return: The VaR of the stock
        """
        # Calculate the VaR using the historical simulation method
        var = np.percentile(daily_returns, (1 - confidence_level) * 100)

        return var

    def calculate_es(self, daily_returns, confidence_level):
        """
        Calculate the Expected Shortfall (ES) of a stock.

        :param daily_returns: The daily returns of the stock
        :param confidence_level: The confidence level for the ES
        :return: The ES of the stock
        """
        # Calculate the ES using the historical simulation method
        es = np.mean(daily_returns[daily_returns <= self.calculate_var(daily_returns, confidence_level)])

        return es