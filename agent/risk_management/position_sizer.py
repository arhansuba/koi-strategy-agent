

import math

class PositionSizer:
    def __init__(self, risk_tolerance):
        """
        Initialize the position sizer.

        :param risk_tolerance: The risk tolerance of the investor
        """
        self.risk_tolerance = risk_tolerance

    def calculate_position_size(self, stock_price, risk_level):
        """
        Calculate the optimal position size based on the risk level.

        :param stock_price: The current price of the stock
        :param risk_level: The risk level of the stock
        :return: The optimal position size
        """
        # Calculate the maximum position size based on the risk tolerance and risk level
        max_position_size = self.risk_tolerance / risk_level

        # Calculate the number of shares to purchase based on the maximum position size and stock price
        position_size = math.floor(max_position_size / stock_price)

        return position_size