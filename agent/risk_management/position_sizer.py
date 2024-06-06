import math

class PositionSizer:
    def __init__(self, risk_tolerance, account_size, leverage=1):
        """
        Initialize the position sizer.

        :param risk_tolerance: The risk tolerance of the investor (as a decimal, e.g. 0.02 for 2%)
        :param account_size: The current size of the trading account
        :param leverage: The leverage to apply to the position size (default: 1)
        """
        self.risk_tolerance = risk_tolerance
        self.account_size = account_size
        self.leverage = leverage

    def calculate_position_size(self, stock_price, risk_level, stop_loss, take_profit):
        """
        Calculate the optimal position size based on the risk level, stop loss, and take profit.

        :param stock_price: The current price of the stock
        :param risk_level: The risk level of the stock (as a decimal, e.g. 0.05 for 5%)
        :param stop_loss: The stop loss price
        :param take_profit: The take profit price
        :return: The optimal position size
        """
        # Calculate the maximum position size based on the risk tolerance and risk level
        max_position_size = self.account_size * self.risk_tolerance / risk_level

        # Calculate the pip value (assuming 1 pip = 0.0001)
        pip_value = stock_price * 0.0001

        # Calculate the stop loss distance (in pips)
        stop_loss_distance = (stock_price - stop_loss) / pip_value

        # Calculate the take profit distance (in pips)
        take_profit_distance = (take_profit - stock_price) / pip_value

        # Calculate the risk-reward ratio
        risk_reward_ratio = take_profit_distance / stop_loss_distance

        # Calculate the position size based on the risk-reward ratio and maximum position size
        position_size = math.floor(max_position_size * risk_reward_ratio / stock_price)

        # Apply leverage to the position size
        position_size *= self.leverage

        return position_size

    def calculate_position_value(self, position_size, stock_price):
        """
        Calculate the value of the position.

        :param position_size: The size of the position
        :param stock_price: The current price of the stock
        :return: The value of the position
        """
        return position_size * stock_price

    def calculate_margin_requirement(self, position_size, stock_price, leverage):
        """
        Calculate the margin requirement for the position.

        :param position_size: The size of the position
        :param stock_price: The current price of the stock
        :param leverage: The leverage to apply to the margin requirement
        :return: The margin requirement
        """
        return position_size * stock_price / leverage