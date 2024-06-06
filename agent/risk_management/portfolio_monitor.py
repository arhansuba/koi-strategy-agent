
class PortfolioMonitor:
    def __init__(self, portfolio):
        """
        Initialize the portfolio monitor.

        :param portfolio: The portfolio to monitor
        """
        self.portfolio = portfolio
        self.minimum_value_threshold = 0.5  # adjust this value based on your requirements
        self.maximum_risk_threshold = 0.8  # adjust this value based on your requirements

    def monitor_portfolio(self):
        """
        Monitor the portfolio for risks and generate a detailed report.

        :return: A detailed report of the portfolio's health
        """
        report = ""

        # Calculate the current value of the portfolio
        portfolio_value = self.calculate_portfolio_value()
        report += f"Portfolio value: {portfolio_value:.2f}\n"

        # Calculate the current risk of the portfolio
        portfolio_risk = self.calculate_portfolio_risk()
        report += f"Portfolio risk: {portfolio_risk:.2f}\n"

        # Check if the portfolio value is below the minimum threshold
        if portfolio_value < self.minimum_value_threshold:
            report += "Warning: Portfolio value is below the minimum threshold.\n"

        # Check if the portfolio risk is above the maximum threshold
        if portfolio_risk > self.maximum_risk_threshold:
            report += "Warning: Portfolio risk is above the maximum threshold.\n"

        # If there are no issues, report that the portfolio is healthy
        if not report.strip().endswith("Warning:"):
            report += "Portfolio is healthy.\n"

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

    def get_portfolio_positions(self):
        """
        Get a list of positions in the portfolio.

        :return: A list of positions in the portfolio
        """
        return self.portfolio.positions

    def get_position_value(self, position):
        """
        Get the value of a specific position.

        :param position: The position to get the value for
        :return: The value of the position
        """
        return position.value

    def get_position_risk(self, position):
        """
        Get the risk of a specific position.

        :param position: The position to get the risk for
        :return: The risk of the position
        """
        return position.risk