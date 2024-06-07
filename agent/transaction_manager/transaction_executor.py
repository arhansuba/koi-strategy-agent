# transaction_executor.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .transaction_confirmator import TransactionConfirmator

#from transaction_manager import TransactionConfirmator

class TransactionExecutor:
    def __init__(self, db_url, confirmator):
        """
        Initialize the transaction executor.

        :param db_url: The database URL
        :param confirmator: The transaction confirmator
        """
        self.db_url = db_url
        self.confirmator = confirmator
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def execute_transaction(self, transaction):
        """
        Execute a transaction.

        :param transaction: The transaction to execute
        :return: The execution result
        """
        # Confirm the transaction using the confirmator
        confirmation = self.confirmator.confirm_transaction(transaction)

        if confirmation:
            # Execute the transaction
            try:
                self.session.add(transaction)
                self.session.commit()
                return True
            except Exception as e:
                self.session.rollback()
                print(f"Error executing transaction: {e}")
                return False
        else:
            print("Transaction not confirmed")
            return False

    def execute_transactions(self, transactions):
        """
        Execute multiple transactions.

        :param transactions: The transactions to execute
        :return: The execution results
        """
        results = []
        for transaction in transactions:
            result = self.execute_transaction(transaction)
            results.append(result)
        return results

    def get_account_balance(self, account_id):
        """
        Get the balance of an account.

        :param account_id: The account ID
        :return: The account balance
        """
        query = self.session.query(Account).filter_by(id=account_id)
        account = query.first()
        return account.balance

    def update_account_balance(self, account_id, amount):
        """
        Update the balance of an account.

        :param account_id: The account ID
        :param amount: The amount to update
        """
        query = self.session.query(Account).filter_by(id=account_id)
        account = query.first()
        account.balance += amount
        self.session.commit()

class Account:
    def __init__(self, id, balance):
        self.id = id
        self.balance = balance

class Transaction:
    def __init__(self, id, account_id, amount, type):
        self.id = id
        self.account_id = account_id
        self.amount = amount
        self.type = type

if __name__ == "__main__":
    db_url = "sqlite:///transactions.db"
    confirmator = TransactionConfirmator()
    executor = TransactionExecutor(db_url, confirmator)

    # Create some sample transactions
    transactions = [
        Transaction(1, 1, 100, "deposit"),
        Transaction(2, 1, 50, "withdrawal"),
        Transaction(3, 2, 200, "deposit"),
        Transaction(4, 2, 100, "withdrawal")
    ]

    # Execute the transactions
    results = executor.execute_transactions(transactions)

    # Print the execution results
    for result in results:
        print(result)

    # Get the account balances
    account1_balance = executor.get_account_balance(1)
    account2_balance = executor.get_account_balance(2)

    print(f"Account 1 balance: {account1_balance}")
    print(f"Account 2 balance: {account2_balance}")