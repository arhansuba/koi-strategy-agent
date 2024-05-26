import asyncio
from zksync import Web3, ZkSync

class TransactionExecutor:
    def __init__(self, zk_sync_url: str):
        self.zk_sync = ZkSync(Web3(zk_sync_url))

    async def execute_transaction(self, tx: dict):
        # Execute the transaction on zkSync
        tx_hash = self.zk_sync.send_transaction(tx)
        return tx_hash