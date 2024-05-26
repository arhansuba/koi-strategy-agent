import asyncio
from zksync import Web3, ZkSync

class TransactionConfirmator:
    def __init__(self, zk_sync_url: str):
        self.zk_sync = ZkSync(Web3(zk_sync_url))

    async def confirm_transaction(self, tx_hash: str):
        # Wait for the transaction to be confirmed on zkSync
        while True:
            tx_receipt = self.zk_sync.get_transaction_receipt(tx_hash)
            if tx_receipt.status == "confirmed":
                break
            await asyncio.sleep(1)
        return tx_receipt