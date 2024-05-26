import asyncio
from zksync import Web3, ZkSync

class GasEstimator:
    def __init__(self, zk_sync_url: str):
        self.zk_sync = ZkSync(Web3(zk_sync_url))

    async def estimate_gas(self, tx: dict):
        # Estimate the gas cost for the transaction
        gas_estimate = self.zk_sync.estimate_gas(tx)
        return gas_estimate