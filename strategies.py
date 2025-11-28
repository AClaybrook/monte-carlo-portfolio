"""
strategies.py
Defines how future capital (DCA) is allocated based on market conditions.
"""
import numpy as np

class AllocationStrategy:
    def get_allocation(self, current_prices, current_drawdowns, base_allocations):
        """
        Returns the weight distribution for the *new* contribution.
        Returns shape: (Batch_Size, Num_Assets)
        """
        raise NotImplementedError

class StaticAllocationStrategy(AllocationStrategy):
    """
    Standard DCA: Always allocate according to the fixed portfolio weights.
    """
    def get_allocation(self, current_prices, current_drawdowns, base_allocations):
        # base_allocations is (Num_Assets,)
        # We need to broadcast it to (Batch_Size, Num_Assets)
        batch_size = current_prices.shape[0]
        return np.tile(base_allocations, (batch_size, 1))

class BuyTheDipStrategy(AllocationStrategy):
    """
    Allocates new capital heavily into assets that are in a drawdown.
    """
    def __init__(self, target_ticker_index, threshold=0.05, aggressive_weight=0.8):
        self.target_idx = target_ticker_index # Index of asset to watch (e.g., TQQQ)
        self.threshold = threshold            # Drop required (e.g., 5%)
        self.aggressive_weight = aggressive_weight # How much of the contribution goes here if dipped

    def get_allocation(self, current_prices, current_drawdowns, base_allocations):
        batch_size, n_assets = current_prices.shape

        # Default allocation (broadcasted)
        weights = np.tile(base_allocations, (batch_size, 1))

        # Check the specific asset for drawdown > threshold
        # current_drawdowns is (Batch, Assets)
        target_dd = current_drawdowns[:, self.target_idx]

        # Mask: Which simulations are currently in the dip?
        dip_mask = target_dd < -self.threshold

        if np.any(dip_mask):
            # For the rows where dip_mask is True, change allocation

            # 1. Zero out the old weights for the dipped rows
            # (We will reconstruct them to ensure sum = 1.0)
            weights[dip_mask] = 0

            # 2. Assign aggressive weight to the target asset
            weights[dip_mask, self.target_idx] = self.aggressive_weight

            # 3. Distribute the remaining (1 - aggressive) proportionally among others
            remaining_weight = 1.0 - self.aggressive_weight

            # Calculate sum of base weights excluding the target
            base_others_sum = np.sum(base_allocations) - base_allocations[self.target_idx]

            if base_others_sum > 0:
                for i in range(n_assets):
                    if i == self.target_idx: continue
                    # scaled weight
                    w = (base_allocations[i] / base_others_sum) * remaining_weight
                    weights[dip_mask, i] = w
            else:
                # If target was 100% of base, just put 100% in target (no others exist)
                weights[dip_mask, self.target_idx] = 1.0

        return weights