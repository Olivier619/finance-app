
import pandas as pd
import numpy as np
import pandas_ta as ta
from trading_strategies import CoupledStrategy, MACrossover, RSIStrategy

# Create mock data
dates = pd.date_range(start='2023-01-01', periods=100)
close = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
df = pd.DataFrame({'timestamp': dates, 'close': close})

# Add mock signals for individual strategies to ensure they work
# We will just verify the CoupledStrategy logic
class MockStrategy:
    def __init__(self, name, signal_val):
        self.name = name
        self.signal_val = signal_val
    
    def generate_signals(self, df):
        df['signal'] = self.signal_val
        return df

# Test 1: Consensus ALL BUY
strat1 = MockStrategy("S1", 1)
strat2 = MockStrategy("S2", 1)
coupled = CoupledStrategy([strat1, strat2])
res = coupled.generate_signals(df.copy())
print(f"Test 1 (All Buy): Signal Should be 1. Got: {res['signal'].iloc[0]}")

# Test 2: Consensus SPLIT (1 vs -1) -> 0
strat3 = MockStrategy("S3", -1)
coupled_split = CoupledStrategy([strat1, strat3])
res_split = coupled_split.generate_signals(df.copy())
print(f"Test 2 (Split): Signal Should be 0. Got: {res_split['signal'].iloc[0]}")

# Test 3: Consensus MAJORITY BUY (1, 1, -1) -> 1
coupled_maj = CoupledStrategy([strat1, strat2, strat3])
res_maj = coupled_maj.generate_signals(df.copy())
print(f"Test 3 (Majority Buy): Signal Should be 1. Got: {res_maj['signal'].iloc[0]}")
