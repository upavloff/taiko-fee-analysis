"""
Taiko Fee Mechanism Simulator

This module implements a comprehensive simulation framework for analyzing
Taiko's proposed fee mechanism design. It includes models for L1 dynamics,
vault management, and user behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class SimulationParams:
    """Parameters for the fee mechanism simulation."""

    # Mechanism parameters
    mu: float = 0.5           # Weight on L1 cost estimate [0,1]
    nu: float = 0.3           # Weight on deficit correction [0,1]
    H: int = 144              # Deficit correction horizon (in time steps)
    target_balance: float = 1000  # Target vault balance

    # Economic parameters
    base_demand: float = 100      # Base transaction demand per time step
    fee_elasticity: float = 0.2   # Demand elasticity to fees

    # L1 cost parameters
    base_l1_cost: float = 0.1     # Base L1 cost per transaction
    gas_per_batch: int = 200000   # Gas cost per L1 batch
    txs_per_batch: int = 100      # Transactions per batch
    batch_frequency: float = 0.1   # Batches per time step (0.1 = batch every 10 steps)

    # Simulation parameters
    total_steps: int = 10000      # Total simulation steps
    time_step_seconds: int = 2    # Seconds per time step (Taiko L2 block time)

    # Risk management
    fee_cap: Optional[float] = None  # Maximum fee per transaction
    min_vault_balance: float = 0     # Minimum vault balance before emergency


class L1DynamicsModel(ABC):
    """Abstract base class for L1 basefee dynamics models."""

    @abstractmethod
    def generate_sequence(self, steps: int, initial_basefee: float = 20e9) -> np.ndarray:
        """Generate a sequence of L1 basefees."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this model."""
        pass


class GeometricBrownianMotion(L1DynamicsModel):
    """L1 basefee following geometric Brownian motion."""

    def __init__(self, mu: float = 0.0, sigma: float = 0.3, dt: float = 1/300):
        """
        Args:
            mu: Drift parameter (annualized)
            sigma: Volatility parameter (annualized)
            dt: Time step in years (12 seconds = 1/300 of an hour ≈ 1/2628000 of a year)
        """
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def generate_sequence(self, steps: int, initial_basefee: float = 20e9) -> np.ndarray:
        """Generate GBM sequence for L1 basefee."""
        noise = np.random.normal(0, 1, steps)
        log_returns = (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * noise

        log_prices = np.cumsum(log_returns)
        basefees = initial_basefee * np.exp(log_prices)

        # Allow natural basefee dynamics for realistic sub-gwei periods
        # Removed artificial 1 gwei floor to match documented low-fee datasets (0.055-0.092 gwei)
        return np.maximum(basefees, 1e6)  # 0.001 gwei minimum (technical floor)

    def get_name(self) -> str:
        return f"GBM(μ={self.mu:.2f}, σ={self.sigma:.2f})"


class RegimeSwitchingModel(L1DynamicsModel):
    """L1 basefee with regime switching between low/medium/high states."""

    def __init__(self,
                 regimes: Dict[str, float] = None,
                 transition_matrix: np.ndarray = None,
                 volatilities: Dict[str, float] = None):
        """
        Args:
            regimes: Dict mapping regime names to mean basefees
            transition_matrix: Markov transition matrix between regimes
            volatilities: Dict mapping regime names to volatility parameters
        """
        if regimes is None:
            regimes = {'low': 10e9, 'medium': 30e9, 'high': 100e9}
        if volatilities is None:
            volatilities = {'low': 0.2, 'medium': 0.3, 'high': 0.5}
        if transition_matrix is None:
            # Default: sticky regimes with small switching probability
            transition_matrix = np.array([
                [0.99, 0.008, 0.002],  # low -> {low, medium, high}
                [0.01, 0.98, 0.01],    # medium -> {low, medium, high}
                [0.005, 0.02, 0.975]   # high -> {low, medium, high}
            ])

        self.regimes = regimes
        self.regime_names = list(regimes.keys())
        self.regime_means = np.array(list(regimes.values()))
        self.transition_matrix = transition_matrix
        self.volatilities = volatilities

    def generate_sequence(self, steps: int, initial_basefee: float = 20e9) -> np.ndarray:
        """Generate regime-switching sequence."""
        # Start in regime closest to initial basefee
        current_regime = np.argmin(np.abs(self.regime_means - initial_basefee))

        basefees = np.zeros(steps)
        regimes = np.zeros(steps, dtype=int)

        for t in range(steps):
            regimes[t] = current_regime
            regime_name = self.regime_names[current_regime]

            # Generate basefee in current regime
            mean_basefee = self.regime_means[current_regime]
            volatility = self.volatilities[regime_name]

            if t == 0:
                basefees[t] = initial_basefee
            else:
                # GBM within regime
                log_return = -0.5 * volatility**2 / 252 + volatility * np.sqrt(1/252) * np.random.normal()
                basefees[t] = basefees[t-1] * np.exp(log_return)

                # Mean reversion toward regime center
                reversion_speed = 0.01
                basefees[t] = basefees[t] * (1 - reversion_speed) + mean_basefee * reversion_speed

            # Regime switching
            next_regime = np.random.choice(len(self.regime_names),
                                         p=self.transition_matrix[current_regime])
            current_regime = next_regime

        return np.maximum(basefees, 1e9)

    def get_name(self) -> str:
        return "Regime Switching"


class SpikeEventsModel(L1DynamicsModel):
    """L1 basefee with occasional spike events overlaid on baseline dynamics."""

    def __init__(self,
                 baseline_model: L1DynamicsModel,
                 spike_probability: float = 0.001,
                 spike_magnitude_range: Tuple[float, float] = (3.0, 10.0),
                 spike_duration_range: Tuple[int, int] = (5, 50)):
        """
        Args:
            baseline_model: Underlying L1 dynamics model
            spike_probability: Probability of spike starting at any time step
            spike_magnitude_range: Range of spike multipliers
            spike_duration_range: Range of spike durations in time steps
        """
        self.baseline_model = baseline_model
        self.spike_prob = spike_probability
        self.spike_mag_range = spike_magnitude_range
        self.spike_dur_range = spike_duration_range

    def generate_sequence(self, steps: int, initial_basefee: float = 20e9) -> np.ndarray:
        """Generate baseline sequence with spike events."""
        baseline = self.baseline_model.generate_sequence(steps, initial_basefee)
        basefees = baseline.copy()

        t = 0
        while t < steps:
            if np.random.random() < self.spike_prob:
                # Generate spike
                spike_magnitude = np.random.uniform(*self.spike_mag_range)
                spike_duration = np.random.randint(*self.spike_dur_range)

                # Apply spike with exponential decay
                for i in range(min(spike_duration, steps - t)):
                    decay_factor = np.exp(-i / (spike_duration / 3))  # Decay over 1/3 of duration
                    spike_multiplier = 1 + (spike_magnitude - 1) * decay_factor
                    basefees[t + i] *= spike_multiplier

                t += spike_duration
            else:
                t += 1

        return basefees

    def get_name(self) -> str:
        return f"Spikes + {self.baseline_model.get_name()}"


class FeeVault:
    """Manages the fee vault balance and fee calculations."""

    def __init__(self, initial_balance: float, target_balance: float):
        self.balance = initial_balance
        self.target = target_balance
        self.history = {'balance': [initial_balance], 'inflow': [0], 'outflow': [0]}

    @property
    def deficit(self) -> float:
        """Current deficit (target - balance). Positive means underfunded."""
        return self.target - self.balance

    def collect_fees(self, amount: float):
        """Add fee collection to vault."""
        self.balance += amount
        self.history['inflow'].append(amount)

    def pay_l1_costs(self, amount: float):
        """Pay L1 batch costs from vault."""
        self.balance -= amount
        self.history['outflow'].append(amount)

    def record_balance(self):
        """Record current balance in history."""
        self.history['balance'].append(self.balance)


class TaikoFeeSimulator:
    """Main simulator for the Taiko fee mechanism."""

    def __init__(self, params: SimulationParams, l1_model: L1DynamicsModel):
        self.params = params
        self.l1_model = l1_model
        self.vault = FeeVault(params.target_balance, params.target_balance)

        # Initialize state tracking
        self.reset_state()

    def reset_state(self):
        """Reset simulation state."""
        self.vault = FeeVault(self.params.target_balance, self.params.target_balance)
        self.time_step = 0

        # History tracking
        self.history = {
            'time_step': [],
            'l1_basefee': [],
            'estimated_l1_cost': [],
            'estimated_fee': [],
            'transaction_volume': [],
            'vault_balance': [],
            'vault_deficit': [],
            'fee_collected': [],
            'l1_cost_paid': [],
            'batch_occurred': []
        }

        # L1 cost estimation (exponential weighted moving average)
        self.l1_cost_ewma = self.params.base_l1_cost
        self.ewma_alpha = 0.1  # EWMA smoothing parameter

    def estimate_l1_cost_per_tx(self, l1_basefee: float) -> float:
        """Estimate L1 cost per transaction based on current basefee."""
        gas_cost_per_tx = self.params.gas_per_batch / self.params.txs_per_batch
        cost_per_tx = l1_basefee * gas_cost_per_tx / 1e18  # Convert from wei to ETH

        # Update EWMA estimate
        self.l1_cost_ewma = (1 - self.ewma_alpha) * self.l1_cost_ewma + self.ewma_alpha * cost_per_tx
        return self.l1_cost_ewma

    def calculate_estimated_fee(self, l1_cost_estimate: float) -> float:
        """Calculate estimated fee using the proposed mechanism."""
        # Direct L1 cost component
        l1_component = self.params.mu * l1_cost_estimate

        # Deficit correction component
        deficit = self.vault.deficit
        deficit_component = self.params.nu * deficit / self.params.H

        estimated_fee = l1_component + deficit_component

        # Apply fee cap if specified
        if self.params.fee_cap is not None:
            estimated_fee = min(estimated_fee, self.params.fee_cap)

        # Ensure non-negative fees
        return max(estimated_fee, 0)

    def calculate_transaction_volume(self, estimated_fee: float) -> float:
        """Calculate transaction volume with fee elasticity."""
        # Volume decreases exponentially with fees
        volume = self.params.base_demand * np.exp(-self.params.fee_elasticity * estimated_fee)

        # Add some noise
        noise_factor = np.random.normal(1.0, 0.1)  # 10% noise
        return max(volume * noise_factor, 0)

    def calculate_l1_batch_cost(self, l1_basefee: float) -> float:
        """Calculate cost of an L1 batch."""
        return l1_basefee * self.params.gas_per_batch / 1e18  # Convert to ETH

    def step(self, l1_basefee: float):
        """
        Execute one simulation time step with realistic lumpy cash flow timing.

        CRITICAL: This implements the timing fix that creates realistic vault economics:
        - Fee collection: Every 2s (every Taiko L2 block)
        - L1 batch cost payment: Every 12s (every 6 Taiko steps, when t % 6 === 0)

        This creates natural saw-tooth deficit patterns that match real protocol economics.
        """
        # Estimate L1 cost per transaction
        l1_cost_estimate = self.estimate_l1_cost_per_tx(l1_basefee)

        # Calculate estimated fee
        estimated_fee = self.calculate_estimated_fee(l1_cost_estimate)

        # Calculate transaction volume (with fee elasticity)
        tx_volume = self.calculate_transaction_volume(estimated_fee)

        # ALWAYS collect fees (every 2s Taiko L2 block)
        total_fees = estimated_fee * tx_volume
        self.vault.collect_fees(total_fees)

        # ONLY pay L1 costs when batch is submitted (every 12s = every 6 Taiko steps)
        # This matches the JavaScript implementation: isL1BatchStep = (t % 6 === 0)
        batch_occurred = (self.time_step % 6 == 0)
        l1_cost_paid = 0
        if batch_occurred:
            l1_cost_paid = self.calculate_l1_batch_cost(l1_basefee)
            self.vault.pay_l1_costs(l1_cost_paid)

        # Record history
        self.history['time_step'].append(self.time_step)
        self.history['l1_basefee'].append(l1_basefee)
        self.history['estimated_l1_cost'].append(l1_cost_estimate)
        self.history['estimated_fee'].append(estimated_fee)
        self.history['transaction_volume'].append(tx_volume)
        self.history['vault_balance'].append(self.vault.balance)
        self.history['vault_deficit'].append(self.vault.deficit)
        self.history['fee_collected'].append(total_fees)
        self.history['l1_cost_paid'].append(l1_cost_paid)
        self.history['batch_occurred'].append(batch_occurred)

        self.time_step += 1

    def run_simulation(self) -> pd.DataFrame:
        """Run complete simulation and return results."""
        self.reset_state()

        # Generate L1 basefee sequence
        l1_basefees = self.l1_model.generate_sequence(
            self.params.total_steps,
            initial_basefee=20e9  # Start at 20 gwei
        )

        # Run simulation steps
        for basefee in l1_basefees:
            self.step(basefee)

        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        df['time_hours'] = df['time_step'] * self.params.time_step_seconds / 3600

        return df