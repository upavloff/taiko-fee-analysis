"""
Advanced Fee Mechanism Variants

This module implements enhanced versions of the base fee mechanism with
improved control algorithms, multi-timescale dynamics, and more sophisticated
demand modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from fee_mechanism_simulator import TaikoFeeSimulator, SimulationParams, FeeVault


@dataclass
class AdvancedSimulationParams(SimulationParams):
    """Extended parameters for advanced mechanisms."""

    # Multi-timescale control
    fast_horizon: int = 24          # Fast response horizon (steps)
    slow_horizon: int = 288         # Slow response horizon (steps)
    fast_weight: float = 0.3        # Weight on fast correction term
    slow_weight: float = 0.7        # Weight on slow correction term

    # Dynamic target parameters
    dynamic_target: bool = False    # Enable dynamic target adjustment
    target_volatility_window: int = 144  # Window for volatility calculation
    target_adjustment_factor: float = 0.1  # Speed of target adjustment

    # Enhanced elasticity model
    elasticity_saturation: float = 2.0  # Saturation point for fee response
    elasticity_recovery_rate: float = 0.05  # Rate of demand recovery

    # Predictive components
    use_predictive_l1: bool = False  # Use predictive L1 cost estimation
    prediction_horizon: int = 12     # Prediction horizon (steps)

    # Circuit breakers and caps
    dynamic_fee_cap: bool = False    # Enable dynamic fee caps
    cap_multiplier: float = 5.0      # Base cap as multiple of recent average
    emergency_threshold: float = 0.1  # Emergency threshold (as fraction of target)


class DemandModel:
    """Enhanced demand model with memory and saturation effects."""

    def __init__(self,
                 base_demand: float,
                 elasticity: float,
                 saturation_point: float = 2.0,
                 recovery_rate: float = 0.05):
        """
        Args:
            base_demand: Base transaction demand
            elasticity: Fee elasticity parameter
            saturation_point: Fee level where elasticity saturates
            recovery_rate: Rate at which demand recovers from high fees
        """
        self.base_demand = base_demand
        self.elasticity = elasticity
        self.saturation_point = saturation_point
        self.recovery_rate = recovery_rate

        # State variables
        self.demand_memory = base_demand
        self.recent_fees = []

    def calculate_volume(self, current_fee: float, noise_std: float = 0.1) -> float:
        """Calculate transaction volume with memory and saturation effects."""

        # Update fee history (keep last 20 periods)
        self.recent_fees.append(current_fee)
        if len(self.recent_fees) > 20:
            self.recent_fees.pop(0)

        # Calculate average recent fee for memory effect
        avg_recent_fee = np.mean(self.recent_fees[-5:]) if len(self.recent_fees) >= 5 else current_fee

        # Saturating elasticity function
        if current_fee <= self.saturation_point:
            fee_factor = np.exp(-self.elasticity * current_fee)
        else:
            # Linear decrease beyond saturation point (less elastic)
            saturation_factor = np.exp(-self.elasticity * self.saturation_point)
            excess_fee = current_fee - self.saturation_point
            fee_factor = saturation_factor * (1 - 0.1 * excess_fee)  # 10% per unit beyond saturation

        fee_factor = max(fee_factor, 0.01)  # Minimum 1% of base demand

        # Memory effect: gradually adjust towards new demand level
        target_demand = self.base_demand * fee_factor
        self.demand_memory = (1 - self.recovery_rate) * self.demand_memory + self.recovery_rate * target_demand

        # Add noise
        noise_factor = np.random.normal(1.0, noise_std)
        return max(self.demand_memory * noise_factor, 0)

    def reset(self):
        """Reset demand model state."""
        self.demand_memory = self.base_demand
        self.recent_fees = []


class DynamicTargetManager:
    """Manages dynamic target balance based on L1 volatility."""

    def __init__(self,
                 base_target: float,
                 volatility_window: int = 144,
                 adjustment_factor: float = 0.1):
        """
        Args:
            base_target: Base target balance
            volatility_window: Window for calculating L1 volatility
            adjustment_factor: Speed of target adjustment
        """
        self.base_target = base_target
        self.volatility_window = volatility_window
        self.adjustment_factor = adjustment_factor

        self.current_target = base_target
        self.l1_cost_history = []

    def update_target(self, l1_cost: float) -> float:
        """Update target balance based on recent L1 volatility."""
        self.l1_cost_history.append(l1_cost)

        if len(self.l1_cost_history) > self.volatility_window:
            self.l1_cost_history.pop(0)

        if len(self.l1_cost_history) >= 10:  # Need minimum history
            # Calculate rolling volatility
            costs = np.array(self.l1_cost_history)
            volatility = np.std(costs[-min(self.volatility_window, len(costs)):])

            # Adjust target: higher volatility -> higher target
            volatility_multiplier = 1 + volatility * 10  # Scale factor
            new_target = self.base_target * volatility_multiplier

            # Smooth adjustment
            self.current_target = (1 - self.adjustment_factor) * self.current_target + \
                                  self.adjustment_factor * new_target

        return self.current_target

    def get_target(self) -> float:
        """Get current target balance."""
        return self.current_target

    def reset(self):
        """Reset target manager state."""
        self.current_target = self.base_target
        self.l1_cost_history = []


class L1CostPredictor:
    """Predictive model for L1 costs."""

    def __init__(self, prediction_horizon: int = 12):
        """
        Args:
            prediction_horizon: Number of steps to predict ahead
        """
        self.prediction_horizon = prediction_horizon
        self.cost_history = []

    def update_history(self, l1_cost: float):
        """Update cost history."""
        self.cost_history.append(l1_cost)
        # Keep reasonable history length
        if len(self.cost_history) > 100:
            self.cost_history.pop(0)

    def predict_cost(self) -> float:
        """Predict future L1 cost."""
        if len(self.cost_history) < 5:
            return self.cost_history[-1] if self.cost_history else 0

        costs = np.array(self.cost_history)

        # Simple trend-based prediction
        if len(costs) >= 10:
            # Linear trend over last 10 observations
            x = np.arange(len(costs[-10:]))
            trend_slope = np.polyfit(x, costs[-10:], 1)[0]
            predicted_cost = costs[-1] + trend_slope * self.prediction_horizon
        else:
            # Use recent average
            predicted_cost = np.mean(costs[-5:])

        return max(predicted_cost, 0.001)  # Ensure positive

    def reset(self):
        """Reset predictor state."""
        self.cost_history = []


class AdvancedFeeVault(FeeVault):
    """Enhanced fee vault with multi-timescale tracking."""

    def __init__(self, initial_balance: float, target_balance: float):
        super().__init__(initial_balance, target_balance)

        # Additional state tracking
        self.fast_deficit_ewma = 0
        self.slow_deficit_ewma = 0

    def update_deficit_tracking(self, fast_alpha: float = 0.1, slow_alpha: float = 0.01):
        """Update exponential weighted moving averages of deficit."""
        current_deficit = self.deficit

        # Fast EWMA (responds quickly to recent changes)
        self.fast_deficit_ewma = (1 - fast_alpha) * self.fast_deficit_ewma + fast_alpha * current_deficit

        # Slow EWMA (tracks longer-term trends)
        self.slow_deficit_ewma = (1 - slow_alpha) * self.slow_deficit_ewma + slow_alpha * current_deficit

    def get_fast_deficit(self) -> float:
        """Get fast-responding deficit measure."""
        return self.fast_deficit_ewma

    def get_slow_deficit(self) -> float:
        """Get slow-responding deficit measure."""
        return self.slow_deficit_ewma


class MultiTimescaleSimulator(TaikoFeeSimulator):
    """Enhanced simulator with multi-timescale control."""

    def __init__(self, params: AdvancedSimulationParams, l1_model):
        # Initialize base class with standard params
        base_params = SimulationParams(**params.__dict__)
        super().__init__(base_params, l1_model)

        self.advanced_params = params

        # Replace vault with advanced version
        self.vault = AdvancedFeeVault(params.target_balance, params.target_balance)

        # Initialize advanced components
        self.demand_model = DemandModel(
            params.base_demand,
            params.fee_elasticity,
            params.elasticity_saturation,
            params.elasticity_recovery_rate
        )

        if params.dynamic_target:
            self.target_manager = DynamicTargetManager(
                params.target_balance,
                params.target_volatility_window,
                params.target_adjustment_factor
            )
        else:
            self.target_manager = None

        if params.use_predictive_l1:
            self.l1_predictor = L1CostPredictor(params.prediction_horizon)
        else:
            self.l1_predictor = None

        # Additional state
        self.recent_fees = []
        self.fee_cap_history = []

    def calculate_estimated_fee(self, l1_cost_estimate: float) -> float:
        """Enhanced fee calculation with multi-timescale control."""

        # Update vault deficit tracking
        self.vault.update_deficit_tracking()

        # L1 component (using prediction if enabled)
        if self.l1_predictor and self.advanced_params.use_predictive_l1:
            predicted_l1_cost = self.l1_predictor.predict_cost()
            l1_component = self.params.mu * predicted_l1_cost
        else:
            l1_component = self.params.mu * l1_cost_estimate

        # Multi-timescale deficit correction
        if hasattr(self.vault, 'get_fast_deficit') and hasattr(self.vault, 'get_slow_deficit'):
            fast_deficit = self.vault.get_fast_deficit()
            slow_deficit = self.vault.get_slow_deficit()

            fast_correction = self.advanced_params.fast_weight * fast_deficit / self.advanced_params.fast_horizon
            slow_correction = self.advanced_params.slow_weight * slow_deficit / self.advanced_params.slow_horizon

            deficit_component = self.params.nu * (fast_correction + slow_correction)
        else:
            # Fallback to standard deficit correction
            deficit = self.vault.deficit
            deficit_component = self.params.nu * deficit / self.params.H

        estimated_fee = l1_component + deficit_component

        # Dynamic fee cap
        if self.advanced_params.dynamic_fee_cap:
            dynamic_cap = self.calculate_dynamic_fee_cap()
            estimated_fee = min(estimated_fee, dynamic_cap)
        elif self.params.fee_cap is not None:
            estimated_fee = min(estimated_fee, self.params.fee_cap)

        # Emergency circuit breaker
        if self.vault.balance < self.advanced_params.emergency_threshold * self.vault.target:
            emergency_multiplier = 2.0  # Double fees in emergency
            estimated_fee *= emergency_multiplier

        return max(estimated_fee, 0)

    def calculate_dynamic_fee_cap(self) -> float:
        """Calculate dynamic fee cap based on recent fee history."""
        if len(self.recent_fees) < 10:
            return float('inf')  # No cap if insufficient history

        recent_avg = np.mean(self.recent_fees[-20:]) if len(self.recent_fees) >= 20 else np.mean(self.recent_fees)
        dynamic_cap = recent_avg * self.advanced_params.cap_multiplier

        self.fee_cap_history.append(dynamic_cap)
        return dynamic_cap

    def calculate_transaction_volume(self, estimated_fee: float) -> float:
        """Enhanced volume calculation using advanced demand model."""
        return self.demand_model.calculate_volume(estimated_fee)

    def step(self, l1_basefee: float):
        """Enhanced simulation step."""
        # Update predictive components
        if self.l1_predictor:
            l1_cost_estimate = self.estimate_l1_cost_per_tx(l1_basefee)
            self.l1_predictor.update_history(l1_cost_estimate)

        # Update dynamic target
        if self.target_manager:
            l1_cost_estimate = self.estimate_l1_cost_per_tx(l1_basefee)
            new_target = self.target_manager.update_target(l1_cost_estimate)
            self.vault.target = new_target

        # Run standard step
        super().step(l1_basefee)

        # Track fee history for dynamic caps
        self.recent_fees.append(self.history['estimated_fee'][-1])
        if len(self.recent_fees) > 100:
            self.recent_fees.pop(0)

    def reset_state(self):
        """Reset enhanced simulation state."""
        super().reset_state()

        # Reset advanced components
        self.demand_model.reset()
        if self.target_manager:
            self.target_manager.reset()
        if self.l1_predictor:
            self.l1_predictor.reset()

        # Reset additional state
        self.recent_fees = []
        self.fee_cap_history = []

        # Replace vault with advanced version
        self.vault = AdvancedFeeVault(self.params.target_balance, self.params.target_balance)


class OptimalControlBenchmark:
    """Theoretical optimal control benchmark using dynamic programming."""

    def __init__(self, params: SimulationParams):
        self.params = params

    def solve_optimal_policy(self, l1_sequence: np.ndarray,
                           volume_sequence: np.ndarray) -> np.ndarray:
        """
        Solve for optimal fee policy using perfect foresight.

        This provides a theoretical upper bound on mechanism performance.
        """
        T = len(l1_sequence)

        # State: vault balance relative to target
        # Control: fee level

        # Simplified optimal policy: set fees to exactly cover expected L1 costs
        # plus small buffer for uncertainty

        optimal_fees = np.zeros(T)
        vault_balance = self.params.target_balance

        for t in range(T):
            # Estimate future L1 costs over planning horizon
            future_horizon = min(self.params.H, T - t)
            future_l1_costs = l1_sequence[t:t+future_horizon]

            # Expected future cost per transaction
            expected_cost = np.mean(future_l1_costs) * self.params.gas_per_batch / self.params.txs_per_batch / 1e18

            # Add buffer based on volatility
            cost_volatility = np.std(future_l1_costs) if len(future_l1_costs) > 1 else 0
            volatility_buffer = cost_volatility * self.params.gas_per_batch / self.params.txs_per_batch / 1e18

            # Deficit correction
            deficit = self.params.target_balance - vault_balance
            deficit_correction = deficit / future_horizon

            optimal_fee = expected_cost + volatility_buffer + deficit_correction / volume_sequence[t]
            optimal_fees[t] = max(optimal_fee, 0)

            # Update vault balance
            vault_balance += optimal_fees[t] * volume_sequence[t]

            # Pay L1 costs (simplified)
            if t % int(1 / self.params.batch_frequency) == 0:
                batch_cost = l1_sequence[t] * self.params.gas_per_batch / 1e18
                vault_balance -= batch_cost

        return optimal_fees

    def calculate_optimal_metrics(self, l1_sequence: np.ndarray,
                                volume_sequence: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for optimal policy."""
        optimal_fees = self.solve_optimal_policy(l1_sequence, volume_sequence)

        return {
            'optimal_avg_fee': np.mean(optimal_fees),
            'optimal_fee_std': np.std(optimal_fees),
            'optimal_fee_cv': np.std(optimal_fees) / np.mean(optimal_fees) if np.mean(optimal_fees) > 0 else 0
        }