/**
 * JavaScript Implementation of the Enhanced Metrics Framework
 * Based on the Python implementation in src/analysis/enhanced_metrics.py
 */

/**
 * Enhanced Metrics Calculator
 * Implements the rigorous mathematical framework for evaluating Taiko fee mechanism performance
 */
class EnhancedMetricsCalculator {
    constructor(targetBalance = 1.0, taikoBlockTime = 2.0, ethBlockTime = 12.0) {
        this.targetBalance = targetBalance;
        this.taikoBlockTime = taikoBlockTime;
        this.ethBlockTime = ethBlockTime;

        // Risk thresholds
        this.insolvencyThreshold = 0.1 * targetBalance;  // 10% deficit is insolvency risk
        this.extremeFeeThreshold = 0.01;                 // 0.01 ETH fee is extreme
    }

    /**
     * Calculate user experience metrics
     * These metrics directly impact user adoption and protocol competitiveness
     */
    calculateUserExperienceMetrics(simulationResults) {
        const fees = simulationResults.estimated_fee;
        const timeSteps = fees.length;

        // Fee Affordability: Heavily penalize high fees using log scale
        // Formula: -log(1 + avg_fee_eth × 1000)
        const avgFee = this.mean(fees);
        const feeAffordability = avgFee > 0 ? -Math.log(1 + avgFee * 1000) : 1.0;

        // Fee Predictability: Reward stable fees
        // Formula: 1 - coefficient_of_variation
        const feeCv = this.std(fees) / this.mean(fees);
        const feePredictability = Math.max(0, 1 - feeCv);

        // Fee Stability: Lower relative volatility = better UX
        // Formula: 1 - coefficient_of_variation (same as predictability but conceptually different)
        const feeStability = feePredictability;

        // Fee Predictability over 1-hour windows (30 steps at 2s per step)
        const oneHourSteps = Math.floor(3600 / this.taikoBlockTime);  // 1800 steps
        const feePredictability1h = this.calculateWindowPredictability(fees, oneHourSteps);

        // Fee Predictability over 6-hour windows (180 steps)
        const sixHourSteps = Math.floor(6 * 3600 / this.taikoBlockTime);  // 10800 steps
        const feePredictability6h = this.calculateWindowPredictability(fees, sixHourSteps);

        // Fee Rate of Change (95th percentile)
        const feeRateOfChangeP95 = this.calculateFeeRateOfChangeP95(fees);

        // User Cost Burden (simplified calculation)
        const totalUserFees = fees.reduce((sum, fee, i) => {
            const txVolume = simulationResults.transaction_volume[i] || 0;
            return sum + fee * txVolume;
        }, 0);
        const userCostBurden = totalUserFees / (timeSteps * avgFee * 100); // Normalized

        return {
            fee_affordability_score: feeAffordability,
            fee_predictability_score: feePredictability,
            fee_stability_score: feeStability,
            fee_predictability_1h: feePredictability1h,
            fee_predictability_6h: feePredictability6h,
            fee_rate_of_change_p95: feeRateOfChangeP95,
            user_cost_burden: userCostBurden
        };
    }

    /**
     * Calculate protocol stability metrics
     * Critical for protocol security and robustness
     */
    calculateProtocolStabilityMetrics(simulationResults) {
        const vaultBalances = simulationResults.vault_balance;
        const vaultDeficits = simulationResults.vault_deficit;
        const timeSteps = vaultBalances.length;

        // Insolvency Probability: P(vault_balance < critical_threshold)
        const insolvencyEvents = vaultBalances.filter(balance => balance < this.insolvencyThreshold).length;
        const insolvencyProbability = insolvencyEvents / timeSteps;

        // Deficit Weighted Duration: ∑(deficit_magnitude × duration)² / total_time
        const deficitWeightedDuration = this.calculateDeficitWeightedDuration(vaultDeficits, timeSteps);

        // Vault Stress Resilience: Average recovery rate during deficit periods
        const stressResilience = this.calculateStressResilience(vaultDeficits, vaultBalances);

        // Max Continuous Underfunding
        const maxContinuousUnderfunding = this.calculateMaxContinuousUnderfunding(vaultDeficits);

        // Vault Robustness Score: 1 - P(deficit > 0.5×target_balance)
        const severeDeficitEvents = vaultDeficits.filter(deficit => deficit > 0.5 * this.targetBalance).length;
        const vaultRobustnessScore = 1 - (severeDeficitEvents / timeSteps);

        // Crisis Resilience Score: 1 - max_deficit_duration / simulation_length
        const maxDeficit = Math.max(...vaultDeficits);
        const crisisResilienceScore = Math.max(0, 1 - maxDeficit / this.targetBalance);

        return {
            insolvency_probability: insolvencyProbability,
            deficit_weighted_duration: deficitWeightedDuration,
            vault_stress_resilience: stressResilience,
            max_continuous_underfunding: maxContinuousUnderfunding,
            vault_robustness_score: vaultRobustnessScore,
            crisis_resilience_score: crisisResilienceScore
        };
    }

    /**
     * Calculate economic efficiency metrics
     */
    calculateEconomicEfficiencyMetrics(simulationResults) {
        const vaultBalances = simulationResults.vault_balance;
        const feeCollected = simulationResults.fee_collected;
        const l1CostPaid = simulationResults.l1_cost_paid;

        // Capital Efficiency: Average vault utilization
        const avgBalance = this.mean(vaultBalances);
        const capitalEfficiency = Math.min(avgBalance / this.targetBalance, 2.0); // Cap at 200%

        // Cost Recovery Efficiency
        const totalFeesCollected = this.sum(feeCollected);
        const totalL1Costs = this.sum(l1CostPaid);
        const costRecoveryEfficiency = totalL1Costs > 0 ? totalFeesCollected / totalL1Costs : 1.0;

        return {
            capital_efficiency_score: capitalEfficiency,
            cost_recovery_efficiency: costRecoveryEfficiency
        };
    }

    /**
     * Calculate composite scores for multi-objective optimization
     */
    calculateCompositeScores(allMetrics) {
        // UX Score weights (focused on what users actually care about)
        const uxScore =
            0.4 * allMetrics.fee_affordability_score +
            0.3 * allMetrics.fee_stability_score +
            0.15 * allMetrics.fee_predictability_1h +
            0.15 * allMetrics.fee_predictability_6h;

        // Safety Score weights (protocol solvency risks)
        const safetyScore =
            0.4 * (1 - allMetrics.insolvency_probability) +
            0.3 * (1 - allMetrics.deficit_weighted_duration) +
            0.3 * allMetrics.vault_stress_resilience;

        // Efficiency Score
        const efficiencyScore =
            0.6 * allMetrics.capital_efficiency_score +
            0.4 * allMetrics.cost_recovery_efficiency;

        // Overall Score (weighted combination)
        const overallScore =
            0.5 * uxScore +
            0.35 * safetyScore +
            0.15 * efficiencyScore;

        return {
            user_experience_composite: uxScore,
            protocol_stability_composite: safetyScore,
            economic_efficiency_composite: efficiencyScore,
            overall_optimization_score: overallScore,
            ux_score: uxScore,  // Alias for consistency
            safety_score: safetyScore  // Alias for consistency
        };
    }

    /**
     * Calculate all metrics for a simulation result
     */
    calculateAllMetrics(simulationResults, params = {}) {
        const uxMetrics = this.calculateUserExperienceMetrics(simulationResults);
        const stabilityMetrics = this.calculateProtocolStabilityMetrics(simulationResults);
        const efficiencyMetrics = this.calculateEconomicEfficiencyMetrics(simulationResults);

        // Combine all metrics
        const allMetrics = {
            ...uxMetrics,
            ...stabilityMetrics,
            ...efficiencyMetrics
        };

        // Calculate composite scores
        const compositeScores = this.calculateCompositeScores(allMetrics);

        return {
            ...allMetrics,
            ...compositeScores
        };
    }

    // === Helper Methods ===

    /**
     * Calculate mean of array
     */
    mean(arr) {
        return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    }

    /**
     * Calculate standard deviation of array
     */
    std(arr) {
        const mean = this.mean(arr);
        const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
        return Math.sqrt(variance);
    }

    /**
     * Calculate sum of array
     */
    sum(arr) {
        return arr.reduce((a, b) => a + b, 0);
    }

    /**
     * Calculate percentile
     */
    percentile(arr, p) {
        const sorted = [...arr].sort((a, b) => a - b);
        const index = Math.floor(sorted.length * p / 100);
        return sorted[Math.min(index, sorted.length - 1)];
    }

    /**
     * Calculate window-based predictability
     */
    calculateWindowPredictability(fees, windowSize) {
        if (fees.length < windowSize) return 1.0;

        let totalPredictability = 0;
        let windowCount = 0;

        for (let i = 0; i <= fees.length - windowSize; i++) {
            const window = fees.slice(i, i + windowSize);
            const windowCv = this.std(window) / this.mean(window);
            totalPredictability += Math.max(0, 1 - windowCv);
            windowCount++;
        }

        return windowCount > 0 ? totalPredictability / windowCount : 1.0;
    }

    /**
     * Calculate 95th percentile of fee rate of change
     */
    calculateFeeRateOfChangeP95(fees) {
        if (fees.length < 2) return 0;

        const rateChanges = [];
        for (let i = 1; i < fees.length; i++) {
            if (fees[i-1] > 0) {
                const change = Math.abs(fees[i] - fees[i-1]) / fees[i-1];
                rateChanges.push(change);
            }
        }

        return rateChanges.length > 0 ? this.percentile(rateChanges, 95) : 0;
    }

    /**
     * Calculate deficit weighted duration
     */
    calculateDeficitWeightedDuration(deficits, totalTime) {
        let weightedSum = 0;
        let currentDeficitRun = 0;

        for (const deficit of deficits) {
            if (deficit > 0) {
                currentDeficitRun++;
                weightedSum += Math.pow(deficit * currentDeficitRun, 2);
            } else {
                currentDeficitRun = 0;
            }
        }

        return weightedSum / totalTime;
    }

    /**
     * Calculate stress resilience
     */
    calculateStressResilience(deficits, balances) {
        let recoveryEvents = 0;
        let totalRecoveryRate = 0;

        for (let i = 1; i < deficits.length; i++) {
            if (deficits[i-1] > 0 && deficits[i] < deficits[i-1]) {
                const recoveryRate = (deficits[i-1] - deficits[i]) / Math.max(deficits[i-1], 0.001);
                totalRecoveryRate += recoveryRate;
                recoveryEvents++;
            }
        }

        return recoveryEvents > 0 ? totalRecoveryRate / recoveryEvents : 0;
    }

    /**
     * Calculate maximum continuous underfunding
     */
    calculateMaxContinuousUnderfunding(deficits) {
        let maxContinuous = 0;
        let currentContinuous = 0;

        for (const deficit of deficits) {
            if (deficit > 0) {
                currentContinuous++;
                maxContinuous = Math.max(maxContinuous, currentContinuous);
            } else {
                currentContinuous = 0;
            }
        }

        return maxContinuous;
    }
}

/**
 * Multi-Scenario Evaluator
 * Evaluates parameter sets across multiple historical scenarios
 */
class MultiScenarioEvaluator {
    constructor(historicalDataLoader, metricsCalculator) {
        this.historicalDataLoader = historicalDataLoader;
        this.metricsCalculator = metricsCalculator;
    }

    /**
     * Evaluate a parameter set across all historical scenarios
     */
    async evaluateParameterSet(params) {
        if (!this.historicalDataLoader.loaded) {
            throw new Error('Historical data not loaded. Call historicalDataLoader.loadAllDatasets() first.');
        }

        const results = {};
        const datasetNames = this.historicalDataLoader.getDatasetNames();

        for (const datasetName of datasetNames) {
            const dataset = this.historicalDataLoader.getDataset(datasetName);

            // Create simulation parameters
            const simulationParams = new ImprovedSimulationParams({
                mu: params.mu,
                nu: params.nu,
                H: params.H,
                total_steps: Math.min(dataset.length, 1800) // Limit to reasonable size
            });

            // Run simulation
            const simulator = new ImprovedResearchTaikoFeeSimulator(simulationParams, dataset);
            const simulationResults = simulator.runSimulation();

            // Calculate metrics
            const metrics = this.metricsCalculator.calculateAllMetrics(simulationResults, params);

            results[datasetName] = {
                simulationResults: simulationResults,
                metrics: metrics,
                summary: simulator.getResultsSummary()
            };
        }

        // Calculate aggregated metrics across all scenarios
        const aggregatedMetrics = this.aggregateMetricsAcrossScenarios(results);

        return {
            individual_scenarios: results,
            aggregated_metrics: aggregatedMetrics,
            parameters: params
        };
    }

    /**
     * Aggregate metrics across all scenarios
     */
    aggregateMetricsAcrossScenarios(scenarioResults) {
        const datasetNames = Object.keys(scenarioResults);
        const aggregated = {};

        // Get all metric names from first scenario
        const firstScenario = scenarioResults[datasetNames[0]];
        const metricNames = Object.keys(firstScenario.metrics);

        // Calculate mean and worst-case for each metric
        for (const metricName of metricNames) {
            const values = datasetNames.map(name => scenarioResults[name].metrics[metricName]);

            aggregated[`${metricName}_mean`] = this.metricsCalculator.mean(values);
            aggregated[`${metricName}_min`] = Math.min(...values);
            aggregated[`${metricName}_max`] = Math.max(...values);
            aggregated[`${metricName}_std`] = this.metricsCalculator.std(values);
        }

        // Calculate robust composite scores (using worst-case scenarios)
        aggregated.robust_ux_score = Math.min(
            ...datasetNames.map(name => scenarioResults[name].metrics.ux_score)
        );

        aggregated.robust_safety_score = Math.min(
            ...datasetNames.map(name => scenarioResults[name].metrics.safety_score)
        );

        aggregated.robust_overall_score =
            0.5 * aggregated.robust_ux_score +
            0.5 * aggregated.robust_safety_score;

        return aggregated;
    }
}

// Export for global use
window.EnhancedMetricsCalculator = EnhancedMetricsCalculator;
window.MultiScenarioEvaluator = MultiScenarioEvaluator;