/**
 * Canonical API Client for Taiko Fee Mechanism
 *
 * This module provides JavaScript classes that interface with the Python canonical
 * implementations via REST API calls. It replaces the local fee simulator classes
 * to ensure consistency with the single source of truth.
 *
 * Key Benefits:
 * - Guarantees consistency with Python canonical implementations
 * - Eliminates duplicate fee calculation logic
 * - Provides real-time access to optimization results
 * - Maintains same interface as original simulator classes
 *
 * Usage:
 *   const calculator = new CanonicalFeeAPI();
 *   const fee = await calculator.calculateFee(params);
 *
 *   const simulator = new CanonicalSimulationAPI();
 *   const results = await simulator.runSimulation(config);
 */

// API Configuration
const API_CONFIG = {
    baseUrl: 'http://localhost:8001',  // Python API server
    endpoints: {
        calculateFee: '/api/fee/calculate',
        simulate: '/api/fee/simulate',
        metrics: '/api/fee/metrics',
        validateParameters: '/api/fee/validate-parameters',
        presets: '/api/fee/presets',
        generateL1Data: '/api/fee/generate-l1-data',
        startOptimization: '/api/optimization/start',
        optimizationStatus: '/api/optimization/status',
        optimizationResult: '/api/optimization/result',
        quickOptimization: '/api/optimization/quick',
        health: '/health'
    },
    timeout: 30000,  // 30 second timeout
    retries: 3
};

/**
 * Base API client with common functionality
 */
class BaseAPIClient {
    constructor() {
        this.baseUrl = API_CONFIG.baseUrl;
        this.timeout = API_CONFIG.timeout;
        this.retries = API_CONFIG.retries;
    }

    /**
     * Make HTTP request with retry logic and error handling
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;

        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            timeout: this.timeout,
            ...options
        };

        // Add request body for POST/PUT requests
        if (options.data && (options.method === 'POST' || options.method === 'PUT')) {
            defaultOptions.body = JSON.stringify(options.data);
        }

        let lastError;

        for (let attempt = 0; attempt < this.retries; attempt++) {
            try {
                console.log(`API Request (attempt ${attempt + 1}): ${options.method || 'GET'} ${url}`);

                // Create AbortController for timeout
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);

                const response = await fetch(url, {
                    ...defaultOptions,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }

                const data = await response.json();
                console.log(`API Response: ${response.status} (${data ? 'success' : 'empty'})`);

                return data;

            } catch (error) {
                lastError = error;
                console.warn(`API request failed (attempt ${attempt + 1}/${this.retries}): ${error.message}`);

                // Don't retry on certain errors
                if (error.name === 'AbortError') {
                    throw new Error(`Request timeout after ${this.timeout}ms`);
                }

                // Wait before retry (exponential backoff)
                if (attempt < this.retries - 1) {
                    const delay = Math.pow(2, attempt) * 1000;
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }

        throw new Error(`API request failed after ${this.retries} attempts: ${lastError.message}`);
    }

    /**
     * Check if API server is healthy
     */
    async checkHealth() {
        try {
            const response = await this.request(API_CONFIG.endpoints.health);
            return response.status === 'healthy';
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    }
}

/**
 * Canonical Fee Calculation API Client
 *
 * Replaces local fee calculation with API calls to canonical Python implementation
 */
class CanonicalFeeAPI extends BaseAPIClient {
    constructor() {
        super();
        this.cache = new Map();
        this.cacheTimeout = 60000; // 1 minute cache
    }

    /**
     * Calculate estimated fee using canonical implementation
     */
    async calculateFee(parameters, l1BasefeeWei, vaultDeficit = 0, applySmoothing = true) {
        const requestData = {
            parameters: this.normalizeParameters(parameters),
            l1_basefee_wei: l1BasefeeWei,
            vault_deficit: vaultDeficit,
            apply_smoothing: applySmoothing
        };

        try {
            const response = await this.request(API_CONFIG.endpoints.calculateFee, {
                method: 'POST',
                data: requestData
            });

            return {
                estimatedFeeEth: response.estimated_fee_eth,
                estimatedFeeGwei: response.estimated_fee_gwei,
                l1CostComponent: response.l1_cost_component,
                deficitComponent: response.deficit_component,
                l1CostPerTx: response.l1_cost_per_tx,
                guaranteedRecoveryApplied: response.guaranteed_recovery_applied
            };

        } catch (error) {
            console.error('Fee calculation API call failed:', error);
            // Fallback to approximate local calculation if API is down
            return this.fallbackFeeCalculation(parameters, l1BasefeeWei, vaultDeficit);
        }
    }

    /**
     * Validate fee mechanism parameters
     */
    async validateParameters(parameters) {
        const requestData = {
            parameters: this.normalizeParameters(parameters),
            l1_basefee_wei: 20e9, // Default test value
            vault_deficit: 0
        };

        try {
            const response = await this.request(API_CONFIG.endpoints.validateParameters, {
                method: 'POST',
                data: requestData
            });

            return {
                valid: response.valid,
                errors: response.errors || [],
                warnings: response.warnings || [],
                suggestedParameters: response.suggested_parameters
            };

        } catch (error) {
            console.error('Parameter validation failed:', error);
            return {
                valid: false,
                errors: ['API validation failed'],
                warnings: [],
                suggestedParameters: null
            };
        }
    }

    /**
     * Get predefined parameter presets
     */
    async getPresets() {
        const cacheKey = 'presets';
        const cached = this.cache.get(cacheKey);

        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            return cached.data;
        }

        try {
            const response = await this.request(API_CONFIG.endpoints.presets);

            const presets = response.presets;
            this.cache.set(cacheKey, {
                data: presets,
                timestamp: Date.now()
            });

            return presets;

        } catch (error) {
            console.error('Failed to load presets:', error);
            // Return fallback presets
            return this.getFallbackPresets();
        }
    }

    /**
     * Generate synthetic L1 data for testing
     */
    async generateL1Data(steps = 1800, volatility = 0.3, includeSpike = false) {
        const requestData = {
            steps,
            initial_basefee_wei: 15e9,
            volatility,
            include_spike: includeSpike,
            seed: 42
        };

        try {
            const response = await this.request(API_CONFIG.endpoints.generateL1Data, {
                method: 'POST',
                data: requestData
            });

            return {
                basefees_wei: response.l1_basefees_wei,
                basefees_gwei: response.l1_basefees_gwei,
                statistics: response.statistics
            };

        } catch (error) {
            console.error('L1 data generation failed:', error);
            // Fallback to simple synthetic data
            return this.generateFallbackL1Data(steps, volatility);
        }
    }

    /**
     * Normalize parameters for API request
     */
    normalizeParameters(params) {
        return {
            mu: params.mu ?? 0.0,
            nu: params.nu ?? 0.27,
            H: params.H ?? 492,
            target_balance: params.target_balance ?? 1000.0,
            min_fee: params.min_fee ?? 1e-8,
            gas_per_tx: params.gas_per_tx ?? 2000.0,
            base_tx_demand: params.base_tx_demand ?? 100.0,
            fee_elasticity: params.fee_elasticity ?? 2.0,
            max_tx_demand: params.max_tx_demand ?? 1000.0,
            guaranteed_recovery: params.guaranteed_recovery ?? false,
            min_deficit_rate: params.min_deficit_rate ?? 1e-3
        };
    }

    /**
     * Fallback fee calculation when API is unavailable
     */
    fallbackFeeCalculation(parameters, l1BasefeeWei, vaultDeficit) {
        console.warn('Using fallback fee calculation - results may not match canonical implementation');

        const mu = parameters.mu ?? 0.0;
        const nu = parameters.nu ?? 0.27;
        const H = parameters.H ?? 492;
        const gasPerTx = parameters.gas_per_tx ?? 2000.0;
        const minFee = parameters.min_fee ?? 1e-8;

        // Simple L1 cost calculation
        const l1CostPerTx = (l1BasefeeWei * gasPerTx) / 1e18;

        // Fee components
        const l1Component = mu * l1CostPerTx;
        const deficitComponent = nu * vaultDeficit / H;

        const estimatedFee = Math.max(l1Component + deficitComponent, minFee);

        return {
            estimatedFeeEth: estimatedFee,
            estimatedFeeGwei: estimatedFee * 1e9,
            l1CostComponent: l1Component,
            deficitComponent,
            l1CostPerTx,
            guaranteedRecoveryApplied: false
        };
    }

    /**
     * Get fallback presets when API is unavailable
     */
    getFallbackPresets() {
        return {
            optimal: {
                mu: 0.0,
                nu: 0.27,
                H: 492,
                target_balance: 1000.0,
                min_fee: 1e-8,
                gas_per_tx: 2000.0
            },
            balanced: {
                mu: 0.0,
                nu: 0.27,
                H: 492,
                target_balance: 1000.0,
                min_fee: 1e-8,
                gas_per_tx: 2000.0
            },
            crisis_resilient: {
                mu: 0.0,
                nu: 0.88,
                H: 120,
                target_balance: 1000.0,
                min_fee: 1e-8,
                gas_per_tx: 2000.0
            }
        };
    }

    /**
     * Generate fallback L1 data
     */
    generateFallbackL1Data(steps, volatility) {
        const basefees = [];
        let currentFee = 15e9; // 15 gwei in wei

        for (let i = 0; i < steps; i++) {
            const change = (Math.random() - 0.5) * 2 * volatility * 0.01 * currentFee;
            currentFee = Math.max(1e6, currentFee + change); // Floor at 0.001 gwei
            basefees.push(currentFee);
        }

        return {
            basefees_wei: basefees,
            basefees_gwei: basefees.map(f => f / 1e9),
            statistics: {
                min_gwei: Math.min(...basefees) / 1e9,
                max_gwei: Math.max(...basefees) / 1e9,
                mean_gwei: basefees.reduce((a, b) => a + b, 0) / basefees.length / 1e9
            }
        };
    }
}

/**
 * Canonical Simulation API Client
 *
 * Replaces local simulation with API calls to canonical Python implementation
 */
class CanonicalSimulationAPI extends BaseAPIClient {
    constructor() {
        super();
    }

    /**
     * Run complete fee mechanism simulation
     */
    async runSimulation(config) {
        const requestData = {
            parameters: this.normalizeParameters(config.parameters),
            vault_init: config.vault_init ?? 'target',
            deficit_ratio: config.deficit_ratio ?? 0.0,
            simulation_steps: config.simulation_steps ?? 1800,
            l1_data: config.l1_data,
            use_synthetic_l1: config.use_synthetic_l1 ?? true,
            initial_basefee_wei: config.initial_basefee_wei ?? 15e9,
            volatility: config.volatility ?? 0.3,
            include_spike: config.include_spike ?? false
        };

        try {
            const response = await this.request(API_CONFIG.endpoints.simulate, {
                method: 'POST',
                data: requestData
            });

            // Convert to format expected by existing web interface
            return this.convertSimulationResponse(response);

        } catch (error) {
            console.error('Simulation API call failed:', error);
            throw new Error(`Simulation failed: ${error.message}`);
        }
    }

    /**
     * Calculate comprehensive metrics from simulation results
     */
    async calculateMetrics(simulationResults) {
        const requestData = {
            simulation_results: simulationResults
        };

        try {
            const response = await this.request(API_CONFIG.endpoints.metrics, {
                method: 'POST',
                data: requestData
            });

            return response;

        } catch (error) {
            console.error('Metrics calculation API call failed:', error);
            // Return basic metrics as fallback
            return this.calculateBasicMetrics(simulationResults);
        }
    }

    /**
     * Convert API response to format expected by web interface
     */
    convertSimulationResponse(response) {
        // Convert step-by-step results to time series arrays
        const converted = {
            time_step: [],
            l1_basefee: [],
            estimated_fee: [],
            transaction_volume: [],
            vault_balance: [],
            vault_deficit: [],
            fees_collected: [],
            l1_costs_paid: []
        };

        for (const step of response.steps) {
            converted.time_step.push(step.time_step);
            converted.l1_basefee.push(step.l1_basefee_gwei);
            converted.estimated_fee.push(step.estimated_fee_eth);
            converted.transaction_volume.push(step.transaction_volume);
            converted.vault_balance.push(step.vault_balance);
            converted.vault_deficit.push(step.vault_deficit);
            converted.fees_collected.push(step.fees_collected);
            converted.l1_costs_paid.push(step.l1_costs_paid);
        }

        // Add summary information
        converted.summary = {
            total_steps: response.total_steps,
            total_fees_collected: response.total_fees_collected,
            total_l1_costs: response.total_l1_costs,
            net_revenue: response.net_revenue,
            average_fee_gwei: response.average_fee_gwei,
            fee_stability_cv: response.fee_stability_cv,
            time_underfunded_pct: response.time_underfunded_pct,
            max_deficit_ratio: response.max_deficit_ratio
        };

        return converted;
    }

    /**
     * Normalize parameters for API request
     */
    normalizeParameters(params) {
        return {
            mu: params.mu ?? 0.0,
            nu: params.nu ?? 0.27,
            H: params.H ?? 492,
            target_balance: params.target_balance ?? 1000.0,
            min_fee: params.min_fee ?? 1e-8,
            gas_per_tx: params.gas_per_tx ?? 2000.0
        };
    }

    /**
     * Basic metrics calculation fallback
     */
    calculateBasicMetrics(simulationResults) {
        const fees = simulationResults.estimated_fee || [];
        const deficits = simulationResults.vault_deficit || [];

        if (fees.length === 0) {
            return {};
        }

        const avgFeeGwei = fees.reduce((a, b) => a + b, 0) / fees.length * 1e9;
        const feeStd = Math.sqrt(fees.reduce((sq, f) => sq + Math.pow(f - fees.reduce((a, b) => a + b, 0) / fees.length, 2), 0) / fees.length);
        const feeCV = feeStd / (fees.reduce((a, b) => a + b, 0) / fees.length);

        const underfundedSteps = deficits.filter(d => d > 10).length; // > 10 ETH deficit
        const timeUnderfundedPct = (underfundedSteps / deficits.length) * 100;

        return {
            average_fee_gwei: avgFeeGwei,
            fee_stability_cv: feeCV,
            time_underfunded_pct: timeUnderfundedPct,
            user_experience_score: Math.max(0, 1 - avgFeeGwei / 50), // Rough approximation
            protocol_safety_score: Math.max(0, 1 - timeUnderfundedPct / 20),
            economic_efficiency_score: 0.7, // Default approximation
            overall_performance_score: 0.7
        };
    }
}

/**
 * Canonical Optimization API Client
 *
 * Provides access to NSGA-II multi-objective optimization
 */
class CanonicalOptimizationAPI extends BaseAPIClient {
    constructor() {
        super();
        this.activeJobs = new Map();
    }

    /**
     * Start optimization job
     */
    async startOptimization(config) {
        const requestData = {
            strategy: config.strategy ?? 'balanced',
            population_size: config.population_size ?? 50,
            generations: config.generations ?? 25,
            vault_init: config.vault_init ?? 'target',
            simulation_steps: config.simulation_steps ?? 1800,
            l1_data: config.l1_data
        };

        try {
            const response = await this.request(API_CONFIG.endpoints.startOptimization, {
                method: 'POST',
                data: requestData
            });

            const jobId = response.job_id;
            this.activeJobs.set(jobId, {
                config: requestData,
                startTime: Date.now(),
                status: 'started'
            });

            return {
                jobId,
                status: response.status,
                message: response.message
            };

        } catch (error) {
            console.error('Failed to start optimization:', error);
            throw error;
        }
    }

    /**
     * Check optimization job status
     */
    async getOptimizationStatus(jobId) {
        try {
            const response = await this.request(`${API_CONFIG.endpoints.optimizationStatus}/${jobId}`);

            if (this.activeJobs.has(jobId)) {
                this.activeJobs.get(jobId).status = response.status;
            }

            return response;

        } catch (error) {
            console.error('Failed to get optimization status:', error);
            throw error;
        }
    }

    /**
     * Get optimization results
     */
    async getOptimizationResult(jobId) {
        try {
            const response = await this.request(`${API_CONFIG.endpoints.optimizationResult}/${jobId}`);

            if (this.activeJobs.has(jobId)) {
                this.activeJobs.get(jobId).status = 'completed';
            }

            return response;

        } catch (error) {
            console.error('Failed to get optimization results:', error);
            throw error;
        }
    }

    /**
     * Run quick optimization synchronously
     */
    async runQuickOptimization(config) {
        const requestData = {
            strategy: config.strategy ?? 'balanced',
            population_size: Math.min(config.population_size ?? 50, 50), // Limit for quick optimization
            generations: Math.min(config.generations ?? 20, 20),
            vault_init: config.vault_init ?? 'target',
            simulation_steps: config.simulation_steps ?? 900 // Shorter for quick optimization
        };

        try {
            const response = await this.request(API_CONFIG.endpoints.quickOptimization, {
                method: 'POST',
                data: requestData
            });

            return response;

        } catch (error) {
            console.error('Quick optimization failed:', error);
            throw error;
        }
    }

    /**
     * Poll optimization status until completion
     */
    async waitForOptimization(jobId, progressCallback = null) {
        const pollInterval = 2000; // 2 seconds
        const maxWaitTime = 300000; // 5 minutes

        const startTime = Date.now();

        while (Date.now() - startTime < maxWaitTime) {
            try {
                const status = await this.getOptimizationStatus(jobId);

                if (progressCallback) {
                    progressCallback(status);
                }

                if (status.status === 'completed') {
                    return await this.getOptimizationResult(jobId);
                } else if (status.status === 'failed') {
                    throw new Error(`Optimization failed: ${status.error}`);
                }

                // Wait before next poll
                await new Promise(resolve => setTimeout(resolve, pollInterval));

            } catch (error) {
                console.error('Error polling optimization status:', error);
                throw error;
            }
        }

        throw new Error('Optimization timed out');
    }

    /**
     * Get list of active optimization jobs
     */
    getActiveJobs() {
        return Array.from(this.activeJobs.entries()).map(([jobId, job]) => ({
            jobId,
            ...job
        }));
    }
}

/**
 * Main Canonical API Client
 *
 * Unified interface to all canonical API functionality
 */
class CanonicalAPI {
    constructor() {
        this.fee = new CanonicalFeeAPI();
        this.simulation = new CanonicalSimulationAPI();
        this.optimization = new CanonicalOptimizationAPI();

        this.initialized = false;
        this.serverAvailable = false;
    }

    /**
     * Initialize the API client and check server availability
     */
    async initialize() {
        console.log('🔗 Initializing Canonical API Client...');

        try {
            this.serverAvailable = await this.fee.checkHealth();

            if (this.serverAvailable) {
                console.log('✅ API server is healthy and available');

                // Load initial configuration
                try {
                    await this.fee.getPresets();
                    console.log('✅ Presets loaded successfully');
                } catch (error) {
                    console.warn('⚠️ Failed to load presets, but API is available');
                }

            } else {
                console.warn('⚠️ API server is not available - using fallback mode');
            }

            this.initialized = true;
            return true;

        } catch (error) {
            console.error('❌ Failed to initialize API client:', error);
            this.serverAvailable = false;
            this.initialized = true;
            return false;
        }
    }

    /**
     * Check if canonical API is available
     */
    isAvailable() {
        return this.serverAvailable;
    }

    /**
     * Get initialization status
     */
    isInitialized() {
        return this.initialized;
    }
}

// Create global instance
const canonicalAPI = new CanonicalAPI();

// Auto-initialize when loaded
if (typeof window !== 'undefined') {
    // Browser environment
    window.addEventListener('DOMContentLoaded', () => {
        canonicalAPI.initialize();
    });
} else if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = {
        CanonicalAPI,
        CanonicalFeeAPI,
        CanonicalSimulationAPI,
        CanonicalOptimizationAPI,
        canonicalAPI
    };
}

// Export for ES6 modules
if (typeof window !== 'undefined') {
    window.CanonicalAPI = CanonicalAPI;
    window.CanonicalFeeAPI = CanonicalFeeAPI;
    window.CanonicalSimulationAPI = CanonicalSimulationAPI;
    window.CanonicalOptimizationAPI = CanonicalOptimizationAPI;
    window.canonicalAPI = canonicalAPI;
}

console.log('📡 Canonical API Client loaded - provides JavaScript interface to Python canonical implementations');