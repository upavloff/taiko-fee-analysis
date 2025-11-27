// Main application logic for Taiko Fee Explorer

class TaikoFeeExplorer {
    constructor() {
        this.chartManager = new ChartManager();
        this.currentSimulationData = null;
        this.isRunning = false;
        this.autoRunEnabled = true;
        this.autoRunDelay = 800; // ms delay after last parameter change

        this.initializeEventListeners();
        this.updateParameterDisplays();

        // Initialize minimum deficit rate visibility based on guaranteed recovery checkbox
        this.toggleMinDeficitRateVisibility(document.getElementById('guaranteed-recovery').checked);

        // Auto-run simulation on page load
        setTimeout(() => {
            this.runSimulation();
        }, 500);
    }

    initializeEventListeners() {
        // Parameter sliders
        document.getElementById('mu-slider').addEventListener('input', (e) => {
            document.getElementById('mu-value').textContent = e.target.value;
            this.clearActivePreset();
            this.markParametersModified();
        });

        document.getElementById('nu-slider').addEventListener('input', (e) => {
            document.getElementById('nu-value').textContent = e.target.value;
            this.clearActivePreset();
            this.markParametersModified();
        });

        document.getElementById('H-slider').addEventListener('input', (e) => {
            document.getElementById('H-value').textContent = e.target.value;
            this.clearActivePreset();
            this.markParametersModified();
        });

        document.getElementById('volatility-slider').addEventListener('input', (e) => {
            document.getElementById('volatility-value').textContent = e.target.value;
            this.clearActivePreset();
            this.markParametersModified();
        });

        document.getElementById('spike-delay-slider').addEventListener('input', (e) => {
            document.getElementById('spike-delay-value').textContent = e.target.value;
            this.clearActivePreset();
            this.markParametersModified();
        });

        document.getElementById('spike-height-slider').addEventListener('input', (e) => {
            document.getElementById('spike-height-value').textContent = e.target.value;
            this.clearActivePreset();
            this.markParametersModified();
        });

        document.getElementById('duration-slider').addEventListener('input', (e) => {
            document.getElementById('duration-value').textContent = parseFloat(e.target.value).toFixed(1);
            this.clearActivePreset();
            this.markParametersModified();
        });

        document.getElementById('tx-per-block-slider').addEventListener('input', (e) => {
            document.getElementById('tx-per-block-value').textContent = e.target.value;
            this.clearActivePreset();
            this.markParametersModified();
        });

        // Vault initialization
        document.getElementById('vault-init').addEventListener('change', () => {
            this.clearActivePreset();
            this.markParametersModified();
        });

        // Guaranteed recovery toggle
        document.getElementById('guaranteed-recovery').addEventListener('change', (e) => {
            this.toggleMinDeficitRateVisibility(e.target.checked);
            this.clearActivePreset();
            this.markParametersModified();
        });

        // Minimum deficit rate slider
        document.getElementById('min-deficit-rate-slider').addEventListener('input', (e) => {
            const exponentValue = parseFloat(e.target.value);
            const actualValue = Math.pow(10, exponentValue);
            document.getElementById('min-deficit-rate-value').textContent = actualValue.toExponential(3);
            this.clearActivePreset();
            this.markParametersModified();
        });

        // Tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
                this.clearActivePreset();
                this.runSimulation(); // Immediate update when switching data source
            });
        });

        // Historical period selection
        document.querySelectorAll('input[name="historical-period"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this.clearActivePreset();
                this.runSimulation(); // Immediate update when changing historical periods
            });
        });

        // Seed input
        document.getElementById('seed-input').addEventListener('input', () => {
            this.clearActivePreset();
            this.markParametersModified();
        });

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Check if tooltip trigger was clicked
                if (e.target.classList.contains('preset-tooltip-trigger')) {
                    e.stopPropagation();
                    const presetName = e.target.closest('.preset-btn').dataset.preset;
                    this.showPresetTooltip(presetName);
                    return;
                }

                const presetName = e.target.dataset.preset;
                this.loadPreset(presetName);
                this.setActivePreset(e.target);
            });
        });


        // Tooltips
        this.initializeTooltips();

        // Formula info modal
        this.initializeFormulaModal();

        // Preset tooltip modal
        this.initializePresetTooltip();
    }

    initializeTooltips() {
        // Simple tooltip implementation for general tooltips
        const tooltips = document.querySelectorAll('.tooltip');
        tooltips.forEach(tooltip => {
            let tooltipElement = null;

            tooltip.addEventListener('mouseenter', (e) => {
                const text = e.target.getAttribute('data-tooltip');
                tooltipElement = document.createElement('div');
                tooltipElement.className = 'tooltip-popup';
                tooltipElement.textContent = text;
                tooltipElement.style.cssText = `
                    position: absolute;
                    background: #1e293b;
                    color: white;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 0.9em;
                    max-width: 300px;
                    z-index: 1000;
                    pointer-events: none;
                    word-wrap: break-word;
                `;

                document.body.appendChild(tooltipElement);

                const rect = e.target.getBoundingClientRect();
                tooltipElement.style.left = (rect.left + window.scrollX) + 'px';
                tooltipElement.style.top = (rect.top + window.scrollY - tooltipElement.offsetHeight - 10) + 'px';
            });

            tooltip.addEventListener('mouseleave', () => {
                if (tooltipElement) {
                    document.body.removeChild(tooltipElement);
                    tooltipElement = null;
                }
            });
        });

        // Enhanced tooltip implementation for preset buttons
        const presetButtons = document.querySelectorAll('.preset-btn');
        presetButtons.forEach(button => {
            let tooltipElement = null;

            button.addEventListener('mouseenter', (e) => {
                // Don't show tooltip if clicking on info icon
                if (e.target.classList.contains('preset-tooltip-trigger')) return;

                const text = button.getAttribute('data-tooltip');
                if (!text) return;

                tooltipElement = document.createElement('div');
                tooltipElement.className = 'preset-hover-tooltip';
                tooltipElement.innerHTML = text;
                tooltipElement.style.cssText = `
                    position: absolute;
                    background: #2d3748;
                    color: white;
                    padding: 12px;
                    border-radius: 8px;
                    font-size: 11px;
                    max-width: 350px;
                    z-index: 1000;
                    pointer-events: none;
                    word-wrap: break-word;
                    line-height: 1.4;
                    border: 1px solid #4a5568;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                `;

                document.body.appendChild(tooltipElement);

                const rect = button.getBoundingClientRect();
                tooltipElement.style.left = (rect.left + window.scrollX) + 'px';
                tooltipElement.style.top = (rect.top + window.scrollY - tooltipElement.offsetHeight - 12) + 'px';

                // Adjust if tooltip goes off screen
                if (tooltipElement.getBoundingClientRect().left < 0) {
                    tooltipElement.style.left = '10px';
                }
                if (tooltipElement.getBoundingClientRect().right > window.innerWidth) {
                    tooltipElement.style.left = (window.innerWidth - tooltipElement.offsetWidth - 10) + 'px';
                }
            });

            button.addEventListener('mouseleave', () => {
                if (tooltipElement) {
                    document.body.removeChild(tooltipElement);
                    tooltipElement = null;
                }
            });
        });
    }

    initializeFormulaModal() {
        const infoBtn = document.getElementById('formula-info-btn');
        const infoBox = document.getElementById('formula-info-box');

        if (infoBtn && infoBox) {
            infoBtn.addEventListener('click', () => {
                if (infoBox.style.display === 'none') {
                    this.updateInfoBox();
                    infoBox.style.display = 'block';
                } else {
                    infoBox.style.display = 'none';
                }
            });
        }
    }

    updateInfoBox() {
        const params = this.getCurrentParameters();

        document.getElementById('info-mu').textContent = params.mu;
        document.getElementById('info-nu').textContent = params.nu;
        document.getElementById('info-h').textContent = params.H;
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`.tab-button[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    updateParameterDisplays() {
        document.getElementById('mu-value').textContent = document.getElementById('mu-slider').value;
        document.getElementById('nu-value').textContent = document.getElementById('nu-slider').value;
        document.getElementById('H-value').textContent = document.getElementById('H-slider').value;
        document.getElementById('volatility-value').textContent = document.getElementById('volatility-slider').value;
        document.getElementById('tx-per-block-value').textContent = document.getElementById('tx-per-block-slider').value;
        document.getElementById('duration-value').textContent = parseFloat(document.getElementById('duration-slider').value).toFixed(1);

        // Update minimum deficit rate display
        const exponentValue = parseFloat(document.getElementById('min-deficit-rate-slider').value);
        const actualValue = Math.pow(10, exponentValue);
        document.getElementById('min-deficit-rate-value').textContent = actualValue.toExponential(3);
    }

    toggleMinDeficitRateVisibility(show) {
        const deficitRateGroup = document.getElementById('min-deficit-rate-group');
        deficitRateGroup.style.display = show ? 'block' : 'none';
    }

    getCurrentParameters() {
        // Determine active tab
        const activeTab = document.querySelector('.tab-button.active').dataset.tab;
        const isSimulated = activeTab === 'simulated';
        const historicalPeriod = isSimulated ? null : document.querySelector('input[name="historical-period"]:checked').value;

        // Calculate minimum deficit rate from the exponential slider
        const minDeficitRateExponent = parseFloat(document.getElementById('min-deficit-rate-slider').value);
        const minDeficitRateValue = Math.pow(10, minDeficitRateExponent);

        const params = {
            mu: parseFloat(document.getElementById('mu-slider').value),
            nu: parseFloat(document.getElementById('nu-slider').value),
            H: parseInt(document.getElementById('H-slider').value),
            l1Volatility: parseFloat(document.getElementById('volatility-slider').value),
            l1Source: isSimulated ? 'simulated' : 'historical',
            historicalPeriod: historicalPeriod,
            seed: parseInt(document.getElementById('seed-input').value),
            vaultInit: document.getElementById('vault-init').value,
            txsPerBatch: parseInt(document.getElementById('tx-per-block-slider').value),
            spikeDelay: parseInt(document.getElementById('spike-delay-slider').value),
            spikeHeight: parseFloat(document.getElementById('spike-height-slider').value),
            guaranteedRecovery: document.getElementById('guaranteed-recovery').checked,
            minDeficitRate: minDeficitRateValue,
            durationHours: parseFloat(document.getElementById('duration-slider').value),
            targetBalance: 100,
            feeElasticity: 0.2,
            minFee: 1e-8
        };

        console.log(`App parameters:`, params);
        return params;
    }

    loadPreset(presetName) {
        const preset = PRESETS[presetName];
        if (!preset) return;

        // Update sliders
        document.getElementById('mu-slider').value = preset.mu;
        document.getElementById('nu-slider').value = preset.nu;
        document.getElementById('H-slider').value = preset.H;

        // Update displays
        this.updateParameterDisplays();

        // Mark as modified since preset changed parameters
        this.markParametersModified();

        console.log(`Loaded preset: ${presetName}`, preset);
    }

    setActivePreset(buttonElement) {
        // Clear all active states
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
        });

        // Set active state
        buttonElement.classList.add('active');
    }

    clearActivePreset() {
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
        });
    }

    markParametersModified() {
        // Auto-run simulation after delay
        if (this.autoRunEnabled && !this.isRunning) {
            clearTimeout(this.autoRunTimeout);
            this.autoRunTimeout = setTimeout(() => {
                if (!this.isRunning) {
                    this.runSimulation();
                }
            }, this.autoRunDelay);
        }
    }


    async runSimulation() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.showLoading(true);

        try {
            const params = this.getCurrentParameters();
            console.log('Running simulation with parameters:', params);

            // Create simulator
            const simulator = new TaikoFeeSimulator(params);

            // Calculate steps from duration (1800 steps per hour for 2-second Taiko blocks)
            const simulationSteps = Math.round(params.durationHours * 1800);

            // Convert percentage-based spike delay to actual steps
            params.spikeDelaySteps = Math.round((params.spikeDelay / 100) * simulationSteps);

            // Run simulation (use setTimeout to make it non-blocking)
            await new Promise(resolve => setTimeout(resolve, 100));
            const simulationData = await simulator.runSimulation(simulationSteps);

            console.log('Simulation completed, data points:', simulationData.length);

            // Calculate metrics
            const metricsCalculator = new MetricsCalculator(params.targetBalance, simulator.gasPerTx);
            const metrics = metricsCalculator.calculateMetrics(simulationData);

            console.log('Calculated metrics:', metrics);

            // Store data
            this.currentSimulationData = simulationData;

            // Update UI
            this.updateMetricsDisplay(metrics);
            this.updateCharts(simulationData, simulator.gasPerTx);

        } catch (error) {
            console.error('Simulation error:', error);
            alert('Error running simulation: ' + error.message);
        } finally {
            this.isRunning = false;
            this.showLoading(false);
        }
    }

    updateMetricsDisplay(metrics) {
        const evaluations = this.chartManager.evaluateMetrics(metrics);

        // Update each metric card
        this.chartManager.updateMetricCard('avg-fee-card', metrics.avgFee, evaluations.avgFee);
        this.chartManager.updateMetricCard('fee-cv-card', metrics.feeCV, evaluations.feeCV);
        this.chartManager.updateMetricCard('underfunded-card', metrics.timeUnderfundedPct, evaluations.timeUnderfunded);
        this.chartManager.updateMetricCard('tracking-card', metrics.l1TrackingError, evaluations.l1Tracking);
    }

    updateCharts(simulationData, gasPerTx) {
        const params = this.getCurrentParameters();

        // Create/update all charts
        this.chartManager.createFeeChart('fee-chart', simulationData, gasPerTx);
        this.chartManager.createVaultChart('vault-chart', simulationData, params.targetBalance);
        this.chartManager.createL1Chart('l1-chart', simulationData);
        this.chartManager.createCorrelationChart('correlation-chart', simulationData, gasPerTx);
        this.chartManager.createL1EstimationChart('l1-estimation-chart', simulationData);
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');

        if (show) {
            overlay.style.display = 'flex';
        } else {
            overlay.style.display = 'none';
        }
    }

    // Export functionality
    exportResults() {
        if (!this.currentSimulationData) {
            alert('No simulation data to export. Please run a simulation first.');
            return;
        }

        const params = this.getCurrentParameters();
        const exportData = {
            timestamp: new Date().toISOString(),
            parameters: params,
            simulationData: this.currentSimulationData
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

        const exportFileDefaultName = 'taiko-fee-simulation.json';

        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    }

    // Preset Tooltip Methods
    initializePresetTooltip() {
        const modal = document.getElementById('preset-tooltip-modal');
        const closeBtn = document.querySelector('.preset-tooltip-close');

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                modal.style.display = 'none';
            });
        }

        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'block') {
                modal.style.display = 'none';
            }
        });
    }

    showPresetTooltip(presetName) {
        const preset = PRESETS[presetName];
        if (!preset) return;

        const modal = document.getElementById('preset-tooltip-modal');
        const title = document.getElementById('preset-tooltip-title');
        const details = document.getElementById('preset-tooltip-details');

        title.textContent = `${preset.description}`;
        details.innerHTML = this.generatePresetTooltipContent(presetName, preset);

        modal.style.display = 'block';

        // Re-render MathJax for the new LaTeX content
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
            MathJax.typesetPromise([modal]).catch((err) => console.log('MathJax error:', err));
        }
    }

    generatePresetTooltipContent(presetName, preset) {
        const researchData = this.getPresetResearchData(presetName);

        return `
            <div class="preset-detail-section">
                <h5>🔧 Parameter Configuration</h5>
                <div class="preset-parameters">
                    <div class="preset-param">μ = ${preset.mu}</div>
                    <div class="preset-param">ν = ${preset.nu}</div>
                    <div class="preset-param">H = ${preset.H}</div>
                </div>
            </div>

            <div class="preset-detail-section">
                <h5>📐 Mathematical Formula</h5>
                <div class="preset-formula">
                    $$F_E(t) = \\max\\left(${preset.mu} \\times C_{L1}(t) + ${preset.nu} \\times \\frac{D(t)}{${preset.H}}, F_{\\text{min}}\\right)$$
                </div>
                <p style="font-size: 11px; color: #666; margin: 8px 0 0 0;">
                    ${this.getFormulaExplanation(preset)}
                </p>
            </div>

            <div class="preset-detail-section">
                <h5>🎯 Objective & Constraints</h5>
                <div class="preset-optimization">
                    <div class="preset-objective">
                        <h6>Primary Objective:</h6>
                        <p>${preset.objective || 'Not specified'}</p>
                    </div>
                    <div class="preset-constraints">
                        <h6>Constraints:</h6>
                        <p>${preset.constraints || 'Not specified'}</p>
                    </div>
                    <div class="preset-risk">
                        <h6>Risk Profile:</h6>
                        <p>${preset.riskProfile || 'Not specified'}</p>
                    </div>
                </div>
            </div>

            <div class="preset-detail-section">
                <h5>⚖️ Trade-offs Analysis</h5>
                <div class="preset-tradeoffs">
                    <p>${preset.tradeoffs || 'Not specified'}</p>
                    <h6>Research Validation:</h6>
                    <ul class="preset-optimization-list">
                        <li>Comprehensive simulations across crisis scenarios</li>
                        <li>Multi-objective optimization: minimize fees + maximize vault stability</li>
                        <li>Historical data: May 2022 crash, July 2022 spike, May 2023 PEPE, Recent low fees</li>
                        <li>Pareto frontier analysis for optimal trade-offs</li>
                    </ul>
                </div>
            </div>

            <div class="preset-detail-section">
                <h5>🔬 Research Methodology & Status</h5>
                <div class="preset-performance">
                    <div class="preset-methodology">
                        <h6>Research Methodology:</h6>
                        <p>${researchData.methodology}</p>
                    </div>
                    <div class="preset-data-source">
                        <h6>Data Sources:</h6>
                        <p>${researchData.dataSource}</p>
                    </div>
                    <div class="preset-research-status">
                        <h6>Current Status:</h6>
                        <p>${researchData.researchStatus}</p>
                    </div>
                    <div class="preset-metric">
                        <span>Expected Time Underfunded:</span>
                        <span class="preset-metric-value">${researchData.timeUnderfunded}</span>
                    </div>
                </div>
            </div>

            <div class="preset-detail-section">
                <h5>🏗️ Use Case & Trade-offs</h5>
                <div class="preset-use-case">
                    <strong>Best Used For:</strong> ${preset.useCase}
                </div>
                <div class="preset-use-case">
                    <strong>Trade-off Analysis:</strong> ${researchData.tradeoffs}
                </div>
            </div>
        `;
    }

    getFormulaExplanation(preset) {
        if (preset.mu === 0.0) {
            return `Pure deficit correction: fees only depend on vault deficit (${preset.nu} × D(t)/${preset.H}), completely ignoring L1 costs. This minimizes fees while maintaining vault stability.`;
        } else if (preset.nu === 0.0) {
            return `Pure L1 tracking: fees only depend on L1 costs (${preset.mu} × C_L1(t)), ignoring vault deficit. This provides predictable cost reflection but no vault management.`;
        } else {
            return `Hybrid approach: fees balance L1 cost reflection (${preset.mu} × C_L1(t)) with vault deficit correction (${preset.nu} × D(t)/${preset.H}). Higher μ = more L1 sensitivity, higher ν = faster deficit correction.`;
        }
    }

    getPresetResearchData(presetName) {
        const researchData = {
            'optimal': {
                avgFee: 'N/A - Research Pending',
                timeUnderfunded: 'N/A - Research Pending',
                feeVolatility: 'N/A - Research Pending',
                l1TrackingError: 'N/A - Research Pending',
                methodology: '360 simulations across 4 historical crisis scenarios with multi-objective Pareto optimization',
                researchStatus: 'Configuration designed for μ=0.0, ν=0.3, H=288 based on deficit correction optimization. Full performance validation pending.',
                dataSource: 'Historical Ethereum data: July 2022 spike, May 2022 UST crash, May 2023 PEPE crisis, Recent low fees',
                tradeoffs: 'Research-based deficit correction strategy. Ignores L1 costs (μ=0.0) for minimal fees while using moderate deficit correction (ν=0.3) for stability.'
            },
            'balanced': {
                avgFee: 'N/A - Research Pending',
                timeUnderfunded: 'N/A - Research Pending',
                feeVolatility: 'N/A - Research Pending',
                l1TrackingError: 'N/A - Research Pending',
                methodology: '360 simulations with comprehensive parameter sweep and crisis scenario testing',
                researchStatus: 'Configuration designed for μ=0.0, ν=0.1, H=576 for conservative deficit correction. Performance metrics under validation.',
                dataSource: 'Multi-regime backtesting on real Ethereum L1 basefee data spanning normal and crisis periods',
                tradeoffs: 'Very conservative deficit correction approach. Extended horizon (H=576) provides stability but slower response to deficits compared to optimal config.'
            },
            'crisis-resilient': {
                avgFee: 'N/A - Research Pending',
                timeUnderfunded: 'Expected < 5%',
                feeVolatility: 'Expected Low',
                l1TrackingError: 'N/A (Pure deficit approach)',
                methodology: 'Crisis stress testing across extreme volatility scenarios with aggressive deficit correction',
                researchStatus: 'Configuration μ=0.0, ν=0.9, H=144 designed for maximum vault recovery speed during crises.',
                dataSource: 'Historical crisis scenarios: ETH fee spikes (8-533 gwei), meme coin congestion, UST/Luna crash',
                tradeoffs: 'Aggressive deficit correction (ν=0.9) prioritizes vault stability over fee minimization. Faster response than other configs but potentially higher fees during recovery periods.'
            }
        };

        return researchData[presetName] || researchData['optimal'];
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Taiko Fee Explorer initializing...');

    // Check if required dependencies are loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js not loaded');
        return;
    }

    if (typeof TaikoFeeSimulator === 'undefined') {
        console.error('Simulator not loaded');
        return;
    }

    // Initialize the application
    window.taikoExplorer = new TaikoFeeExplorer();

    console.log('Taiko Fee Explorer initialized successfully');

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to run simulation
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            window.taikoExplorer.runSimulation();
        }

        // Ctrl/Cmd + E to export results
        if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
            e.preventDefault();
            window.taikoExplorer.exportResults();
        }
    });

    // Show keyboard shortcuts in console
    console.log('Keyboard shortcuts:');
    console.log('Ctrl/Cmd + Enter: Run simulation');
    console.log('Ctrl/Cmd + E: Export results');
});