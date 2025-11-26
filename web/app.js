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
                const presetName = e.target.dataset.preset;
                this.loadPreset(presetName);
                this.setActivePreset(e.target);
            });
        });


        // Tooltips
        this.initializeTooltips();

        // Formula info modal
        this.initializeFormulaModal();
    }

    initializeTooltips() {
        // Simple tooltip implementation
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
    }

    getCurrentParameters() {
        // Determine active tab
        const activeTab = document.querySelector('.tab-button.active').dataset.tab;
        const isSimulated = activeTab === 'simulated';
        const historicalPeriod = isSimulated ? null : document.querySelector('input[name="historical-period"]:checked').value;

        return {
            mu: parseFloat(document.getElementById('mu-slider').value),
            nu: parseFloat(document.getElementById('nu-slider').value),
            H: parseInt(document.getElementById('H-slider').value),
            l1Volatility: parseFloat(document.getElementById('volatility-slider').value),
            l1Source: isSimulated ? 'simulated' : 'historical',
            historicalPeriod: historicalPeriod,
            seed: parseInt(document.getElementById('seed-input').value),
            vaultInit: document.getElementById('vault-init').value,
            baseTxVolume: parseInt(document.getElementById('tx-per-block-slider').value),
            spikeDelay: parseInt(document.getElementById('spike-delay-slider').value),
            spikeHeight: parseFloat(document.getElementById('spike-height-slider').value),
            targetBalance: 1000,
            feeElasticity: 0.2,
            minFee: 1e-8
        };
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

            // Run simulation (use setTimeout to make it non-blocking)
            await new Promise(resolve => setTimeout(resolve, 100));
            const simulationData = await simulator.runSimulation(300);

            console.log('Simulation completed, data points:', simulationData.length);

            // Calculate metrics
            const metricsCalculator = new MetricsCalculator(params.targetBalance);
            const metrics = metricsCalculator.calculateMetrics(simulationData);

            console.log('Calculated metrics:', metrics);

            // Store data
            this.currentSimulationData = simulationData;

            // Update UI
            this.updateMetricsDisplay(metrics);
            this.updateCharts(simulationData);

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

    updateCharts(simulationData) {
        const params = this.getCurrentParameters();

        // Create/update all charts
        this.chartManager.createFeeChart('fee-chart', simulationData);
        this.chartManager.createVaultChart('vault-chart', simulationData, params.targetBalance);
        this.chartManager.createL1Chart('l1-chart', simulationData);
        this.chartManager.createCorrelationChart('correlation-chart', simulationData);
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