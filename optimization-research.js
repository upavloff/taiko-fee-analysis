/**
 * Interactive Parameter Optimization Research Interface
 *
 * Main controller for the optimization research tab featuring:
 * - Interactive weight configuration for 11 optimization metrics
 * - Real-time 3D Pareto frontier visualization
 * - NSGA-II multi-objective optimization
 * - Scientific research interface for protocol designers
 */

class OptimizationResearchController {
    constructor() {
        this.currentWeights = this.getDefaultWeights();
        this.lockedWeights = {
            ux: new Set(),
            safety: new Set(),
            efficiency: new Set()
        };
        this.paretoVisualizer = null;
        this.optimizationEngine = null;
        this.isOptimizing = false;
        this.currentSolutions = [];
        this.metricDefinitions = this.getMetricDefinitions();
        this.activeMetricTooltip = null;

        this.init();
    }

    /**
     * Comprehensive metric definitions with formulas and explanations
     */
    getMetricDefinitions() {
        return {
            'fee_affordability': {
                name: 'Fee Affordability',
                formula: '\\mathcal{A}(f) = -\\log(1 + \\bar{f} \\times 1000)',
                objectiveVariable: '\\mathcal{A}(f)',
                highlightText: 'A(f)',
                explanation: 'Measures how affordable fees are for users by applying a logarithmic penalty to the average fee. Lower average fees result in higher affordability scores.',
                details: [
                    '• $\\bar{f}$ is the average fee per gas unit over the simulation period',
                    '• Multiplied by 1000 to convert from ETH to milli-ETH scale',
                    '• Logarithmic function ensures diminishing returns - very high fees are severely penalized',
                    '• Higher scores indicate more affordable fees for users'
                ],
                range: 'Range: (-∞, 0], Higher is better',
                category: 'ux'
            },
            'fee_stability': {
                name: 'Fee Stability',
                formula: '\\mathcal{S}(f) = 1 - \\text{CV}(f)',
                objectiveVariable: '\\mathcal{S}(f)',
                highlightText: 'S(f)',
                explanation: 'Measures fee consistency by using the coefficient of variation. More stable fees provide better user experience and predictability.',
                details: [
                    '• $\\text{CV}(f) = \\frac{\\sigma_f}{\\bar{f}}$ is the coefficient of variation',
                    '• $\\sigma_f$ is the standard deviation of fees',
                    '• $\\bar{f}$ is the mean fee level',
                    '• Subtracting from 1 makes higher values represent better stability'
                ],
                range: 'Range: (-∞, 1], Higher is better',
                category: 'ux'
            },
            'fee_predictability_1h': {
                name: 'Fee Predictability (1h)',
                formula: '\\mathcal{P}_{1h}(f) = 1 - \\frac{\\text{RMSE}_{1h}}{\\bar{f}}',
                objectiveVariable: '\\mathcal{P}_{1h}(f)',
                highlightText: 'P_{1h}(f)',
                explanation: 'Measures how well fees can be predicted over a 1-hour window using recent trends. Better predictability helps users plan transactions.',
                details: [
                    '• $\\text{RMSE}_{1h}$ is the root mean square error of 1-hour predictions',
                    '• Uses simple moving average or trend-based prediction',
                    '• Normalized by mean fee to make it scale-invariant',
                    '• Higher scores indicate more predictable fee patterns'
                ],
                range: 'Range: (-∞, 1], Higher is better',
                category: 'ux'
            },
            'fee_predictability_6h': {
                name: 'Fee Predictability (6h)',
                formula: '\\mathcal{P}_{6h}(f) = 1 - \\frac{\\text{RMSE}_{6h}}{\\bar{f}}',
                objectiveVariable: '\\mathcal{P}_{6h}(f)',
                highlightText: 'P_{6h}(f)',
                explanation: 'Measures how well fees can be predicted over a 6-hour window. Longer-term predictability for strategic transaction timing.',
                details: [
                    '• $\\text{RMSE}_{6h}$ is the root mean square error of 6-hour predictions',
                    '• Uses trend analysis over longer periods',
                    '• Important for applications that need to plan ahead',
                    '• Generally lower than 1h predictability due to longer prediction horizon'
                ],
                range: 'Range: (-∞, 1], Higher is better',
                category: 'ux'
            },
            'insolvency_protection': {
                name: 'Insolvency Protection',
                formula: '\\mathcal{I}(v) = 1 - P(V < V_{\\text{crit}})',
                objectiveVariable: 'P_{\\text{insol}}',
                highlightText: 'insol',
                explanation: 'Measures the probability that the vault remains solvent (above critical threshold) throughout the simulation period.',
                details: [
                    '• $P(V < V_{\\text{crit}})$ is probability of vault falling below critical level',
                    '• $V_{\\text{crit}}$ is typically 10% of target vault balance',
                    '• Insolvency would halt the protocol or require emergency measures',
                    '• Higher scores indicate better protection against protocol failure'
                ],
                range: 'Range: [0, 1], Higher is better',
                category: 'safety'
            },
            'deficit_duration': {
                name: 'Deficit Duration Control',
                formula: '\\mathcal{D}(v) = 1 - \\frac{\\sum (d_i \\cdot t_i)^2}{T \\cdot V_{\\text{target}}^2}',
                objectiveVariable: '\\mathcal{D}_{\\text{weighted}}',
                highlightText: 'weighted',
                explanation: 'Measures how quickly deficits are corrected, with stronger penalties for longer deficit periods.',
                details: [
                    '• $d_i$ is the deficit amount at time step $i$',
                    '• $t_i$ is the duration of deficit period',
                    '• Squared terms heavily penalize extended deficits',
                    '• Normalized by total time $T$ and target vault size'
                ],
                range: 'Range: (-∞, 1], Higher is better',
                category: 'safety'
            },
            'vault_stress': {
                name: 'Vault Stress Resilience',
                formula: '\\mathcal{R}(v) = \\frac{\\Delta V_{\\text{recovery}}}{\\Delta t_{\\text{stress}}}',
                objectiveVariable: '\\mathcal{R}_{\\text{stress}}',
                highlightText: 'stress',
                explanation: 'Measures how quickly the vault recovers during high-stress periods (rapid L1 cost increases).',
                details: [
                    '• $\\Delta V_{\\text{recovery}}$ is vault balance improvement during stress',
                    '• $\\Delta t_{\\text{stress}}$ is duration of high-stress period',
                    '• Stress periods defined as top 10% of L1 cost spikes',
                    '• Higher values indicate faster stress recovery'
                ],
                range: 'Range: [0, +∞), Higher is better',
                category: 'safety'
            },
            'continuous_underfunding': {
                name: 'Underfunding Resistance',
                formula: '\\mathcal{U}(v) = 1 - \\frac{T_{\\text{deficit}}}{T_{\\text{total}}}',
                objectiveVariable: 'P_{\\text{deficit}}',
                highlightText: 'deficit',
                explanation: 'Measures the fraction of time the vault spends in deficit, with penalty for continuous underfunding.',
                details: [
                    '• $T_{\\text{deficit}}$ is total time spent below target balance',
                    '• $T_{\\text{total}}$ is total simulation time',
                    '• Extended underfunding periods receive additional penalties',
                    '• Higher scores indicate better maintenance of target balance'
                ],
                range: 'Range: [0, 1], Higher is better',
                category: 'safety'
            },
            'vault_utilization': {
                name: 'Vault Utilization',
                formula: '\\mathcal{V}(v) = 1 - \\frac{\\overline{|V - V_{\\text{target}}|}}{V_{\\text{target}}}',
                objectiveVariable: '\\mathcal{U}_{\\text{vault}}',
                highlightText: 'vault',
                explanation: 'Measures how close the vault balance stays to its optimal target level throughout the simulation.',
                details: [
                    '• $\\overline{|V - V_{\\text{target}}|}$ is mean absolute deviation from target',
                    '• $V_{\\text{target}}$ is the optimal vault balance',
                    '• Penalizes both over and underfunding equally',
                    '• Higher scores indicate better capital utilization'
                ],
                range: 'Range: (-∞, 1], Higher is better',
                category: 'efficiency'
            },
            'deficit_correction': {
                name: 'Deficit Correction Rate',
                formula: '\\mathcal{C}(v) = \\frac{\\sum \\Delta d_i}{\\sum \\Delta t_i}',
                objectiveVariable: '\\mathcal{C}_{\\text{correction}}',
                highlightText: 'correction',
                explanation: 'Measures the average rate at which deficits are corrected when they occur.',
                details: [
                    '• $\\Delta d_i$ is deficit reduction in period $i$',
                    '• $\\Delta t_i$ is time duration of correction period',
                    '• Only considers periods when vault is actively correcting deficits',
                    '• Higher rates indicate more efficient deficit correction'
                ],
                range: 'Range: [0, +∞), Higher is better',
                category: 'efficiency'
            },
            'capital_efficiency': {
                name: 'Capital Efficiency',
                formula: '\\mathcal{E}(v) = \\frac{\\text{Coverage}}{\\text{Capital Held}} \\times \\text{Utilization Factor}',
                objectiveVariable: '\\mathcal{E}_{\\text{capital}}',
                highlightText: 'capital',
                explanation: 'Measures how effectively the vault uses its capital to provide operational coverage.',
                details: [
                    '• Coverage is the total L1 costs successfully covered',
                    '• Capital Held is the average vault balance maintained',
                    '• Utilization Factor accounts for idle vs. active capital',
                    '• Higher efficiency means better capital productivity'
                ],
                range: 'Range: [0, +∞), Higher is better',
                category: 'efficiency'
            }
        };
    }

    /**
     * Initialize the research interface
     */
    init() {
        this.setupTabSwitching();
        this.setupMethodologyPanel();
        this.setupWeightSliders();
        this.setupWeightPresets();
        this.setupOptimizationControls();
        this.setupAnimationControls();
        this.setupMetricTooltips();
        // Don't initialize visualization immediately - wait for tab to be visible
        this.updateLiveFormulas();
        this.updateFormulaDisplay();

        // Check if optimization tab is active on load
        this.checkInitialTabState();

        console.log('🧬 Optimization Research Interface initialized');
    }

    /**
     * Check if optimization tab is active on initial page load
     */
    checkInitialTabState() {
        const activeTab = document.querySelector('.research-tab-button.active');
        if (activeTab && activeTab.dataset.tab === 'optimization') {
            // Optimization tab is active and visible, safe to initialize visualization
            console.log('🎯 Optimization tab is active on load, initializing visualization...');
            if (!this.paretoVisualizer) {
                // Wait a bit for DOM to be fully ready and visible
                setTimeout(() => {
                    const container = document.querySelector('#pareto-visualization');
                    if (container && container.clientWidth > 0 && container.clientHeight > 0) {
                        console.log('📊 Container visible, initializing visualization...');
                        this.initializeParetoVisualization();
                    } else {
                        console.warn('⚠️ Container not visible yet, will retry when tab is switched');
                    }
                }, 300);
            }
        }
    }

    /**
     * Get default weight values from revised framework
     */
    getDefaultWeights() {
        return {
            // User Experience Weights (sum = 1.0)
            w1_fee_affordability: 0.40,
            w2_fee_stability: 0.30,
            w3_fee_predictability_1h: 0.20,
            w4_fee_predictability_6h: 0.10,

            // Protocol Safety Weights (sum = 1.0)
            w5_insolvency_protection: 0.40,
            w6_deficit_duration: 0.30,
            w7_vault_stress: 0.20,
            w8_continuous_underfunding: 0.10,

            // Economic Efficiency Weights (sum = 1.0)
            w9_vault_utilization: 0.40,
            w10_deficit_correction: 0.30,
            w11_capital_efficiency: 0.30
        };
    }

    /**
     * Normalize weight names from short IDs to canonical keys
     */
    getCanonicalWeightName(weightName) {
        const map = {
            w1: 'w1_fee_affordability',
            w2: 'w2_fee_stability',
            w3: 'w3_fee_predictability_1h',
            w4: 'w4_fee_predictability_6h',
            w5: 'w5_insolvency_protection',
            w6: 'w6_deficit_duration',
            w7: 'w7_vault_stress',
            w8: 'w8_continuous_underfunding',
            w9: 'w9_vault_utilization',
            w10: 'w10_deficit_correction',
            w11: 'w11_capital_efficiency'
        };

        return map[weightName] || weightName;
    }

    /**
     * Setup methodology panel toggle and interactions
     */
    setupMethodologyPanel() {
        const methodologyToggle = document.getElementById('methodology-toggle');
        const methodologyContent = document.getElementById('methodology-content');

        if (methodologyToggle && methodologyContent) {
            // Initially expanded
            methodologyToggle.addEventListener('click', () => {
                methodologyContent.classList.toggle('collapsed');
                methodologyToggle.classList.toggle('collapsed');
            });
        }

        // Initialize MathJax if available
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([methodologyContent]).catch((err) =>
                console.warn('MathJax rendering error:', err)
            );
        }
    }

    /**
     * Setup tab switching functionality
     */
    setupTabSwitching() {
        const tabButtons = document.querySelectorAll('.research-tab-button');
        const tabContents = document.querySelectorAll('.research-tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', (event) => {
                // Ensure tab switching always works, even during loading
                event.stopPropagation();
                event.preventDefault();

                const targetTab = button.dataset.tab;

                // Force immediate tab switch (no delays that could be interfered with by loading)
                this.immediateTabSwitch(targetTab, button, tabButtons, tabContents);

                // Initialize visualization when switching to optimization tab
                if (targetTab === 'optimization') {
                    // Wait for tab content to be visible before initializing
                    setTimeout(() => {
                        const container = document.querySelector('#pareto-visualization');
                        if (container && container.clientWidth > 0 && container.clientHeight > 0) {
                            if (!this.paretoVisualizer) {
                                console.log('📊 Optimization tab now visible, initializing 3D visualization...');
                                this.initializeParetoVisualization();
                            } else if (this.paretoVisualizer.renderer &&
                                     (this.paretoVisualizer.renderer.getSize().width === 0 ||
                                      this.paretoVisualizer.renderer.getSize().height === 0)) {
                                // Visualizer exists but was created with 0×0 size, force resize
                                console.log('🔧 Resizing 3D visualization for visible container...');
                                this.paretoVisualizer.handleResize();
                            }
                        } else {
                            console.warn('⚠️ Container not visible yet during tab switch');
                        }
                    }, 200);
                }

                console.log(`🔄 Switched to ${targetTab} tab`);
            });
        });

        // Setup feedback panel toggle
        const feedbackToggle = document.getElementById('feedback-toggle');
        const feedbackContent = document.getElementById('feedback-content');

        if (feedbackToggle && feedbackContent) {
            feedbackToggle.addEventListener('click', () => {
                feedbackContent.classList.toggle('collapsed');
                feedbackToggle.classList.toggle('collapsed');
            });
        }
    }

    /**
     * Immediately switch tabs without any delays or blocking
     */
    immediateTabSwitch(targetTab, activeButton, tabButtons, tabContents) {
        // Force immediate visual feedback - update button states
        tabButtons.forEach(btn => {
            btn.classList.remove('active');
            btn.style.pointerEvents = 'auto'; // Ensure buttons stay clickable
        });
        activeButton.classList.add('active');

        // Force immediate content visibility switch
        tabContents.forEach(content => {
            content.classList.remove('active');
            content.style.display = 'none'; // Force hide immediately
            if (content.id === `${targetTab}-tab-content`) {
                content.classList.add('active');
                content.style.display = 'block'; // Force show immediately
                content.style.opacity = '1'; // Ensure visibility
            }
        });

        console.log(`🔄 Immediately switched to ${targetTab} tab`);
    }

    /**
     * Setup weight slider controls and normalization
     */
    setupWeightSliders() {
        const categories = ['ux', 'safety', 'efficiency'];

        categories.forEach(category => {
            const sliders = document.querySelectorAll(`[data-category="${category}"] .weight-slider`);
            console.log(`🎛️ Found ${sliders.length} sliders for category: ${category}`);

            sliders.forEach(slider => {
                const originalName = slider.dataset.weight;
                const weightName = this.getCanonicalWeightName(originalName);

                // Normalize data-weight so all downstream logic uses canonical keys
                slider.dataset.weight = weightName;

                // Align lock button data attributes with canonical key
                const lockBtn = slider.parentElement?.querySelector('.lock-weight, .weight-lock, .pin-weight');
                if (lockBtn) {
                    lockBtn.dataset.weight = weightName;
                }

                if (this.currentWeights[weightName] !== undefined) {
                    slider.value = this.currentWeights[weightName];
                    this.updateSliderDisplay(slider);
                } else {
                    console.warn(`Warning: Weight ${weightName} not found in currentWeights`);
                }

                // Add event listeners
                slider.addEventListener('input', () => {
                    this.handleWeightChange(slider, category);
                });

                slider.addEventListener('change', () => {
                    this.normalizeWeightsWithLocks(category, weightName);
                    this.updateLiveFormulas();
                    this.updateFormulaDisplay();
                    this.updateWeightsAndEngine();
                });
            });

            // Lock buttons for the category
            const lockButtons = document.querySelectorAll(`[data-category="${category}"] .pin-weight, [data-category="${category}"] .weight-lock, [data-category="${category}"] .lock-weight`);
            lockButtons.forEach(btn => {
                btn.addEventListener('click', () => this.toggleWeightLock(btn, category));
            });
        });

        console.log('🎛️ Current weights:', this.currentWeights);
    }

    /**
     * Setup weight preset buttons functionality
     */
    setupWeightPresets() {
        // Define weight presets for different optimization strategies
        const weightPresets = {
            'balanced': {
                name: '⚖️ Balanced',
                description: 'Equal emphasis on all three objectives',
                weights: {
                    // User Experience (emphasis on affordability and stability)
                    w1_fee_affordability: 0.35,
                    w2_fee_stability: 0.35,
                    w3_fee_predictability_1h: 0.20,
                    w4_fee_predictability_6h: 0.10,

                    // Protocol Safety (emphasis on insolvency protection)
                    w5_insolvency_protection: 0.40,
                    w6_deficit_duration: 0.25,
                    w7_vault_stress: 0.25,
                    w8_continuous_underfunding: 0.10,

                    // Economic Efficiency (balanced across all metrics)
                    w9_vault_utilization: 0.35,
                    w10_deficit_correction: 0.35,
                    w11_capital_efficiency: 0.30
                }
            },
            'ux-focused': {
                name: '👥 UX Focused',
                description: 'Prioritizes user experience - lower fees and predictability',
                weights: {
                    // User Experience (maximum emphasis on affordability)
                    w1_fee_affordability: 0.60,
                    w2_fee_stability: 0.25,
                    w3_fee_predictability_1h: 0.10,
                    w4_fee_predictability_6h: 0.05,

                    // Protocol Safety (minimum viable safety)
                    w5_insolvency_protection: 0.70,
                    w6_deficit_duration: 0.15,
                    w7_vault_stress: 0.10,
                    w8_continuous_underfunding: 0.05,

                    // Economic Efficiency (focus on capital efficiency)
                    w9_vault_utilization: 0.20,
                    w10_deficit_correction: 0.20,
                    w11_capital_efficiency: 0.60
                }
            },
            'safety-focused': {
                name: '🛡️ Safety Focused',
                description: 'Prioritizes protocol safety and stability above all',
                weights: {
                    // User Experience (minimal emphasis, basic affordability)
                    w1_fee_affordability: 0.25,
                    w2_fee_stability: 0.40,
                    w3_fee_predictability_1h: 0.25,
                    w4_fee_predictability_6h: 0.10,

                    // Protocol Safety (maximum emphasis on all safety metrics)
                    w5_insolvency_protection: 0.35,
                    w6_deficit_duration: 0.30,
                    w7_vault_stress: 0.25,
                    w8_continuous_underfunding: 0.10,

                    // Economic Efficiency (focus on vault stability)
                    w9_vault_utilization: 0.50,
                    w10_deficit_correction: 0.30,
                    w11_capital_efficiency: 0.20
                }
            },
            'efficiency-focused': {
                name: '💰 Efficiency Focused',
                description: 'Maximizes economic efficiency and capital utilization',
                weights: {
                    // User Experience (efficiency-oriented - stable fees)
                    w1_fee_affordability: 0.20,
                    w2_fee_stability: 0.40,
                    w3_fee_predictability_1h: 0.25,
                    w4_fee_predictability_6h: 0.15,

                    // Protocol Safety (efficient safety measures)
                    w5_insolvency_protection: 0.30,
                    w6_deficit_duration: 0.20,
                    w7_vault_stress: 0.30,
                    w8_continuous_underfunding: 0.20,

                    // Economic Efficiency (maximum emphasis on all efficiency metrics)
                    w9_vault_utilization: 0.25,
                    w10_deficit_correction: 0.35,
                    w11_capital_efficiency: 0.40
                }
            }
        };

        // Add click handlers to preset buttons
        const presetButtons = document.querySelectorAll('.quick-presets .preset-btn');
        presetButtons.forEach(button => {
            button.addEventListener('click', () => {
                const presetKey = button.dataset.preset;
                const preset = weightPresets[presetKey];

                if (preset) {
                    this.applyWeightPreset(preset);
                    this.setActivePresetButton(button, presetButtons);
                    console.log(`🎯 Applied weight preset: ${preset.name}`);
                } else {
                    console.warn(`Unknown preset: ${presetKey}`);
                }
            });
        });

        console.log('🎯 Setup weight presets for', presetButtons.length, 'buttons');
    }

    /**
     * Apply a weight preset configuration
     */
    applyWeightPreset(preset) {
        // Update internal weights
        this.currentWeights = { ...preset.weights };

        // Update all sliders to reflect new weights
        Object.keys(preset.weights).forEach(weightName => {
            const slider = document.querySelector(`[data-weight="${weightName}"]`);
            if (slider) {
                slider.value = preset.weights[weightName];
                this.updateSliderDisplay(slider);
            }
        });

        // Update all category displays
        ['ux', 'safety', 'efficiency'].forEach(category => {
            this.updateCategorySliders(category);
            this.updateCategorySum(category);
        });

        // Update formulas and prepare for next optimization
        this.updateLiveFormulas();
        this.updateFormulaDisplay();
        this.updateWeightsAndEngine();

        console.log('📊 Applied preset weights:', preset.weights);
    }

    /**
     * Set active state for preset button
     */
    setActivePresetButton(activeButton, allButtons) {
        // Remove active state from all buttons
        allButtons.forEach(btn => btn.classList.remove('active'));

        // Add active state to clicked button
        activeButton.classList.add('active');
    }

    /**
     * Handle individual weight changes with live preview
     */
    handleWeightChange(slider, category) {
        const weightName = this.getCanonicalWeightName(slider.dataset.weight);
        const newValue = parseFloat(slider.value);

        this.currentWeights[weightName] = newValue;
        this.updateSliderDisplay(slider);

        // Auto-normalize respecting locks
        this.normalizeWeightsWithLocks(category, weightName);
        this.updateLiveFormulas();
        this.updateFormulaDisplay();
    }

    /**
     * Normalize weights within category to sum to 1.0 (respecting locks)
     */
    normalizeWeightsWithLocks(category, changedWeightName = null) {
        const categoryWeights = this.getCategoryWeights(category);
        const locks = this.lockedWeights[category] || new Set();

        // Enforce lock limit (N-2)
        if (locks.size > categoryWeights.length - 2) {
            const first = locks.values().next().value;
            if (first) locks.delete(first);
        }

        const lockedSum = categoryWeights
            .filter(w => locks.has(w))
            .reduce((sum, w) => sum + this.currentWeights[w], 0);

        // Cap changed weight if it exceeds available budget
        if (changedWeightName && !locks.has(changedWeightName)) {
            const maxAllowed = Math.max(0, 1 - lockedSum);
            if (this.currentWeights[changedWeightName] > maxAllowed) {
                this.currentWeights[changedWeightName] = maxAllowed;
                const slider = document.querySelector(`[data-weight="${changedWeightName}"]`);
                if (slider) {
                    slider.value = maxAllowed;
                    this.updateSliderDisplay(slider);
                }
            }
        }

        const changedValue = changedWeightName ? this.currentWeights[changedWeightName] : 0;
        const remainingNames = categoryWeights.filter(w => !locks.has(w) && w !== changedWeightName);
        const remainingSum = remainingNames.reduce((sum, w) => sum + this.currentWeights[w], 0);
        const targetRemaining = Math.max(0, 1 - lockedSum - changedValue);

        if (remainingNames.length > 0) {
            if (remainingSum === 0) {
                const evenShare = targetRemaining / remainingNames.length;
                remainingNames.forEach(w => {
                    this.currentWeights[w] = evenShare;
                });
            } else {
                const scale = targetRemaining / remainingSum;
                remainingNames.forEach(w => {
                    this.currentWeights[w] = this.currentWeights[w] * scale;
                });
            }
        }

        this.updateCategorySliders(category);
        this.updateCategorySum(category);
    }

    /**
     * Get weight names for a category
     */
    getCategoryWeights(category) {
        const weights = {
            'ux': ['w1_fee_affordability', 'w2_fee_stability', 'w3_fee_predictability_1h', 'w4_fee_predictability_6h'],
            'safety': ['w5_insolvency_protection', 'w6_deficit_duration', 'w7_vault_stress', 'w8_continuous_underfunding'],
            'efficiency': ['w9_vault_utilization', 'w10_deficit_correction', 'w11_capital_efficiency']
        };
        return weights[category] || [];
    }

    /**
     * Update slider display values
     */
    updateSliderDisplay(slider) {
        const value = parseFloat(slider.value);
        const weightName = this.getCanonicalWeightName(slider.dataset.weight);

        // Update the corresponding span element by mapping weight names to display IDs
        const weightIdMap = {
            'w1_fee_affordability': 'w1-value',
            'w2_fee_stability': 'w2-value',
            'w3_fee_predictability_1h': 'w3-value',
            'w4_fee_predictability_6h': 'w4-value',
            'w5_insolvency_protection': 'w5-value',
            'w6_deficit_duration': 'w6-value',
            'w7_vault_stress': 'w7-value',
            'w8_continuous_underfunding': 'w8-value',
            'w9_vault_utilization': 'w9-value',
            'w10_deficit_correction': 'w10-value',
            'w11_capital_efficiency': 'w11-value'
        };

        const displayId = weightIdMap[weightName];
        if (displayId) {
            const valueSpan = document.getElementById(displayId);
            if (valueSpan) {
                valueSpan.textContent = value.toFixed(3);
            }
        }

        // Legacy approach for backwards compatibility
        const display = slider.nextElementSibling;
        if (display && display.classList.contains('weight-value')) {
            display.textContent = value.toFixed(3);
        }
    }

    /**
     * Update all sliders in a category
     */
    updateCategorySliders(category) {
        const sliders = document.querySelectorAll(`[data-category="${category}"] .weight-slider`);
        sliders.forEach(slider => {
            const weightName = this.getCanonicalWeightName(slider.dataset.weight);
            slider.value = this.currentWeights[weightName];
            this.updateSliderDisplay(slider);

            const lockBtn = slider.parentElement?.querySelector('.pin-weight, .weight-lock, .lock-weight');
            const isLocked = this.lockedWeights[category]?.has(weightName);
            if (lockBtn) {
                lockBtn.textContent = isLocked ? '🔒' : '🔓';
            }
            slider.disabled = !!isLocked;
        });
    }

    /**
     * Update category sum display
     */
    updateCategorySum(category) {
        const categoryWeights = this.getCategoryWeights(category);
        const sum = categoryWeights.reduce((total, w) => total + this.currentWeights[w], 0);

        const sumDisplay = document.querySelector(`#${category}-category-sum`) ||
                           document.querySelector(`#${category}-sum`);
        if (sumDisplay) {
            sumDisplay.textContent = sum.toFixed(3);

            // Visual feedback for normalization needed
            if (Math.abs(sum - 1.0) > 0.001) {
                sumDisplay.classList.add('needs-normalization');
            } else {
                sumDisplay.classList.remove('needs-normalization');
            }
        }
    }

    /**
     * Update live LaTeX formulas in methodology panel
     */
    updateLiveFormulas() {
        const w = this.currentWeights;

        // Update UX formula
        const uxFormula = document.getElementById('ux-formula');
        if (uxFormula) {
            uxFormula.innerHTML = `$$UX = ${w.w1_fee_affordability.toFixed(3)} \\cdot M_1 + ${w.w2_fee_stability.toFixed(3)} \\cdot M_2 + ${w.w3_fee_predictability_1h.toFixed(3)} \\cdot M_3 + ${w.w4_fee_predictability_6h.toFixed(3)} \\cdot M_4$$`;
        }

        // Update Safety formula
        const safetyFormula = document.getElementById('safety-formula');
        if (safetyFormula) {
            safetyFormula.innerHTML = `$$Safety = ${w.w5_insolvency_protection.toFixed(3)} \\cdot M_5 + ${w.w6_deficit_duration.toFixed(3)} \\cdot M_6 + ${w.w7_vault_stress.toFixed(3)} \\cdot M_7 + ${w.w8_continuous_underfunding.toFixed(3)} \\cdot M_8$$`;
        }

        // Update Efficiency formula
        const efficiencyFormula = document.getElementById('efficiency-formula');
        if (efficiencyFormula) {
            efficiencyFormula.innerHTML = `$$Efficiency = ${w.w9_vault_utilization.toFixed(3)} \\cdot M_9 + ${w.w10_deficit_correction.toFixed(3)} \\cdot M_{10} + ${w.w11_capital_efficiency.toFixed(3)} \\cdot M_{11}$$`;
        }

        // Re-render MathJax if available
        if (window.MathJax && window.MathJax.typesetPromise) {
            const formulaElements = [uxFormula, safetyFormula, efficiencyFormula].filter(el => el);
            window.MathJax.typesetPromise(formulaElements).catch((err) =>
                console.warn('MathJax re-rendering error:', err)
            );
        }
    }

    /**
     * Update the mathematical formula display
     */
    updateFormulaDisplay() {
        const w = this.currentWeights;

        const uxFormula = `${w.w1_fee_affordability.toFixed(3)} × M₁ + ${w.w2_fee_stability.toFixed(3)} × M₂ + ${w.w3_fee_predictability_1h.toFixed(3)} × M₃ + ${w.w4_fee_predictability_6h.toFixed(3)} × M₄`;

        const safetyFormula = `${w.w5_insolvency_protection.toFixed(3)} × M₅ + ${w.w6_deficit_duration.toFixed(3)} × M₆ + ${w.w7_vault_stress.toFixed(3)} × M₇ + ${w.w8_continuous_underfunding.toFixed(3)} × M₈`;

        const efficiencyFormula = `${w.w9_vault_utilization.toFixed(3)} × M₉ + ${w.w10_deficit_correction.toFixed(3)} × M₁₀ + ${w.w11_capital_efficiency.toFixed(3)} × M₁₁`;

        const formulaContainer = document.querySelector('.formula-display');
        if (formulaContainer) {
            formulaContainer.innerHTML = `
                <div class="formula-section">
                    <strong>UX Score:</strong><br>
                    <code>${uxFormula}</code>
                </div>
                <div class="formula-section">
                    <strong>Safety Score:</strong><br>
                    <code>${safetyFormula}</code>
                </div>
                <div class="formula-section">
                    <strong>Efficiency Score:</strong><br>
                    <code>${efficiencyFormula}</code>
                </div>
            `;
        }
    }

    /**
     * Toggle lock for a specific weight (limit locks to N-2 per category)
     */
    toggleWeightLock(button, category) {
        const weightName = this.getCanonicalWeightName(button.dataset.weight);
        const locks = this.lockedWeights[category] || new Set();
        if (locks.has(weightName)) {
            locks.delete(weightName);
        } else {
            if (locks.size >= this.getCategoryWeights(category).length - 2) {
                console.warn('Lock limit reached; unlock another weight first.');
                return;
            }
            locks.add(weightName);
        }
        this.lockedWeights[category] = locks;
        this.updateCategorySliders(category);
        this.normalizeWeightsWithLocks(category);
        this.updateLiveFormulas();
        this.updateFormulaDisplay();
    }

    /**
     * Setup optimization control buttons
     */
    setupOptimizationControls() {
        const startBtn = document.querySelector('#start-optimization');
        const stopBtn = document.querySelector('#stop-optimization');
        const resetBtn = document.querySelector('#reset-optimization');
        const exportBtn = document.querySelector('#export-results');

        if (startBtn) {
            startBtn.addEventListener('click', () => this.startOptimization());
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopOptimization());
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetOptimization());
        }

        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }

        // Setup solutions sort dropdown
        const sortDropdown = document.querySelector('#sort-solutions');
        if (sortDropdown) {
            sortDropdown.addEventListener('change', (e) => this.sortSolutions(e.target.value));
        }

        // Setup algorithm parameter sliders
        this.setupAlgorithmParameterSliders();
    }

    /**
     * Setup algorithm parameter slider controls
     */
    setupAlgorithmParameterSliders() {
        const sliders = [
            { id: 'population-slider', valueId: 'population-value' },
            { id: 'generations-slider', valueId: 'generations-value' },
            { id: 'crossover-slider', valueId: 'crossover-value' },
            { id: 'mutation-slider', valueId: 'mutation-value' }
        ];

        sliders.forEach(({ id, valueId }) => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(valueId);

            if (slider && valueDisplay) {
                slider.addEventListener('input', (event) => {
                    const value = parseFloat(event.target.value);

                    // Update display with appropriate formatting
                    if (id === 'crossover-slider' || id === 'mutation-slider') {
                        valueDisplay.textContent = value.toFixed(2);
                    } else {
                        valueDisplay.textContent = value.toString();
                    }

                    // Log for debugging
                    console.log(`🔧 Algorithm parameter updated: ${id} = ${value}`);
                });

                // Initialize display with current value
                const currentValue = parseFloat(slider.value);
                if (id === 'crossover-slider' || id === 'mutation-slider') {
                    valueDisplay.textContent = currentValue.toFixed(2);
                } else {
                    valueDisplay.textContent = currentValue.toString();
                }
            } else {
                console.warn(`⚠️ Algorithm slider not found: ${id} or ${valueId}`);
            }
        });
    }

    /**
     * Get current algorithm parameters from sliders
     */
    getCurrentAlgorithmParameters() {
        const populationSlider = document.getElementById('population-slider');
        const generationsSlider = document.getElementById('generations-slider');
        const crossoverSlider = document.getElementById('crossover-slider');
        const mutationSlider = document.getElementById('mutation-slider');

        return {
            populationSize: populationSlider ? parseInt(populationSlider.value) : 100,
            maxGenerations: generationsSlider ? parseInt(generationsSlider.value) : 50,
            crossoverRate: crossoverSlider ? parseFloat(crossoverSlider.value) : 0.9,
            mutationRate: mutationSlider ? parseFloat(mutationSlider.value) : 0.1
        };
    }

    /**
     * Setup metric tooltip click handlers
     */
    setupMetricTooltips() {
        // Add click handlers to all metric tags
        const metricTags = document.querySelectorAll('.clickable-metric');

        metricTags.forEach(tag => {
            tag.addEventListener('click', (event) => {
                event.stopPropagation();
                const metricKey = tag.dataset.metric;
                this.showMetricTooltip(metricKey, tag);
            });

            // Add visual cue that it's clickable
            tag.style.cursor = 'pointer';
            tag.title = 'Click to see formula and explanation';
        });

        // Close tooltip when clicking outside
        document.addEventListener('click', () => {
            this.hideMetricTooltip();
        });

        console.log(`🔬 Setup click handlers for ${metricTags.length} metric tags`);
    }

    /**
     * Show detailed metric tooltip with formula and explanation
     */
    showMetricTooltip(metricKey, targetElement) {
        const metric = this.metricDefinitions[metricKey];
        if (!metric) {
            console.warn(`Metric definition not found: ${metricKey}`);
            return;
        }

        // Remove any existing tooltip
        this.hideMetricTooltip();

        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.className = 'metric-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-header">
                <h4>${metric.name}</h4>
                <button class="tooltip-close">&times;</button>
            </div>
            <div class="tooltip-content">
                <div class="formula-section">
                    <h5>Mathematical Formula:</h5>
                    <div class="formula-display" id="tooltip-formula-${metricKey}">
                        $$${metric.formula}$$
                    </div>
                </div>
                <div class="explanation-section">
                    <h5>Explanation:</h5>
                    <p>${metric.explanation}</p>
                </div>
                <div class="details-section">
                    <h5>Technical Details:</h5>
                    <ul>
                        ${metric.details.map(detail => `<li>${detail}</li>`).join('')}
                    </ul>
                </div>
                <div class="range-section">
                    <h5>Value Range:</h5>
                    <p><strong>${metric.range}</strong></p>
                </div>
            </div>
        `;

        // Position tooltip with smart positioning to keep it fully visible
        const rect = targetElement.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        // Calculate preferred position
        let top = rect.bottom + 15;
        let left = Math.max(20, rect.left - 300);

        // Adjust horizontal position if it would go off-screen
        const tooltipWidth = 600;
        if (left + tooltipWidth > viewportWidth - 20) {
            left = viewportWidth - tooltipWidth - 20;
        }

        // Adjust vertical position if it would go off-screen
        const estimatedHeight = 500; // Conservative estimate
        if (top + estimatedHeight > viewportHeight - 20) {
            // Position above the element instead
            top = Math.max(20, rect.top - estimatedHeight - 15);
        }

        tooltip.style.cssText = `
            position: fixed;
            top: ${top}px;
            left: ${left}px;
            width: ${tooltipWidth - 40}px;
            max-width: 90vw;
            max-height: 80vh;
            z-index: 10000;
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            padding: 0;
            font-family: system-ui, sans-serif;
            animation: tooltipFadeIn 0.2s ease-out;
            overflow-y: auto;
        `;

        // Add close button handler
        const closeBtn = tooltip.querySelector('.tooltip-close');
        closeBtn.addEventListener('click', () => this.hideMetricTooltip());

        document.body.appendChild(tooltip);
        this.activeMetricTooltip = tooltip;

        // Highlight corresponding formula term
        this.highlightFormulaVariable(metric.highlightText, metric.category);

        // Render LaTeX if MathJax is available
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([tooltip]).catch((err) =>
                console.warn('MathJax tooltip rendering error:', err)
            );
        }

        console.log(`📊 Showing tooltip for ${metric.name}`);
    }

    /**
     * Hide the active metric tooltip
     */
    hideMetricTooltip() {
        if (this.activeMetricTooltip) {
            this.activeMetricTooltip.remove();
            this.activeMetricTooltip = null;

            // Remove any formula highlighting
            this.clearFormulaHighlighting();
        }
    }

    /**
     * Highlight the corresponding variable in the objective formula
     */
    highlightFormulaVariable(variable, category) {
        // Clear any existing highlighting first
        this.clearFormulaHighlighting();

        // Find the formula in the corresponding objective card
        const categoryCard = document.querySelector(`.${category}-card`);
        if (!categoryCard) return;

        const formulaContainer = categoryCard.querySelector('.mathematical-formulation');
        if (!formulaContainer) return;

        // Add highlighting class to the entire formula container for background effect
        formulaContainer.classList.add('formula-highlighted');

        // Now highlight the specific term in red
        this.highlightSpecificTerm(formulaContainer, variable);

        console.log(`✨ Highlighted formula variable ${variable} in ${category} category`);
    }

    /**
     * Highlight specific mathematical term in red
     */
    highlightSpecificTerm(container, highlightText) {
        // Wait for MathJax to finish rendering, then find and highlight the term
        setTimeout(() => {
            // Find all MathJax rendered elements
            const mathElements = container.querySelectorAll('.MathJax, .MathJax_Display, [id*="MathJax"]');

            if (mathElements.length === 0) {
                // Fallback: try to find mathml elements or other math containers
                const allMathContainers = container.querySelectorAll('*');
                for (let element of allMathContainers) {
                    if (element.textContent && element.textContent.includes(highlightText)) {
                        this.applyRedHighlightToTerm(element, highlightText);
                        break;
                    }
                }
            } else {
                // Search through MathJax elements
                mathElements.forEach(mathElement => {
                    this.applyRedHighlightToTerm(mathElement, highlightText);
                });
            }
        }, 300); // Give MathJax time to render
    }

    /**
     * Apply red highlighting to specific term within an element
     */
    applyRedHighlightToTerm(element, highlightText) {
        // Try different approaches to find and highlight the text
        const textContent = element.textContent || element.innerText;

        if (textContent && textContent.includes(highlightText)) {
            // Method 1: Try to find and wrap the text directly
            const walker = document.createTreeWalker(
                element,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            let textNode;
            while (textNode = walker.nextNode()) {
                const nodeText = textNode.textContent;
                if (nodeText.includes(highlightText)) {
                    const span = document.createElement('span');
                    span.className = 'formula-term-highlight';
                    span.style.cssText = 'color: #ef4444 !important; font-weight: bold; background: rgba(239, 68, 68, 0.1); padding: 1px 3px; border-radius: 3px;';

                    const highlightedText = nodeText.replace(
                        new RegExp(highlightText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'),
                        `<mark style="background: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold; padding: 1px 3px; border-radius: 3px;">$&</mark>`
                    );

                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = highlightedText;

                    while (tempDiv.firstChild) {
                        textNode.parentNode.insertBefore(tempDiv.firstChild, textNode);
                    }
                    textNode.remove();
                    break;
                }
            }
        }
    }

    /**
     * Clear all formula highlighting
     */
    clearFormulaHighlighting() {
        // Remove background highlighting
        const highlightedFormulas = document.querySelectorAll('.formula-highlighted');
        highlightedFormulas.forEach(formula => {
            formula.classList.remove('formula-highlighted');
        });

        // Remove red term highlighting
        const highlightedTerms = document.querySelectorAll('.formula-term-highlight, mark[style*="color: #ef4444"]');
        highlightedTerms.forEach(term => {
            // Replace the highlighted element with its text content
            const parent = term.parentNode;
            if (parent) {
                parent.replaceChild(document.createTextNode(term.textContent), term);
                parent.normalize(); // Merge adjacent text nodes
            }
        });

        // Force MathJax to re-render to clean up any highlighting artifacts
        if (window.MathJax && window.MathJax.typesetPromise) {
            const formulaContainers = document.querySelectorAll('.mathematical-formulation');
            setTimeout(() => {
                window.MathJax.typesetPromise(formulaContainers).catch((err) =>
                    console.warn('MathJax re-render error:', err)
                );
            }, 100);
        }
    }

    /**
     * Setup animation controls panel and event handlers
     */
    setupAnimationControls() {
        const animationBtn = document.querySelector('#animation-settings');
        const animationPanel = document.querySelector('#animation-controls-panel');
        const closeBtn = document.querySelector('#close-animation-panel');

        // Toggle animation panel
        if (animationBtn && animationPanel) {
            animationBtn.addEventListener('click', () => {
                animationPanel.classList.toggle('hidden');
            });
        }

        // Close animation panel
        if (closeBtn && animationPanel) {
            closeBtn.addEventListener('click', () => {
                animationPanel.classList.add('hidden');
            });
        }

        // Animation controls
        this.setupAnimationEventHandlers();
    }

    /**
     * Setup individual animation control event handlers
     */
    setupAnimationEventHandlers() {
        // Animation enabled toggle
        const enabledCheckbox = document.querySelector('#animation-enabled');
        if (enabledCheckbox) {
            enabledCheckbox.addEventListener('change', (e) => {
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.toggleAnimation('enabled', e.target.checked);
                }
            });
        }

        // Animation speed slider
        const speedSlider = document.querySelector('#animation-speed');
        const speedValue = document.querySelector('#speed-value');
        if (speedSlider && speedValue) {
            speedSlider.addEventListener('input', (e) => {
                const speed = parseFloat(e.target.value);
                speedValue.textContent = `${speed.toFixed(1)}x`;
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.setAnimationSpeed(speed);
                }
            });
        }

        // Rotation enabled toggle
        const rotationCheckbox = document.querySelector('#rotation-enabled');
        if (rotationCheckbox) {
            rotationCheckbox.addEventListener('change', (e) => {
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.toggleAnimation('rotationEnabled', e.target.checked);
                }
            });
        }

        // Pulse enabled toggle
        const pulseCheckbox = document.querySelector('#pulse-enabled');
        if (pulseCheckbox) {
            pulseCheckbox.addEventListener('change', (e) => {
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.toggleAnimation('pulseEnabled', e.target.checked);
                }
            });
        }

        // Auto-rotate camera toggle
        const autoRotateCheckbox = document.querySelector('#auto-rotate-camera');
        const cameraSpeedGroup = document.querySelector('#camera-speed-group');
        if (autoRotateCheckbox) {
            autoRotateCheckbox.addEventListener('change', (e) => {
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.toggleAnimation('autoRotateCamera', e.target.checked);
                }
                // Show/hide camera speed controls
                if (cameraSpeedGroup) {
                    cameraSpeedGroup.style.display = e.target.checked ? 'block' : 'none';
                }
            });
        }

        // Camera speed slider
        const cameraSpeedSlider = document.querySelector('#camera-speed');
        const cameraSpeedValue = document.querySelector('#camera-speed-value');
        if (cameraSpeedSlider && cameraSpeedValue) {
            cameraSpeedSlider.addEventListener('input', (e) => {
                const speed = parseFloat(e.target.value);
                cameraSpeedValue.textContent = `${speed.toFixed(1)}x`;
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.setAnimationSettings({ cameraSpeed: speed });
                }
            });
        }

        // Trail enabled toggle
        const trailCheckbox = document.querySelector('#trail-enabled');
        if (trailCheckbox) {
            trailCheckbox.addEventListener('change', (e) => {
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.toggleAnimation('trailEnabled', e.target.checked);
                }
            });
        }

        // Reset animation settings
        const resetBtn = document.querySelector('#reset-animation');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetAnimationSettings());
        }

        // Reset camera position
        const resetCameraBtn = document.querySelector('#reset-camera');
        if (resetCameraBtn) {
            resetCameraBtn.addEventListener('click', () => {
                if (this.paretoVisualizer) {
                    this.paretoVisualizer.resetCameraPosition();
                }
            });
        }
    }

    /**
     * Reset animation settings to defaults
     */
    resetAnimationSettings() {
        const defaultSettings = {
            enabled: true,
            speed: 1.0,
            rotationEnabled: true,
            pulseEnabled: true,
            trailEnabled: false,
            autoRotateCamera: false,
            cameraSpeed: 0.5
        };

        // Update UI controls
        document.querySelector('#animation-enabled').checked = defaultSettings.enabled;
        document.querySelector('#animation-speed').value = defaultSettings.speed;
        document.querySelector('#speed-value').textContent = `${defaultSettings.speed.toFixed(1)}x`;
        document.querySelector('#rotation-enabled').checked = defaultSettings.rotationEnabled;
        document.querySelector('#pulse-enabled').checked = defaultSettings.pulseEnabled;
        document.querySelector('#trail-enabled').checked = defaultSettings.trailEnabled;
        document.querySelector('#auto-rotate-camera').checked = defaultSettings.autoRotateCamera;
        document.querySelector('#camera-speed').value = defaultSettings.cameraSpeed;
        document.querySelector('#camera-speed-value').textContent = `${defaultSettings.cameraSpeed.toFixed(1)}x`;
        document.querySelector('#camera-speed-group').style.display = 'none';

        // Apply to visualizer
        if (this.paretoVisualizer) {
            this.paretoVisualizer.setAnimationSettings(defaultSettings);
        }
    }

    /**
     * Setup 3D visualization container (placeholder only)
     */
    setupVisualization() {
        const container = document.querySelector('#pareto-visualization');
        if (container) {
            // Clear any existing content
            container.innerHTML = '';

            // Only show placeholder - actual initialization happens when tab is visible
            container.innerHTML = `
                <div class="visualization-placeholder">
                    <div class="placeholder-content">
                        <div class="placeholder-icon">📊</div>
                        <h3>3D Visualization Ready</h3>
                        <p>Switch to the Optimization tab to initialize the Pareto frontier display</p>
                    </div>
                </div>
            `;

            console.log('📦 Visualization container prepared (initialization deferred)');
        }
    }

    /**
     * Initialize Three.js Pareto visualization with interaction callbacks
     */
    initializeParetoVisualization() {
        console.log('🔧 Attempting to initialize ParetoVisualizer...');
        console.log('ParetoVisualizer available:', typeof ParetoVisualizer !== 'undefined');
        console.log('THREE available:', typeof THREE !== 'undefined');
        console.log('window.ParetoVisualizer:', !!window.ParetoVisualizer);

        const container = document.querySelector('#pareto-visualization');
        console.log('Container found:', !!container);
        if (container) {
            console.log('Container dimensions:', container.clientWidth, 'x', container.clientHeight);
        }

        if (typeof ParetoVisualizer !== 'undefined' && window.ParetoVisualizer) {
            try {
                console.log('✅ Initializing ParetoVisualizer');
                this.paretoVisualizer = new ParetoVisualizer('#pareto-visualization');

                // Check if initialization was successful
                if (this.paretoVisualizer && this.paretoVisualizer.renderer) {
                    console.log('✅ ParetoVisualizer renderer created successfully');

                    // Set up interaction callbacks
                    this.paretoVisualizer.setCallbacks({
                        onPointSelected: (solution) => this.handlePointSelection(solution),
                        onPointHovered: (solution) => this.handlePointHover(solution)
                    });

                    // 3D plot starts empty - users must run optimization to see results

                    console.log('🎯 Interactive 3D Pareto visualization initialized');
                } else {
                    throw new Error('ParetoVisualizer failed to initialize properly');
                }
            } catch (error) {
                console.error('❌ Error initializing ParetoVisualizer:', error);
                console.error('Error stack:', error.stack);
                this.showVisualizationError(error.message);
            }
        } else {
            console.warn('ParetoVisualizer not available');
            console.warn('typeof ParetoVisualizer:', typeof ParetoVisualizer);
            console.warn('window.ParetoVisualizer:', !!window.ParetoVisualizer);
            this.showVisualizationError('ParetoVisualizer class not found. Check script loading order.');
        }
    }


    /**
     * Show visualization error in the container
     */
    showVisualizationError(errorMessage) {
        const container = document.querySelector('#pareto-visualization');
        if (container) {
            container.innerHTML = `
                <div class="visualization-error">
                    <div class="error-content">
                        <div class="error-icon">⚠️</div>
                        <h3>Visualization Error</h3>
                        <p>${errorMessage}</p>
                        <div class="error-details">
                            <p><strong>Debug Info:</strong></p>
                            <ul>
                                <li>THREE.js available: ${typeof THREE !== 'undefined'}</li>
                                <li>ParetoVisualizer available: ${typeof ParetoVisualizer !== 'undefined'}</li>
                                <li>Container found: ${!!container}</li>
                            </ul>
                        </div>
                        <button onclick="location.reload()" class="btn-secondary">Reload Page</button>
                    </div>
                </div>
            `;
        }
    }

    /**
     * Handle point selection in 3D visualization
     */
    handlePointSelection(solution, source = '3d') {
        console.log('🎯 Point selected:', solution, 'from:', source);

        // Update research feedback with selected solution details
        this.highlightSolutionInList(solution);
        this.updateSelectedSolutionMetrics(solution);

        // If selection comes from list, also highlight the 3D point
        if (source === 'list' && this.paretoVisualizer) {
            this.paretoVisualizer.selectPointBySolution(solution);
        }
    }

    /**
     * Handle point hover in 3D visualization
     */
    handlePointHover(solution) {
        // Could update real-time preview or highlight in solutions list
        this.previewSolutionMetrics(solution);
    }

    /**
     * Highlight solution in solutions list
     */
    highlightSolutionInList(solution) {
        const solutionsList = document.querySelector('.solutions-list');
        if (!solutionsList) return;

        // Remove previous highlights
        solutionsList.querySelectorAll('.solution-item').forEach(item => {
            item.classList.remove('selected');
        });

        // Find and highlight matching solution
        const solutionItems = solutionsList.querySelectorAll('.solution-item');
        solutionItems.forEach(item => {
            const params = item.querySelector('.solution-params strong');
            if (params && params.textContent.includes(`μ=${solution.parameters?.mu?.toFixed(3) || solution.mu?.toFixed(3)}`)) {
                item.classList.add('selected');
                // Note: Removed scrollIntoView to prevent unwanted scrolling to duplicates
            }
        });
    }

    /**
     * Update metrics display for selected solution
     */
    updateSelectedSolutionMetrics(solution) {
        // Handle both parameter formats
        const mu = solution.parameters?.mu || solution.mu;
        const nu = solution.parameters?.nu || solution.nu;
        const H = solution.parameters?.horizon || solution.H;
        const uxScore = solution.scores?.ux || solution.uxScore;
        const safetyScore = solution.scores?.safety || solution.safetyScore;
        const efficiencyScore = solution.scores?.efficiency || solution.efficiencyScore;

        // Update parameter patterns section
        const muZeroRatio = document.getElementById('mu-zero-ratio');
        const avgNu = document.getElementById('avg-nu');
        const commonH = document.getElementById('common-h');

        if (muZeroRatio) {
            muZeroRatio.textContent = mu === 0 ? '✓ Zero' : mu.toFixed(3);
        }
        if (avgNu) {
            avgNu.textContent = nu.toFixed(3);
        }
        if (commonH) {
            commonH.textContent = H.toString();
        }

        // Update objective score ranges to highlight this solution
        this.highlightObjectiveScore('ux', uxScore);
        this.highlightObjectiveScore('safety', safetyScore);
        this.highlightObjectiveScore('efficiency', efficiencyScore);

        // Display detailed solution panel
        this.displaySolutionDetails(solution);
    }

    /**
     * Preview solution metrics on hover (lighter feedback)
     */
    previewSolutionMetrics(solution) {
        // Could add subtle visual feedback here
        console.log('👀 Previewing solution:', solution.mu, solution.nu, solution.H);
    }

    /**
     * Highlight specific objective score
     */
    highlightObjectiveScore(objective, score) {
        const minSpan = document.getElementById(`${objective}-min`);
        const maxSpan = document.getElementById(`${objective}-max`);

        if (minSpan && maxSpan) {
            // Temporarily show the selected solution's score
            const originalMin = minSpan.textContent;
            const originalMax = maxSpan.textContent;

            minSpan.textContent = score.toFixed(3);
            maxSpan.textContent = score.toFixed(3);
            minSpan.style.fontWeight = 'bold';
            maxSpan.style.fontWeight = 'bold';

            // Reset after 2 seconds
            setTimeout(() => {
                minSpan.textContent = originalMin;
                maxSpan.textContent = originalMax;
                minSpan.style.fontWeight = 'normal';
                maxSpan.style.fontWeight = 'normal';
            }, 2000);
        }
    }

    /**
     * Export individual solution data
     */
    exportSolution(solutionDataString) {
        try {
            const solution = JSON.parse(solutionDataString.replace(/&quot;/g, '"'));

            const exportData = {
                solution: solution,
                weights: this.currentWeights,
                timestamp: new Date().toISOString(),
                type: 'individual_solution'
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `taiko-solution-mu${solution.mu.toFixed(3)}-nu${solution.nu.toFixed(3)}-H${solution.H}-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            console.log('📁 Individual solution exported');
        } catch (error) {
            console.error('Failed to export solution:', error);
        }
    }

    /**
     * Display detailed information for selected solution
     */
    displaySolutionDetails(solution) {
        const detailsContainer = document.getElementById('solution-details');
        if (!detailsContainer) {
            console.warn('Solution details container not found');
            return;
        }

        // Extract solution data with fallbacks
        const mu = solution.parameters?.mu || solution.mu || 0;
        const nu = solution.parameters?.nu || solution.nu || 0;
        const H = solution.parameters?.horizon || solution.H || 0;

        const uxScore = solution.scores?.ux || solution.uxScore || 0;
        const safetyScore = solution.scores?.safety || solution.safetyScore || 0;
        const efficiencyScore = solution.scores?.efficiency || solution.efficiencyScore || 0;

        // Calculate overall score (weighted average)
        const overallScore = (uxScore + safetyScore + efficiencyScore) / 3;

        // Generate solution details HTML
        const detailsHTML = `
            <div class="solution-header">
                <div class="solution-title">
                    <h4>Selected Solution</h4>
                    <div class="solution-id">ID: ${solution.id || Math.floor(Date.now() / 1000)}</div>
                </div>
                <div class="overall-score">
                    <div class="score-value">${overallScore.toFixed(3)}</div>
                    <div class="score-label">Overall Score</div>
                </div>
            </div>

            <div class="solution-content">
                <div class="parameters-section">
                    <h5>🎛️ Parameters</h5>
                    <div class="parameter-grid">
                        <div class="parameter-item">
                            <span class="param-label">μ (L1 Weight)</span>
                            <span class="param-value">${mu.toFixed(4)}</span>
                        </div>
                        <div class="parameter-item">
                            <span class="param-label">ν (Deficit Weight)</span>
                            <span class="param-value">${nu.toFixed(4)}</span>
                        </div>
                        <div class="parameter-item">
                            <span class="param-label">H (Horizon)</span>
                            <span class="param-value">${H} steps</span>
                        </div>
                    </div>
                </div>

                <div class="objectives-section">
                    <h5>🎯 Objective Scores</h5>
                    <div class="objective-scores">
                        <div class="objective-item ux">
                            <div class="objective-header">
                                <span class="objective-icon">👥</span>
                                <span class="objective-name">User Experience</span>
                            </div>
                            <div class="objective-score">${uxScore.toFixed(3)}</div>
                            <div class="score-bar">
                                <div class="score-fill ux-fill" style="width: ${Math.max(5, Math.min(100, (uxScore + 1) * 50))}%"></div>
                            </div>
                        </div>

                        <div class="objective-item safety">
                            <div class="objective-header">
                                <span class="objective-icon">🛡️</span>
                                <span class="objective-name">Protocol Safety</span>
                            </div>
                            <div class="objective-score">${safetyScore.toFixed(3)}</div>
                            <div class="score-bar">
                                <div class="score-fill safety-fill" style="width: ${Math.max(5, Math.min(100, (safetyScore + 1) * 50))}%"></div>
                            </div>
                        </div>

                        <div class="objective-item efficiency">
                            <div class="objective-header">
                                <span class="objective-icon">💰</span>
                                <span class="objective-name">Economic Efficiency</span>
                            </div>
                            <div class="objective-score">${efficiencyScore.toFixed(3)}</div>
                            <div class="score-bar">
                                <div class="score-fill efficiency-fill" style="width: ${Math.max(5, Math.min(100, (efficiencyScore + 1) * 50))}%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="actions-section">
                    <button class="btn btn-primary" onclick="alert('Apply parameters: μ=${mu.toFixed(3)}, ν=${nu.toFixed(3)}, H=${H}')">
                        📋 Apply Parameters
                    </button>
                    <button class="btn btn-secondary" onclick="alert('Export feature coming soon!')">
                        💾 Export Solution
                    </button>
                </div>
            </div>
        `;

        detailsContainer.innerHTML = detailsHTML;

        console.log('📋 Displayed solution details for:', {
            mu: mu.toFixed(3),
            nu: nu.toFixed(3),
            H,
            scores: { ux: uxScore.toFixed(3), safety: safetyScore.toFixed(3), efficiency: efficiencyScore.toFixed(3) }
        });
    }

    /**
     * Start the optimization process
     */
    async startOptimization() {
        if (this.isOptimizing) return;

        this.isOptimizing = true;
        this.updateOptimizationUI(true);

        try {
            // Get current algorithm parameters from sliders
            const algorithmParams = this.getCurrentAlgorithmParameters();

            // Initialize or reinitialize optimization engine with current parameters
            if (window.NSGAII) {
                this.optimizationEngine = new NSGAII({
                    populationSize: algorithmParams.populationSize,
                    maxGenerations: algorithmParams.maxGenerations,
                    crossoverRate: algorithmParams.crossoverRate,
                    mutationRate: algorithmParams.mutationRate,
                    weights: this.currentWeights,
                    onProgress: (progress) => this.handleOptimizationProgress(progress),
                    onSolution: (solution) => this.handleNewSolution(solution),
                    onComplete: (results) => this.handleOptimizationComplete(results)
                });

                console.log('🧬 NSGA-II initialized with parameters:', algorithmParams);
            }

            // Initialize visualization if needed
            if (!this.paretoVisualizer) {
                this.initializeParetoVisualization();
            }

            console.log('🚀 Starting NSGA-II optimization with weights:', this.currentWeights);

            // Start real NSGA-II optimization
            if (this.optimizationEngine) {
                await this.optimizationEngine.start();
            } else {
                throw new Error('NSGA-II optimization engine not available');
            }

        } catch (error) {
            console.error('Optimization failed:', error);
            this.stopOptimization();
        }
    }

    /**
     * Handle optimization progress updates from NSGA-II
     */
    handleOptimizationProgress(progress) {
        console.log(`📊 Generation ${progress.generation}/${progress.maxGenerations}, Pareto Front: ${progress.paretoFrontSize}`);

        // Update progress bar
        this.updateProgress(progress.generation, progress.maxGenerations);

        // Update progress stats
        const solutionsCount = document.querySelector('#solutions-count');
        const paretoCount = document.querySelector('#pareto-count');
        const currentGeneration = document.querySelector('#current-generation');
        const totalGenerations = document.querySelector('#total-generations');

        if (solutionsCount) solutionsCount.textContent = progress.populationSize;
        if (paretoCount) paretoCount.textContent = progress.paretoFrontSize;
        if (currentGeneration) currentGeneration.textContent = progress.generation;
        if (totalGenerations) totalGenerations.textContent = progress.maxGenerations;
    }

    /**
     * Handle new solution from NSGA-II
     */
    handleNewSolution(solution) {
        console.log('🧬 New Pareto solution found:', {
            mu: solution.mu.toFixed(3),
            nu: solution.nu.toFixed(3),
            H: solution.H,
            ux: solution.uxScore.toFixed(3),
            safety: solution.safetyScore.toFixed(3),
            efficiency: solution.efficiencyScore.toFixed(3)
        });

        // Add to current solutions
        this.currentSolutions.push({
            id: solution.id || Date.now() + Math.random(),
            mu: solution.mu,
            nu: solution.nu,
            H: solution.H,
            uxScore: solution.uxScore,
            safetyScore: solution.safetyScore,
            efficiencyScore: solution.efficiencyScore,
            isParetoOptimal: solution.rank === 0,
            generation: solution.generation
        });

        // Add to visualization
        if (this.paretoVisualizer) {
            this.paretoVisualizer.addSolution({
                uxScore: solution.uxScore,
                safetyScore: solution.safetyScore,
                efficiencyScore: solution.efficiencyScore,
                isParetoOptimal: solution.rank === 0,
                mu: solution.mu,
                nu: solution.nu,
                H: solution.H,
                generation: solution.generation
            });
        }

        // Update solutions list (show last 20 solutions)
        this.updateSolutionsList(this.currentSolutions.slice(-20));
    }

    /**
     * Handle optimization completion from NSGA-II
     */
    handleOptimizationComplete(results) {
        console.log('✅ NSGA-II optimization completed:', {
            generations: results.generation,
            totalSolutions: results.population.length,
            paretoSolutions: results.paretoFront.length
        });

        this.isOptimizing = false;
        this.updateOptimizationUI(false);

        // Update final stats
        const progressStatus = document.querySelector('#progress-status');
        if (progressStatus) {
            progressStatus.textContent = `Complete (${results.paretoFront.length} Pareto solutions)`;
        }

        // Show final Pareto front in solutions list
        const paretoSolutions = results.paretoFront.map(solution => ({
            id: solution.id || Date.now() + Math.random(),
            mu: solution.mu,
            nu: solution.nu,
            H: solution.H,
            uxScore: solution.uxScore,
            safetyScore: solution.safetyScore,
            efficiencyScore: solution.efficiencyScore,
            isParetoOptimal: true,
            generation: solution.generation || results.generation
        }));

        this.currentSolutions = [...this.currentSolutions, ...paretoSolutions];
        this.updateSolutionsList(paretoSolutions);
    }

    /**
     * Stop the optimization process
     */
    stopOptimization() {
        this.isOptimizing = false;
        this.updateOptimizationUI(false);

        if (this.optimizationEngine) {
            this.optimizationEngine.stop();
        }

        const progressStatus = document.querySelector('#progress-status');
        if (progressStatus) {
            progressStatus.textContent = 'Stopped';
        }

        console.log('⏹️ Optimization stopped');
    }

    /**
     * Reset the optimization
     */
    resetOptimization() {
        this.stopOptimization();
        this.currentSolutions = [];

        if (this.paretoVisualizer) {
            this.paretoVisualizer.clear();
        }

        this.updateSolutionsList([]);
        this.updateProgress(0, 0);

        console.log('🔄 Optimization reset');
    }

    /**
     * Update UI during optimization
     */
    updateOptimizationUI(isRunning) {
        const startBtn = document.querySelector('#start-optimization');
        const stopBtn = document.querySelector('#stop-optimization');
        const exportBtn = document.querySelector('#export-results');
        const progress = document.querySelector('#progress-fill');

        if (startBtn) {
            startBtn.disabled = isRunning;
            startBtn.textContent = isRunning ? 'Optimizing...' : 'Start Optimization';
        }

        if (stopBtn) {
            stopBtn.disabled = !isRunning;
        }

        if (exportBtn) {
            exportBtn.disabled = isRunning || this.currentSolutions.length === 0;
        }

        if (progress && !isRunning) {
            progress.style.width = '0%';
        }
    }

    /**
     * Update weights and reinitialize optimization engine if needed
     */
    updateWeightsAndEngine() {
        // Reset optimization engine to use new weights
        if (this.optimizationEngine && !this.isOptimizing) {
            this.optimizationEngine = null;
        }

        console.log('📊 Weights updated, optimization engine will be reinitialized on next run');
    }

    /**
     * Update progress display
     */
    updateProgress(current, total) {
        const percentage = (current / total) * 100;
        const progressFill = document.querySelector('#progress-fill');
        const progressText = document.querySelector('#progress-status');

        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }

        if (progressText) {
            progressText.textContent = `Generation ${current}/${total} (${percentage.toFixed(1)}%)`;
        }
    }

    /**
     * Update solutions list display
     */
    updateSolutionsList(solutions) {
        const container = document.querySelector('#solutions-list');
        if (!container) return;

        if (solutions.length === 0) {
            container.innerHTML = '<p>No solutions yet. Start optimization to generate Pareto optimal parameters.</p>';
            return;
        }

        const solutionsHTML = solutions.map((solution, index) => `
            <div class="solution-item ${solution.isParetoOptimal ? 'pareto-optimal' : ''}" data-solution-index="${index}">
                <div class="solution-params">
                    <strong>μ=${solution.mu.toFixed(3)}, ν=${solution.nu.toFixed(3)}, H=${solution.H}</strong>
                </div>
                <div class="solution-scores">
                    <span class="score ux">UX: ${solution.uxScore.toFixed(3)}</span>
                    <span class="score safety">Safety: ${solution.safetyScore.toFixed(3)}</span>
                    <span class="score efficiency">Efficiency: ${solution.efficiencyScore.toFixed(3)}</span>
                </div>
                ${solution.isParetoOptimal ? '<div class="pareto-badge">Pareto Optimal</div>' : ''}
            </div>
        `).join('');

        container.innerHTML = solutionsHTML;

        // Add click handlers to solution items for selection
        container.querySelectorAll('.solution-item').forEach((item, index) => {
            item.addEventListener('click', () => {
                const solution = solutions[index];
                if (solution) {
                    // Remove previous selections
                    container.querySelectorAll('.solution-item').forEach(el => el.classList.remove('selected'));
                    // Mark this item as selected
                    item.classList.add('selected');
                    // Handle selection
                    this.handlePointSelection(solution, 'list');
                    console.log('📊 Solution selected from list:', solution);
                }
            });

            // Add visual feedback for clickability
            item.style.cursor = 'pointer';
            item.title = 'Click to view solution details';
        });

        console.log('📊 Updated solutions list with', solutions.length, 'solutions and click handlers');
    }

    /**
     * Sort solutions by specified criteria
     */
    sortSolutions(sortBy) {
        const container = document.querySelector('#solutions-list');
        if (!container) return;

        // Get current solutions from the DOM
        const solutionItems = Array.from(container.querySelectorAll('.solution-item'));
        if (solutionItems.length === 0) return;

        // Extract solution data from DOM elements
        const solutionsWithElements = solutionItems.map(item => {
            const paramsText = item.querySelector('.solution-params strong').textContent;
            const uxText = item.querySelector('.score.ux').textContent;
            const safetyText = item.querySelector('.score.safety').textContent;
            const efficiencyText = item.querySelector('.score.efficiency').textContent;

            // Parse parameters from text like "μ=0.123, ν=0.456, H=72"
            const muMatch = paramsText.match(/μ=([\d.]+)/);
            const nuMatch = paramsText.match(/ν=([\d.]+)/);
            const hMatch = paramsText.match(/H=(\d+)/);

            return {
                element: item,
                mu: muMatch ? parseFloat(muMatch[1]) : 0,
                nu: nuMatch ? parseFloat(nuMatch[1]) : 0,
                H: hMatch ? parseInt(hMatch[1]) : 0,
                uxScore: parseFloat(uxText.replace('UX: ', '')),
                safetyScore: parseFloat(safetyText.replace('Safety: ', '')),
                efficiencyScore: parseFloat(efficiencyText.replace('Efficiency: ', '')),
                isParetoOptimal: item.classList.contains('pareto-optimal')
            };
        });

        // Sort based on criteria
        let sortedSolutions;
        switch (sortBy) {
            case 'dominance':
                // Sort by Pareto optimal first, then by overall score
                sortedSolutions = solutionsWithElements.sort((a, b) => {
                    if (a.isParetoOptimal !== b.isParetoOptimal) {
                        return b.isParetoOptimal ? 1 : -1; // Pareto optimal first
                    }
                    const aOverall = (a.uxScore + a.safetyScore + a.efficiencyScore) / 3;
                    const bOverall = (b.uxScore + b.safetyScore + b.efficiencyScore) / 3;
                    return bOverall - aOverall; // Higher scores first
                });
                break;
            case 'ux':
                sortedSolutions = solutionsWithElements.sort((a, b) => b.uxScore - a.uxScore);
                break;
            case 'safety':
                sortedSolutions = solutionsWithElements.sort((a, b) => b.safetyScore - a.safetyScore);
                break;
            case 'efficiency':
                sortedSolutions = solutionsWithElements.sort((a, b) => b.efficiencyScore - a.efficiencyScore);
                break;
            default:
                sortedSolutions = solutionsWithElements; // No sorting
        }

        // Re-append elements in sorted order
        container.innerHTML = ''; // Clear container
        sortedSolutions.forEach(({ element }) => {
            container.appendChild(element);
        });

        console.log(`📊 Solutions sorted by: ${sortBy}`);
    }

    /**
     * Export optimization results with comprehensive scientific documentation
     */
    exportResults() {
        const paretoSolutions = this.currentSolutions.filter(s => s.isParetoOptimal);

        const data = {
            metadata: {
                title: "Taiko Fee Mechanism Multi-Objective Optimization Results",
                description: "NSGA-II optimization results for Taiko protocol fee mechanism parameters",
                exportDate: new Date().toISOString(),
                version: "1.0.0",
                framework: "Revised Optimization Framework (Post-Timing-Fix)",
                algorithm: "NSGA-II (Non-dominated Sorting Genetic Algorithm II)"
            },
            configuration: {
                weights: {
                    user_experience: {
                        w1_fee_affordability: this.currentWeights.w1_fee_affordability,
                        w2_fee_stability: this.currentWeights.w2_fee_stability,
                        w3_fee_predictability_1h: this.currentWeights.w3_fee_predictability_1h,
                        w4_fee_predictability_6h: this.currentWeights.w4_fee_predictability_6h
                    },
                    protocol_safety: {
                        w5_insolvency_protection: this.currentWeights.w5_insolvency_protection,
                        w6_deficit_duration: this.currentWeights.w6_deficit_duration,
                        w7_vault_stress: this.currentWeights.w7_vault_stress,
                        w8_continuous_underfunding: this.currentWeights.w8_continuous_underfunding
                    },
                    economic_efficiency: {
                        w9_vault_utilization: this.currentWeights.w9_vault_utilization,
                        w10_deficit_correction: this.currentWeights.w10_deficit_correction,
                        w11_capital_efficiency: this.currentWeights.w11_capital_efficiency
                    }
                },
                constraints: {
                    six_step_alignment: "H must be divisible by 6 (batch cycle alignment)",
                    parameter_bounds: {
                        mu: "[0.0, 1.0] - L1 weight parameter",
                        nu: "[0.02, 1.0] - Deficit weight parameter",
                        H: "[6, 576] - Horizon parameter (6-step aligned)"
                    }
                }
            },
            optimization_results: {
                total_solutions_evaluated: this.currentSolutions.length,
                pareto_optimal_solutions: paretoSolutions.length,
                pareto_dominance_ratio: paretoSolutions.length / this.currentSolutions.length,
                convergence_metrics: {
                    hypervolume_indicator: "Not computed",
                    spacing_metric: "Not computed",
                    spread_metric: "Not computed"
                }
            },
            pareto_front: paretoSolutions.map(solution => ({
                parameters: {
                    mu: solution.mu,
                    nu: solution.nu,
                    H: solution.H
                },
                objectives: {
                    ux_score: solution.uxScore,
                    safety_score: solution.safetyScore,
                    efficiency_score: solution.efficiencyScore
                },
                generation: solution.generation || 0,
                rank: 0 // All Pareto solutions have rank 0
            })),
            dominated_solutions: this.currentSolutions
                .filter(s => !s.isParetoOptimal)
                .map(solution => ({
                    parameters: {
                        mu: solution.mu,
                        nu: solution.nu,
                        H: solution.H
                    },
                    objectives: {
                        ux_score: solution.uxScore,
                        safety_score: solution.safetyScore,
                        efficiency_score: solution.efficiencyScore
                    },
                    generation: solution.generation || 0
                })),
            recommendations: this.generateRecommendations(paretoSolutions),
            methodology: {
                objective_functions: {
                    user_experience: "UX = Σ(wi × Mi) where Mi ∈ {fee_affordability, fee_stability, predictability_1h, predictability_6h}",
                    protocol_safety: "Safety = Σ(wi × Mi) where Mi ∈ {insolvency_protection, deficit_duration, vault_stress, continuous_underfunding}",
                    economic_efficiency: "Efficiency = Σ(wi × Mi) where Mi ∈ {vault_utilization, deficit_correction, capital_efficiency}"
                },
                pareto_dominance: "Solution A dominates B if A is at least as good as B in all objectives and strictly better in at least one",
                constraint_handling: "6-step batch alignment enforced as hard constraint during parameter generation",
                normalization: "All objectives normalized to [0,1] range for fair comparison"
            },
            references: {
                taiko_protocol: "https://taiko.xyz/",
                nsga_ii_paper: "Deb, K., et al. 'A fast and elitist multiobjective genetic algorithm: NSGA-II' (2002)",
                eip_1559: "https://eips.ethereum.org/EIPS/eip-1559"
            }
        };

        // Generate multiple export formats
        this.exportJSON(data);
        this.exportCSV(paretoSolutions);

        console.log('📁 Comprehensive results exported in JSON and CSV formats');
    }

    /**
     * Export JSON format
     */
    exportJSON(data) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `taiko-fee-optimization-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Export CSV format for data analysis
     */
    exportCSV(paretoSolutions) {
        const headers = [
            'mu', 'nu', 'H',
            'ux_score', 'safety_score', 'efficiency_score',
            'generation', 'is_pareto_optimal'
        ];

        const csvData = [
            headers.join(','),
            ...this.currentSolutions.map(solution => [
                solution.mu,
                solution.nu,
                solution.H,
                solution.uxScore,
                solution.safetyScore,
                solution.efficiencyScore,
                solution.generation || 0,
                solution.isParetoOptimal ? 1 : 0
            ].join(','))
        ].join('\n');

        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `taiko-fee-pareto-solutions-${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Generate optimization recommendations
     */
    generateRecommendations(paretoSolutions) {
        if (paretoSolutions.length === 0) {
            return { error: "No Pareto optimal solutions found" };
        }

        // Find representative solutions
        const bestUX = paretoSolutions.reduce((best, current) =>
            current.uxScore > best.uxScore ? current : best
        );

        const bestSafety = paretoSolutions.reduce((best, current) =>
            current.safetyScore > best.safetyScore ? current : best
        );

        const bestEfficiency = paretoSolutions.reduce((best, current) =>
            current.efficiencyScore > best.efficiencyScore ? current : best
        );

        return {
            best_user_experience: {
                parameters: { mu: bestUX.mu, nu: bestUX.nu, H: bestUX.H },
                scores: { ux: bestUX.uxScore, safety: bestUX.safetyScore, efficiency: bestUX.efficiencyScore },
                use_case: "Optimize for lowest fees and highest predictability"
            },
            best_safety: {
                parameters: { mu: bestSafety.mu, nu: bestSafety.nu, H: bestSafety.H },
                scores: { ux: bestSafety.uxScore, safety: bestSafety.safetyScore, efficiency: bestSafety.efficiencyScore },
                use_case: "Maximize protocol safety and vault resilience"
            },
            best_efficiency: {
                parameters: { mu: bestEfficiency.mu, nu: bestEfficiency.nu, H: bestEfficiency.H },
                scores: { ux: bestEfficiency.uxScore, safety: bestEfficiency.safetyScore, efficiency: bestEfficiency.efficiencyScore },
                use_case: "Optimize capital utilization and deficit correction"
            },
            analysis: {
                parameter_patterns: {
                    mu_preference: paretoSolutions.filter(s => s.mu < 0.1).length / paretoSolutions.length,
                    average_nu: paretoSolutions.reduce((sum, s) => sum + s.nu, 0) / paretoSolutions.length,
                    common_horizons: [...new Set(paretoSolutions.map(s => s.H))].sort((a, b) => a - b)
                },
                trade_offs: "Higher safety scores typically require higher nu values, potentially reducing efficiency"
            }
        };
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.optimizationResearch = new OptimizationResearchController();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OptimizationResearchController;
}
