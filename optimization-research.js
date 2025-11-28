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

        this.init();
    }

    /**
     * Initialize the research interface
     */
    init() {
        this.setupTabSwitching();
        this.setupMethodologyPanel();
        this.setupWeightSliders();
        this.setupOptimizationControls();
        this.setupAnimationControls();
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
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;

                // Update button states
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');

                // Update content visibility
                tabContents.forEach(content => {
                    content.classList.remove('active');
                    if (content.id === `${targetTab}-tab-content`) {
                        content.classList.add('active');
                    }
                });

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
    handlePointSelection(solution) {
        console.log('🎯 Point selected:', solution);

        // Update research feedback with selected solution details
        this.highlightSolutionInList(solution);
        this.updateSelectedSolutionMetrics(solution);
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
                item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
     * Start the optimization process
     */
    async startOptimization() {
        if (this.isOptimizing) return;

        this.isOptimizing = true;
        this.updateOptimizationUI(true);

        try {
            // Initialize optimization engine if needed
            if (!this.optimizationEngine && window.NSGAII) {
                this.optimizationEngine = new NSGAII({
                    populationSize: 30,
                    maxGenerations: 50,
                    weights: this.currentWeights,
                    onProgress: (progress) => this.handleOptimizationProgress(progress),
                    onSolution: (solution) => this.handleNewSolution(solution),
                    onComplete: (results) => this.handleOptimizationComplete(results)
                });
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
                parameters: { mu: solution.mu, nu: solution.nu, H: solution.H },
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

        const solutionsHTML = solutions.map(solution => `
            <div class="solution-item ${solution.isParetoOptimal ? 'pareto-optimal' : ''}">
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
