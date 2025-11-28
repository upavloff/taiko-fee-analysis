/**
 * 3D Pareto Frontier Visualization using Three.js
 *
 * Real-time visualization of multi-objective optimization results
 * showing Pareto optimal solutions in 3D space with:
 * - UX Score (X-axis, blue)
 * - Safety Score (Y-axis, red)
 * - Efficiency Score (Z-axis, green)
 */

class ParetoVisualizer {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.solutionPoints = [];
        this.paretoPoints = [];
        this.animationId = null;

        // Interactive features
        this.raycaster = null;
        this.mouse = new THREE.Vector2();
        this.selectedPoint = null;
        this.hoveredPoint = null;
        this.intersectionObjects = [];

        // UI elements for interaction
        this.tooltipElement = null;
        this.detailPanel = null;

        // Callbacks
        this.onPointSelected = null;
        this.onPointHovered = null;

        // Animation controls
        this.animationSettings = {
            enabled: true,
            speed: 1.0,
            rotationEnabled: true,
            pulseEnabled: true,
            trailEnabled: false,
            autoRotateCamera: false,
            cameraSpeed: 0.5
        };

        this.animationTime = 0;
        this.pointTrails = new Map();

        this.init();
    }

    /**
     * Initialize the Three.js scene and visualization
     */
    init() {
        if (!this.container) {
            console.error('Container not found for Pareto visualization');
            return;
        }

        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupControls();
        this.setupAxes();
        this.setupLighting();
        this.setupInteractivity();
        this.setupUI();
        this.startAnimation();

        console.log('🎯 3D Pareto visualization initialized');
    }

    /**
     * Setup the Three.js scene
     */
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf8fafc);
        this.scene.fog = new THREE.Fog(0xf8fafc, 2, 15);
    }

    /**
     * Setup camera with optimal viewing angle
     */
    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(3, 3, 3);
        this.camera.lookAt(0.5, 0.5, 0.5);
    }

    /**
     * Setup WebGL renderer
     */
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Clear container and add renderer
        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);

        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }

    /**
     * Setup orbit controls for interactive viewing
     */
    setupControls() {
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.target.set(0.5, 0.5, 0.5);
            this.controls.minDistance = 1;
            this.controls.maxDistance = 10;
        }
    }

    /**
     * Setup 3D axes with labels
     */
    setupAxes() {
        // Create axis lines
        const axisLength = 1.2;

        // UX Axis (X - Blue)
        const uxGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(axisLength, 0, 0)
        ]);
        const uxMaterial = new THREE.LineBasicMaterial({ color: 0x3182ce });
        const uxLine = new THREE.Line(uxGeometry, uxMaterial);
        this.scene.add(uxLine);

        // Safety Axis (Y - Red)
        const safetyGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, axisLength, 0)
        ]);
        const safetyMaterial = new THREE.LineBasicMaterial({ color: 0xe53e3e });
        const safetyLine = new THREE.Line(safetyGeometry, safetyMaterial);
        this.scene.add(safetyLine);

        // Efficiency Axis (Z - Green)
        const efficiencyGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 0, axisLength)
        ]);
        const efficiencyMaterial = new THREE.LineBasicMaterial({ color: 0x38a169 });
        const efficiencyLine = new THREE.Line(efficiencyGeometry, efficiencyMaterial);
        this.scene.add(efficiencyLine);

        // Add axis labels using CSS3DRenderer would be ideal, but for simplicity we'll use text sprites
        this.addAxisLabels();

        // Add grid lines
        this.addGridLines();
    }

    /**
     * Add text labels for axes
     */
    addAxisLabels() {
        const loader = new THREE.FontLoader();

        // Note: In production, you'd load a font file
        // For now, we'll create simple geometric text indicators

        // UX label (Blue arrow at end of X axis)
        const uxArrowGeometry = new THREE.ConeGeometry(0.02, 0.08, 8);
        const uxArrowMaterial = new THREE.MeshBasicMaterial({ color: 0x3182ce });
        const uxArrow = new THREE.Mesh(uxArrowGeometry, uxArrowMaterial);
        uxArrow.position.set(1.25, 0, 0);
        uxArrow.rotateZ(-Math.PI / 2);
        this.scene.add(uxArrow);

        // Safety label (Red arrow at end of Y axis)
        const safetyArrowGeometry = new THREE.ConeGeometry(0.02, 0.08, 8);
        const safetyArrowMaterial = new THREE.MeshBasicMaterial({ color: 0xe53e3e });
        const safetyArrow = new THREE.Mesh(safetyArrowGeometry, safetyArrowMaterial);
        safetyArrow.position.set(0, 1.25, 0);
        this.scene.add(safetyArrow);

        // Efficiency label (Green arrow at end of Z axis)
        const efficiencyArrowGeometry = new THREE.ConeGeometry(0.02, 0.08, 8);
        const efficiencyArrowMaterial = new THREE.MeshBasicMaterial({ color: 0x38a169 });
        const efficiencyArrow = new THREE.Mesh(efficiencyArrowGeometry, efficiencyArrowMaterial);
        efficiencyArrow.position.set(0, 0, 1.25);
        efficiencyArrow.rotateX(Math.PI / 2);
        this.scene.add(efficiencyArrow);
    }

    /**
     * Add grid lines for reference
     */
    addGridLines() {
        const gridMaterial = new THREE.LineBasicMaterial({
            color: 0xcccccc,
            opacity: 0.3,
            transparent: true
        });

        // Create grid on each plane
        for (let i = 0; i <= 10; i++) {
            const val = i / 10;

            // XY plane (Z=0)
            const xyGeometry1 = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(val, 0, 0),
                new THREE.Vector3(val, 1, 0)
            ]);
            const xyLine1 = new THREE.Line(xyGeometry1, gridMaterial);
            this.scene.add(xyLine1);

            const xyGeometry2 = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, val, 0),
                new THREE.Vector3(1, val, 0)
            ]);
            const xyLine2 = new THREE.Line(xyGeometry2, gridMaterial);
            this.scene.add(xyLine2);

            // XZ plane (Y=0)
            const xzGeometry1 = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(val, 0, 0),
                new THREE.Vector3(val, 0, 1)
            ]);
            const xzLine1 = new THREE.Line(xzGeometry1, gridMaterial);
            this.scene.add(xzLine1);

            const xzGeometry2 = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, val),
                new THREE.Vector3(1, 0, val)
            ]);
            const xzLine2 = new THREE.Line(xzGeometry2, gridMaterial);
            this.scene.add(xzLine2);

            // YZ plane (X=0)
            const yzGeometry1 = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, val, 0),
                new THREE.Vector3(0, val, 1)
            ]);
            const yzLine1 = new THREE.Line(yzGeometry1, gridMaterial);
            this.scene.add(yzLine1);

            const yzGeometry2 = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, val),
                new THREE.Vector3(0, 1, val)
            ]);
            const yzLine2 = new THREE.Line(yzGeometry2, gridMaterial);
            this.scene.add(yzLine2);
        }
    }

    /**
     * Setup interactive features - raycasting, mouse events
     */
    setupInteractivity() {
        this.raycaster = new THREE.Raycaster();
        this.raycaster.params.Points.threshold = 0.05; // Increase sensitivity for point selection

        // Mouse event listeners
        this.renderer.domElement.addEventListener('mousemove', (event) => this.onMouseMove(event));
        this.renderer.domElement.addEventListener('click', (event) => this.onMouseClick(event));
        this.renderer.domElement.addEventListener('mouseleave', () => this.onMouseLeave());

        // Prevent context menu on right click
        this.renderer.domElement.addEventListener('contextmenu', (event) => event.preventDefault());
    }

    /**
     * Setup UI elements for interaction feedback
     */
    setupUI() {
        // Create tooltip element
        this.tooltipElement = document.createElement('div');
        this.tooltipElement.className = 'pareto-tooltip';
        this.tooltipElement.style.cssText = `
            position: absolute;
            background: rgba(45, 55, 72, 0.95);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-family: 'SF Pro Text', system-ui, sans-serif;
            pointer-events: none;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.2s ease;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            backdrop-filter: blur(8px);
        `;
        document.body.appendChild(this.tooltipElement);

        // Create detail panel in the visualization container
        this.detailPanel = document.createElement('div');
        this.detailPanel.className = 'pareto-detail-panel';
        this.detailPanel.style.cssText = `
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            width: 280px;
            font-size: 13px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            backdrop-filter: blur(8px);
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
            z-index: 100;
        `;

        this.detailPanel.innerHTML = `
            <div class="detail-header">
                <h4 style="margin: 0 0 12px 0; color: #2d3748; font-weight: 600;">Solution Details</h4>
                <button class="close-detail" style="position: absolute; top: 12px; right: 12px; background: none; border: none; font-size: 16px; color: #6b7280; cursor: pointer;">×</button>
            </div>
            <div class="detail-content"></div>
        `;

        // Position detail panel relative to container
        this.container.style.position = 'relative';
        this.container.appendChild(this.detailPanel);

        // Close button functionality
        const closeBtn = this.detailPanel.querySelector('.close-detail');
        closeBtn.addEventListener('click', () => this.hideDetailPanel());
    }

    /**
     * Handle mouse movement for hover effects
     */
    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.intersectionObjects);

        if (intersects.length > 0) {
            const intersectedObject = intersects[0].object;
            if (intersectedObject !== this.hoveredPoint) {
                // Reset previous hovered point
                if (this.hoveredPoint) {
                    this.resetPointAppearance(this.hoveredPoint);
                }

                // Highlight new hovered point
                this.hoveredPoint = intersectedObject;
                this.highlightPoint(this.hoveredPoint, 'hover');
                this.showTooltip(event, this.hoveredPoint.userData);

                // Change cursor
                this.renderer.domElement.style.cursor = 'pointer';

                // Callback
                if (this.onPointHovered) {
                    this.onPointHovered(this.hoveredPoint.userData);
                }
            }

            // Update tooltip position
            this.updateTooltipPosition(event);

        } else {
            // No intersection
            if (this.hoveredPoint) {
                this.resetPointAppearance(this.hoveredPoint);
                this.hoveredPoint = null;
                this.hideTooltip();
                this.renderer.domElement.style.cursor = 'default';
            }
        }
    }

    /**
     * Handle mouse clicks for selection
     */
    onMouseClick(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.intersectionObjects);

        if (intersects.length > 0) {
            const clickedObject = intersects[0].object;

            // Reset previous selection
            if (this.selectedPoint && this.selectedPoint !== clickedObject) {
                this.resetPointAppearance(this.selectedPoint);
            }

            // Select new point
            this.selectedPoint = clickedObject;
            this.highlightPoint(this.selectedPoint, 'selected');
            this.showDetailPanel(this.selectedPoint.userData);

            // Callback
            if (this.onPointSelected) {
                this.onPointSelected(this.selectedPoint.userData);
            }
        } else {
            // Click on empty space - deselect
            if (this.selectedPoint) {
                this.resetPointAppearance(this.selectedPoint);
                this.selectedPoint = null;
                this.hideDetailPanel();
            }
        }
    }

    /**
     * Handle mouse leaving the canvas
     */
    onMouseLeave() {
        if (this.hoveredPoint) {
            this.resetPointAppearance(this.hoveredPoint);
            this.hoveredPoint = null;
        }
        this.hideTooltip();
        this.renderer.domElement.style.cursor = 'default';
    }

    /**
     * Highlight a point with different styles
     */
    highlightPoint(pointMesh, type = 'hover') {
        if (!pointMesh) return;

        const scale = type === 'selected' ? 2.0 : 1.5;
        const emissive = type === 'selected' ? 0x444444 : 0x222222;

        // Scale animation
        const targetScale = scale;
        const currentScale = pointMesh.scale.x;

        if (Math.abs(currentScale - targetScale) > 0.1) {
            const animate = () => {
                const newScale = currentScale + (targetScale - currentScale) * 0.3;
                pointMesh.scale.set(newScale, newScale, newScale);

                if (Math.abs(newScale - targetScale) > 0.05) {
                    requestAnimationFrame(animate);
                } else {
                    pointMesh.scale.set(targetScale, targetScale, targetScale);
                }
            };
            animate();
        }

        // Material enhancement
        if (pointMesh.material) {
            pointMesh.material.emissive.setHex(emissive);
            pointMesh.material.needsUpdate = true;
        }
    }

    /**
     * Reset point appearance to default
     */
    resetPointAppearance(pointMesh) {
        if (!pointMesh) return;

        const defaultScale = pointMesh.userData.isParetoOptimal ? 1.2 : 1.0;

        pointMesh.scale.set(defaultScale, defaultScale, defaultScale);

        if (pointMesh.material) {
            pointMesh.material.emissive.setHex(0x000000);
            pointMesh.material.needsUpdate = true;
        }
    }

    /**
     * Show tooltip with solution information
     */
    showTooltip(event, solutionData) {
        if (!this.tooltipElement || !solutionData) return;

        const tooltipContent = `
            <div style="font-weight: 600; margin-bottom: 8px; color: #ffffff;">
                ${solutionData.isParetoOptimal ? '⭐ Pareto Optimal' : '📍 Solution'}
            </div>
            <div style="margin-bottom: 4px;">
                <strong>Parameters:</strong> μ=${solutionData.mu.toFixed(3)}, ν=${solutionData.nu.toFixed(3)}, H=${solutionData.H}
            </div>
            <div style="margin-bottom: 4px;">
                <span style="color: #63b3ed;">UX:</span> ${solutionData.uxScore.toFixed(3)} |
                <span style="color: #f56565;">Safety:</span> ${solutionData.safetyScore.toFixed(3)} |
                <span style="color: #68d391;">Efficiency:</span> ${solutionData.efficiencyScore.toFixed(3)}
            </div>
            <div style="font-size: 11px; color: #cbd5e0; margin-top: 6px;">
                Click to view detailed analysis
            </div>
        `;

        this.tooltipElement.innerHTML = tooltipContent;
        this.updateTooltipPosition(event);
        this.tooltipElement.style.opacity = '1';
    }

    /**
     * Update tooltip position
     */
    updateTooltipPosition(event) {
        if (!this.tooltipElement) return;

        const offset = 15;
        this.tooltipElement.style.left = `${event.clientX + offset}px`;
        this.tooltipElement.style.top = `${event.clientY - this.tooltipElement.offsetHeight - offset}px`;
    }

    /**
     * Hide tooltip
     */
    hideTooltip() {
        if (this.tooltipElement) {
            this.tooltipElement.style.opacity = '0';
        }
    }

    /**
     * Show detailed information panel
     */
    showDetailPanel(solutionData) {
        if (!this.detailPanel || !solutionData) return;

        const content = this.detailPanel.querySelector('.detail-content');
        content.innerHTML = `
            <div class="solution-summary" style="margin-bottom: 16px; padding: 12px; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3182ce;">
                <div style="font-weight: 600; color: #1e40af; margin-bottom: 6px;">
                    ${solutionData.isParetoOptimal ? '⭐ Pareto Optimal Solution' : '📍 Feasible Solution'}
                </div>
                <div style="font-size: 12px; color: #1e40af;">
                    Generation ${solutionData.generation || 'N/A'}
                </div>
            </div>

            <div class="parameter-section" style="margin-bottom: 16px;">
                <h5 style="margin: 0 0 8px 0; color: #2d3748; font-size: 14px;">Fee Mechanism Parameters</h5>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px;">
                    <div style="background: #f8fafc; padding: 8px; border-radius: 6px;">
                        <strong>μ (L1 Weight)</strong><br>
                        <span style="color: #3182ce; font-weight: 600;">${solutionData.mu.toFixed(4)}</span>
                    </div>
                    <div style="background: #f8fafc; padding: 8px; border-radius: 6px;">
                        <strong>ν (Deficit Weight)</strong><br>
                        <span style="color: #3182ce; font-weight: 600;">${solutionData.nu.toFixed(4)}</span>
                    </div>
                </div>
                <div style="background: #f8fafc; padding: 8px; border-radius: 6px; margin-top: 8px;">
                    <strong>H (Horizon)</strong><br>
                    <span style="color: #3182ce; font-weight: 600;">${solutionData.H} steps</span>
                    <span style="font-size: 11px; color: #6b7280;">(${(solutionData.H / 6).toFixed(1)} batches)</span>
                </div>
            </div>

            <div class="objectives-section" style="margin-bottom: 16px;">
                <h5 style="margin: 0 0 8px 0; color: #2d3748; font-size: 14px;">Objective Scores</h5>
                <div style="display: flex; flex-direction: column; gap: 6px; font-size: 13px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: #f0f9ff; border-radius: 4px;">
                        <span style="color: #3182ce; font-weight: 500;">👥 User Experience</span>
                        <span style="font-weight: 600;">${solutionData.uxScore.toFixed(4)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: #fef2f2; border-radius: 4px;">
                        <span style="color: #e53e3e; font-weight: 500;">🛡️ Protocol Safety</span>
                        <span style="font-weight: 600;">${solutionData.safetyScore.toFixed(4)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: #f0fff4; border-radius: 4px;">
                        <span style="color: #38a169; font-weight: 500;">💰 Economic Efficiency</span>
                        <span style="font-weight: 600;">${solutionData.efficiencyScore.toFixed(4)}</span>
                    </div>
                </div>
            </div>

            <div class="actions-section">
                <button onclick="window.optimizationResearch?.exportSolution('${JSON.stringify(solutionData).replace(/"/g, '&quot;')}')"
                        style="width: 100%; padding: 8px 12px; background: #3182ce; color: white; border: none; border-radius: 6px; font-size: 13px; cursor: pointer; transition: background 0.2s;">
                    📊 Export Solution Data
                </button>
            </div>
        `;

        this.detailPanel.style.opacity = '1';
        this.detailPanel.style.transform = 'translateX(0)';
    }

    /**
     * Hide detailed information panel
     */
    hideDetailPanel() {
        if (this.detailPanel) {
            this.detailPanel.style.opacity = '0';
            this.detailPanel.style.transform = 'translateX(100%)';
        }
    }

    /**
     * Setup lighting for better visualization
     */
    setupLighting() {
        // Ambient light for overall illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Directional light for shadows and depth
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(2, 3, 1);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 1024;
        directionalLight.shadow.mapSize.height = 1024;
        this.scene.add(directionalLight);
    }

    /**
     * Add a new solution point to the visualization
     */
    addSolution(solution) {
        const { uxScore, safetyScore, efficiencyScore, isParetoOptimal } = solution;

        // Create sphere geometry for the point
        const geometry = new THREE.SphereGeometry(isParetoOptimal ? 0.015 : 0.01, 16, 16);

        // Color based on Pareto optimality
        const color = isParetoOptimal ? 0xffd700 : 0x666666; // Gold for Pareto, gray for dominated
        const material = new THREE.MeshLambertMaterial({
            color: color,
            opacity: isParetoOptimal ? 1.0 : 0.6,
            transparent: true
        });

        const sphere = new THREE.Mesh(geometry, material);

        // Position in 3D space (normalize scores to [0,1] range)
        sphere.position.set(uxScore, safetyScore, efficiencyScore);

        // Add animation entrance effect
        sphere.scale.set(0, 0, 0);
        this.animatePointEntrance(sphere);

        // Store reference to solution data
        sphere.userData = solution;

        this.scene.add(sphere);

        // Register for interaction
        this.intersectionObjects.push(sphere);

        if (isParetoOptimal) {
            this.paretoPoints.push(sphere);
            this.updateParetoFrontSurface();
        } else {
            this.solutionPoints.push(sphere);
        }
    }

    /**
     * Animate point entrance with scaling effect
     */
    animatePointEntrance(sphere) {
        const startTime = Date.now();
        const duration = 500; // ms

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Elastic scale animation
            const scale = progress * (1.1 - 0.1 * Math.cos(progress * Math.PI));
            sphere.scale.set(scale, scale, scale);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                sphere.scale.set(1, 1, 1);
            }
        };

        animate();
    }

    /**
     * Update the Pareto front surface visualization
     */
    updateParetoFrontSurface() {
        // Remove existing surface if any
        const existingSurface = this.scene.getObjectByName('paretoSurface');
        if (existingSurface) {
            this.scene.remove(existingSurface);
        }

        if (this.paretoPoints.length < 3) return; // Need at least 3 points for a surface

        // Create a convex hull or simplified surface representation
        // For now, we'll connect Pareto points with lines to show the frontier
        this.createParetoLines();
    }

    /**
     * Create lines connecting Pareto optimal points
     */
    createParetoLines() {
        // Remove existing lines
        const existingLines = this.scene.getObjectByName('paretoLines');
        if (existingLines) {
            this.scene.remove(existingLines);
        }

        if (this.paretoPoints.length < 2) return;

        const group = new THREE.Group();
        group.name = 'paretoLines';

        // Create lines between nearby Pareto points
        for (let i = 0; i < this.paretoPoints.length; i++) {
            for (let j = i + 1; j < this.paretoPoints.length; j++) {
                const point1 = this.paretoPoints[i].position;
                const point2 = this.paretoPoints[j].position;

                // Only connect points that are reasonably close
                const distance = point1.distanceTo(point2);
                if (distance < 0.3) {
                    const geometry = new THREE.BufferGeometry().setFromPoints([point1, point2]);
                    const material = new THREE.LineBasicMaterial({
                        color: 0xffd700,
                        opacity: 0.3,
                        transparent: true
                    });
                    const line = new THREE.Line(geometry, material);
                    group.add(line);
                }
            }
        }

        this.scene.add(group);
    }

    /**
     * Clear all solutions from the visualization
     */
    clear() {
        // Remove all solution points
        [...this.solutionPoints, ...this.paretoPoints].forEach(point => {
            this.scene.remove(point);
        });

        // Clear arrays
        this.solutionPoints = [];
        this.paretoPoints = [];
        this.intersectionObjects = [];

        // Reset interaction state
        this.selectedPoint = null;
        this.hoveredPoint = null;
        this.hideTooltip();
        this.hideDetailPanel();

        // Remove Pareto surface and lines
        const surface = this.scene.getObjectByName('paretoSurface');
        const lines = this.scene.getObjectByName('paretoLines');

        if (surface) this.scene.remove(surface);
        if (lines) this.scene.remove(lines);
    }

    /**
     * Handle window resize
     */
    handleResize() {
        if (!this.container || !this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * Start the animation loop
     */
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);

            if (!this.animationSettings.enabled) {
                this.renderer.render(this.scene, this.camera);
                return;
            }

            this.animationTime += 0.016 * this.animationSettings.speed; // 60fps base

            // Update controls
            if (this.controls) {
                this.controls.update();

                // Auto-rotate camera if enabled
                if (this.animationSettings.autoRotateCamera) {
                    this.controls.autoRotate = true;
                    this.controls.autoRotateSpeed = this.animationSettings.cameraSpeed;
                } else {
                    this.controls.autoRotate = false;
                }
            }

            // Animate Pareto points
            this.animatePoints();

            // Update trails if enabled
            if (this.animationSettings.trailEnabled) {
                this.updateTrails();
            }

            // Render the scene
            this.renderer.render(this.scene, this.camera);
        };

        animate();
    }

    /**
     * Animate individual points based on settings
     */
    animatePoints() {
        const rotationSpeed = 0.01 * this.animationSettings.speed;
        const pulseSpeed = 2.0 * this.animationSettings.speed;

        [...this.solutionPoints, ...this.paretoPoints].forEach((point, index) => {
            // Rotation animation
            if (this.animationSettings.rotationEnabled) {
                point.rotation.y += rotationSpeed;
                point.rotation.x += rotationSpeed * 0.5;
            }

            // Pulse animation
            if (this.animationSettings.pulseEnabled) {
                const pulsePhase = this.animationTime * pulseSpeed + index * 0.5;
                const pulseScale = 1.0 + 0.1 * Math.sin(pulsePhase);
                point.scale.setScalar(pulseScale);

                // Pulse opacity for non-Pareto points
                if (this.solutionPoints.includes(point)) {
                    const material = point.material;
                    if (material && material.opacity !== undefined) {
                        material.opacity = 0.7 + 0.2 * Math.sin(pulsePhase);
                    }
                }
            }
        });
    }

    /**
     * Update trail effects for moving points
     */
    updateTrails() {
        // Implementation for particle trails - can be extended later
        // This would create trailing effects behind moving points
    }

    /**
     * Stop the animation loop
     */
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    /**
     * Destroy the visualizer and clean up resources
     */
    destroy() {
        this.stopAnimation();

        // Clean up interaction elements
        if (this.tooltipElement) {
            document.body.removeChild(this.tooltipElement);
            this.tooltipElement = null;
        }

        if (this.detailPanel) {
            this.container.removeChild(this.detailPanel);
            this.detailPanel = null;
        }

        // Remove event listeners
        if (this.renderer && this.renderer.domElement) {
            this.renderer.domElement.removeEventListener('mousemove', this.onMouseMove);
            this.renderer.domElement.removeEventListener('click', this.onMouseClick);
            this.renderer.domElement.removeEventListener('mouseleave', this.onMouseLeave);
        }

        if (this.renderer) {
            this.renderer.dispose();
        }

        if (this.controls) {
            this.controls.dispose();
        }

        // Clear the container
        if (this.container) {
            this.container.innerHTML = '';
        }
    }

    /**
     * Export the current view as an image
     */
    exportImage(filename = 'pareto-frontier') {
        if (!this.renderer) return;

        // Render at high resolution
        const originalSize = new THREE.Vector2();
        this.renderer.getSize(originalSize);

        const exportWidth = 1920;
        const exportHeight = 1080;

        this.renderer.setSize(exportWidth, exportHeight);
        this.camera.aspect = exportWidth / exportHeight;
        this.camera.updateProjectionMatrix();

        this.renderer.render(this.scene, this.camera);

        // Create download link
        const canvas = this.renderer.domElement;
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}-${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            // Restore original size
            this.renderer.setSize(originalSize.x, originalSize.y);
            this.camera.aspect = originalSize.x / originalSize.y;
            this.camera.updateProjectionMatrix();
        });
    }

    /**
     * Get statistics about the current visualization
     */
    getStats() {
        return {
            totalSolutions: this.solutionPoints.length + this.paretoPoints.length,
            paretoOptimal: this.paretoPoints.length,
            dominated: this.solutionPoints.length,
            paretoRatio: this.paretoPoints.length / (this.solutionPoints.length + this.paretoPoints.length)
        };
    }

    /**
     * Update animation settings
     */
    setAnimationSettings(settings) {
        this.animationSettings = { ...this.animationSettings, ...settings };
    }

    /**
     * Get current animation settings
     */
    getAnimationSettings() {
        return { ...this.animationSettings };
    }

    /**
     * Toggle specific animation features
     */
    toggleAnimation(feature, enabled) {
        if (feature in this.animationSettings) {
            this.animationSettings[feature] = enabled;
        }
    }

    /**
     * Set animation speed multiplier
     */
    setAnimationSpeed(speed) {
        this.animationSettings.speed = Math.max(0.1, Math.min(5.0, speed));
    }

    /**
     * Reset camera position with smooth animation
     */
    resetCameraPosition(animated = true) {
        if (!this.camera || !this.controls) return;

        const targetPosition = { x: 2, y: 2, z: 2 };
        const targetTarget = { x: 0, y: 0, z: 0 };

        if (animated) {
            // Smooth animation to reset position
            const startPosition = this.camera.position.clone();
            const startTarget = this.controls.target.clone();
            const duration = 1000; // 1 second
            const startTime = Date.now();

            const animateReset = () => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease-out cubic

                // Interpolate camera position
                this.camera.position.lerpVectors(startPosition,
                    new THREE.Vector3(targetPosition.x, targetPosition.y, targetPosition.z),
                    easeProgress);

                // Interpolate target
                this.controls.target.lerpVectors(startTarget,
                    new THREE.Vector3(targetTarget.x, targetTarget.y, targetTarget.z),
                    easeProgress);

                this.controls.update();

                if (progress < 1) {
                    requestAnimationFrame(animateReset);
                }
            };

            animateReset();
        } else {
            // Immediate reset
            this.camera.position.set(targetPosition.x, targetPosition.y, targetPosition.z);
            this.controls.target.set(targetTarget.x, targetTarget.y, targetTarget.z);
            this.controls.update();
        }
    }

    /**
     * Focus camera on a specific region with animation
     */
    focusOnRegion(center, radius = 1, duration = 1000) {
        if (!this.camera || !this.controls) return;

        const startPosition = this.camera.position.clone();
        const startTarget = this.controls.target.clone();

        const targetPosition = new THREE.Vector3()
            .copy(center)
            .add(new THREE.Vector3(radius, radius, radius));

        const targetTarget = center.clone();

        const startTime = Date.now();

        const animateFocus = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease-out cubic

            this.camera.position.lerpVectors(startPosition, targetPosition, easeProgress);
            this.controls.target.lerpVectors(startTarget, targetTarget, easeProgress);
            this.controls.update();

            if (progress < 1) {
                requestAnimationFrame(animateFocus);
            }
        };

        animateFocus();
    }

    /**
     * Focus camera on selected point
     */
    focusOnPoint(solution) {
        if (!this.camera || !this.controls) return;

        const point = this.intersectionObjects.find(obj =>
            obj.userData.id === solution.id ||
            (obj.userData.mu === solution.mu && obj.userData.nu === solution.nu && obj.userData.H === solution.H)
        );

        if (point) {
            // Animate camera to focus on point
            const targetPosition = point.position.clone();
            targetPosition.multiplyScalar(1.5); // Move camera back a bit

            this.controls.target.copy(point.position);
            this.camera.position.copy(targetPosition);
            this.controls.update();

            // Highlight the point
            if (this.selectedPoint && this.selectedPoint !== point) {
                this.resetPointAppearance(this.selectedPoint);
            }
            this.selectedPoint = point;
            this.highlightPoint(point, 'selected');
            this.showDetailPanel(point.userData);
        }
    }

    /**
     * Filter visible solutions based on criteria
     */
    filterSolutions(criteria) {
        [...this.solutionPoints, ...this.paretoPoints].forEach(point => {
            const solution = point.userData;
            let visible = true;

            if (criteria.paretoOnly && !solution.isParetoOptimal) {
                visible = false;
            }

            if (criteria.minUX !== undefined && solution.uxScore < criteria.minUX) {
                visible = false;
            }

            if (criteria.minSafety !== undefined && solution.safetyScore < criteria.minSafety) {
                visible = false;
            }

            if (criteria.minEfficiency !== undefined && solution.efficiencyScore < criteria.minEfficiency) {
                visible = false;
            }

            point.visible = visible;
        });
    }

    /**
     * Reset all filters
     */
    resetFilters() {
        [...this.solutionPoints, ...this.paretoPoints].forEach(point => {
            point.visible = true;
        });
    }

    /**
     * Set callback functions for interaction events
     */
    setCallbacks({ onPointSelected, onPointHovered }) {
        this.onPointSelected = onPointSelected;
        this.onPointHovered = onPointHovered;
    }

    /**
     * Get all solutions data
     */
    getAllSolutions() {
        return [...this.solutionPoints, ...this.paretoPoints].map(point => point.userData);
    }

    /**
     * Get only Pareto optimal solutions
     */
    getParetoSolutions() {
        return this.paretoPoints.map(point => point.userData);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ParetoVisualizer;
}

// Make available globally
window.ParetoVisualizer = ParetoVisualizer;