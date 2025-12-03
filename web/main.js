/**
 * Main entry point for Taiko Fee Mechanism Explorer
 *
 * This module imports and orchestrates all components to maintain
 * the exact same functionality as the original scattered files,
 * but with clean modular organization.
 */

// Core simulation engine components
import './components/historical-data-loader.js';
import './components/taiko-simulator-js.js';
import './components/metrics-framework-js.js';
import './components/specs-simulator.js';

// UI and visualization components
import './components/simulator.js';
import './components/charts.js';
import './components/pareto-visualizer.js';

// Optimization research components
import './components/nsga-ii-web.js';
import './components/optimization-research.js';

// Main application controller (must be last)
import './components/app.js';

console.log('🚀 Taiko Fee Mechanism Explorer initialized');
console.log('📊 All components loaded in correct order');