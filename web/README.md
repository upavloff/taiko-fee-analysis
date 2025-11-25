# Taiko Fee Mechanism Explorer

An interactive web application for exploring and analyzing the Taiko Layer 2 fee mechanism parameters.

## Features

🎛️ **Interactive Parameter Controls**
- Real-time sliders for μ (L1 cost weight), ν (deficit weight), and H (horizon)
- Vault initialization options
- L1 volatility scenarios

🚀 **Preset Configurations**
- Fee Stability Focus (DeFi-optimized)
- Vault Management (Treasury-focused)
- L1 Tracking (Cost-accuracy focused)
- Balanced Approach (General purpose)
- Conservative (Risk-averse)
- Aggressive (High-performance)

📊 **Real-time Visualization**
- Fee evolution over time
- Vault balance dynamics
- L1 basefee simulation
- Fee vs L1 correlation analysis

📈 **Performance Metrics**
- Average fee calculation
- Fee variability (coefficient of variation)
- Time underfunded percentage
- L1 tracking error

🔬 **Methodology**
- Based on extensive research and analysis
- Monte Carlo simulation with Geometric Brownian Motion
- Proper vault initialization (critical for meaningful results)
- Performance evaluation against industry benchmarks

## How to Use

1. **Select a Preset**: Click any preset button to load optimized parameters for specific use cases
2. **Adjust Parameters**: Use sliders to fine-tune parameters in real-time
3. **Run Simulation**: Click "🚀 Run Simulation" to see results
4. **Analyze Results**: Review performance metrics and charts

## Parameter Explanations

### μ (L1 Cost Weight) - 0.0 to 1.0
Controls how much L1 costs influence fees. Higher values make fees track L1 basefee more closely.
- μ=0: Ignores L1 costs entirely (stable but potentially inaccurate)
- μ=1: Includes full L1 cost impact (volatile but accurate)

### ν (Deficit Weight) - 0.1 to 0.9
Controls strength of vault deficit correction. Higher values make fees respond more aggressively to underfunding.
- Low values: Gradual correction, potential long underfunding
- High values: Fast correction, higher fee volatility

### H (Horizon) - 24 to 576 steps
Time horizon for L1 cost prediction. 144 steps ≈ 1 day.
- Short horizon: Fast response, higher volatility
- Long horizon: Smooth response, slower adaptation

## Performance Thresholds

**Excellent Performance:**
- Fee CV < 0.5
- Time Underfunded < 5%
- L1 Tracking Error < 0.3

**Acceptable Performance:**
- Fee CV < 1.0
- Time Underfunded < 15%
- L1 Tracking Error < 0.6

## Keyboard Shortcuts

- `Ctrl/Cmd + Enter`: Run simulation
- `Ctrl/Cmd + E`: Export results

## Technical Implementation

Built with:
- Vanilla JavaScript (no frameworks)
- Chart.js for visualizations
- CSS Grid and Flexbox for responsive layout
- Monte Carlo simulation engine
- Geometric Brownian Motion for L1 basefee modeling

## Research Background

This tool implements findings from comprehensive analysis of the Taiko fee mechanism, including:

- **Vault initialization impact**: Empty vaults create unrealistic fee spikes
- **μ=0 viability**: Trades accuracy for stability
- **Parameter sensitivity**: Different metrics respond differently to parameter changes
- **Real-world validation**: Tested against current Ethereum basefee conditions

## Deployment

This is a static website that can be deployed to:
- GitHub Pages
- Netlify
- Vercel
- Any static hosting service

No backend required - all simulation runs in the browser.

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## License

MIT License - Feel free to use, modify, and distribute.

---

Built with ❤️ for the Taiko and L2 research community.