// Chart.js configurations and utilities for Taiko Fee Explorer

class ChartManager {
    constructor() {
        this.charts = {};
        this.colors = {
            primary: '#4f46e5',
            secondary: '#10b981',
            accent: '#f59e0b',
            danger: '#ef4444',
            muted: '#6b7280'
        };
    }

    createFeeChart(canvasId, data, gasPerTx) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Use Taiko (L2) step timing: 2s per step
        const timeLabels = data.map((d, i) => {
            const seconds = (typeof d.l2ElapsedSeconds === 'number') ? d.l2ElapsedSeconds : (i * 2);
            return `${(seconds / 3600).toFixed(2)}h`;
        });
        // Convert from per-transaction ETH to per-gas gwei: (ETH * 1e9) / gasPerTx
        const feeData = data.map(d => (d.estimatedFee * 1e9) / (gasPerTx || 200)); // Default gasPerTx = 200 (corrected from bug analysis)

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'Taiko Estimated Fee per Gas (gwei)',
                    data: feeData,
                    borderColor: this.colors.primary,
                    backgroundColor: this.colors.primary + '20',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Taiko Estimated Fee per Gas Evolution Over Time',
                        font: { size: 14, weight: 600 },
                        color: '#2d3748',
                        padding: { bottom: 20 }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 32, 44, 0.9)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#4299e1',
                        borderWidth: 1,
                        cornerRadius: 6,
                        displayColors: false,
                        callbacks: {
                            title: function(context) {
                                return `Time: ${context[0].label}`;
                            },
                            label: function(context) {
                                const value = context.raw;
                                return `Taiko Estimated Fee per Gas: ${value.toFixed(3)} gwei`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Fee per Gas (gwei)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        },
                        type: 'linear'
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 6
                    }
                }
            }
        });
    }

    createVaultChart(canvasId, data, targetBalance = 100) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Use Taiko (L2) step timing: 2s per step
        const timeLabels = data.map((d, i) => {
            const seconds = (typeof d.l2ElapsedSeconds === 'number') ? d.l2ElapsedSeconds : (i * 2);
            return `${(seconds / 3600).toFixed(2)}h`;
        });
        const vaultData = data.map(d => d.vaultBalance);

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [
                    {
                        label: 'Vault Balance (ETH)',
                        data: vaultData,
                        borderColor: this.colors.secondary,
                        backgroundColor: this.colors.secondary + '20',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'Target Balance',
                        data: new Array(data.length).fill(targetBalance),
                        borderColor: this.colors.muted,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Vault Balance Over Time',
                        font: { size: 14, weight: 600 },
                        color: '#2d3748',
                        padding: { bottom: 20 }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: { size: 11 },
                            color: '#4a5568'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 32, 44, 0.9)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#4299e1',
                        borderWidth: 1,
                        cornerRadius: 6,
                        callbacks: {
                            title: function(context) {
                                return `Time: ${context[0].label}`;
                            },
                            label: function(context) {
                                const value = context.raw;
                                if (context.dataset.label === 'Vault Balance (ETH)') {
                                    return `${context.dataset.label}: ${value.toFixed(6)} ETH`;
                                } else {
                                    return `${context.dataset.label}: ${value.toFixed(1)} ETH`;
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Balance (ETH)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 6
                    }
                }
            }
        });
    }

    createL1Chart(canvasId, data) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Use L1 timing: real timestamps for historical data, 12s spacing for simulated L1
        const timeLabels = data.map((d, i) => {
            const seconds = (typeof d.l1ElapsedSeconds === 'number') ? d.l1ElapsedSeconds : (i * 12);
            return `${(seconds / 3600).toFixed(2)}h`;
        });
        const l1Data = data.map(d => d.l1Basefee / 1e9); // Convert to gwei

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'L1 Basefee (gwei)',
                    data: l1Data,
                    borderColor: this.colors.muted,
                    backgroundColor: this.colors.muted + '20',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'L1 Basefee Evolution',
                        font: { size: 14, weight: 600 },
                        color: '#2d3748',
                        padding: { bottom: 20 }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 32, 44, 0.9)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#4299e1',
                        borderWidth: 1,
                        cornerRadius: 6,
                        displayColors: false,
                        callbacks: {
                            title: function(context) {
                                return `Time: ${context[0].label}`;
                            },
                            label: function(context) {
                                const value = context.raw;
                                return `Basefee: ${value.toFixed(2)} gwei`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Basefee (gwei)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 6
                    }
                }
            }
        });
    }

    createCorrelationChart(canvasId, data, gasPerTx) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Create scatter plot data
        const scatterData = data.map(d => ({
            x: d.l1Basefee / 1e9, // gwei
            y: (d.estimatedFee * 1e9) / (gasPerTx || 200) // Convert ETH to per-gas gwei (corrected default)
        }));

        this.charts[canvasId] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Taiko Estimated Fee per Gas vs L1 Basefee',
                    data: scatterData,
                    backgroundColor: this.colors.accent + '60',
                    borderColor: this.colors.accent,
                    borderWidth: 1,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: true,
                    mode: 'point'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Taiko Estimated Fee per Gas vs L1 Basefee Correlation',
                        font: { size: 14, weight: 600 },
                        color: '#2d3748',
                        padding: { bottom: 20 }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 32, 44, 0.9)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#4299e1',
                        borderWidth: 1,
                        cornerRadius: 6,
                        displayColors: false,
                        callbacks: {
                            title: function() {
                                return 'Data Point';
                            },
                            label: function(context) {
                                const point = context.raw;
                                return [
                                    `L1 Basefee: ${point.x.toFixed(2)} gwei`,
                                    `Taiko Estimated Fee per Gas: ${point.y.toFixed(3)} gwei`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'L1 Basefee (gwei)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Taiko Estimated Fee per Gas (gwei)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 11 }
                        },
                        type: 'linear'
                    }
                }
            }
        });
    }

    updateMetricCard(cardId, value, status) {
        const card = document.getElementById(cardId);
        const valueElement = card.querySelector('.metric-value');
        const statusElement = card.querySelector('.metric-status');

        // Format value based on metric type
        let formattedValue;
        if (cardId === 'avg-fee-card') {
            // Value is already in per-gas gwei, just format
            formattedValue = value.toFixed(3);
        } else if (cardId === 'fee-cv-card' || cardId === 'tracking-card') {
            formattedValue = value.toFixed(3);
        } else if (cardId === 'underfunded-card') {
            formattedValue = value.toFixed(1);
        } else {
            formattedValue = value.toString();
        }

        valueElement.textContent = formattedValue;

        // Update status
        statusElement.className = 'metric-status ' + status;
        statusElement.textContent = this.getStatusText(status);

        // Update card border based on status
        card.style.borderLeftColor = this.getStatusBorderColor(status);
        card.style.borderLeftWidth = '3px';
    }

    getStatusText(status) {
        switch(status) {
            case 'excellent': return '✅ Excellent';
            case 'good': return '🟡 Good';
            case 'poor': return '❌ Poor';
            default: return '';
        }
    }

    getStatusBorderColor(status) {
        switch(status) {
            case 'excellent': return '#10b981';
            case 'good': return '#f59e0b';
            case 'poor': return '#ef4444';
            default: return '#e2e8f0';
        }
    }

    evaluateMetrics(metrics) {
        const evaluations = {};

        // Fee CV evaluation
        if (metrics.feeCV < 0.5) {
            evaluations.feeCV = 'excellent';
        } else if (metrics.feeCV < 1.0) {
            evaluations.feeCV = 'good';
        } else {
            evaluations.feeCV = 'poor';
        }

        // Time underfunded evaluation
        if (metrics.timeUnderfundedPct < 5) {
            evaluations.timeUnderfunded = 'excellent';
        } else if (metrics.timeUnderfundedPct < 15) {
            evaluations.timeUnderfunded = 'good';
        } else {
            evaluations.timeUnderfunded = 'poor';
        }

        // L1 tracking error evaluation
        if (metrics.l1TrackingError < 0.3) {
            evaluations.l1Tracking = 'excellent';
        } else if (metrics.l1TrackingError < 0.6) {
            evaluations.l1Tracking = 'good';
        } else {
            evaluations.l1Tracking = 'poor';
        }

        // Average fee evaluation (now in per-gas gwei)
        if (metrics.avgFee < 0.01) {
            evaluations.avgFee = 'excellent';
        } else if (metrics.avgFee < 0.1) {
            evaluations.avgFee = 'good';
        } else {
            evaluations.avgFee = 'poor';
        }

        return evaluations;
    }

    createL1EstimationChart(canvasId, data) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Use L1 timing: real timestamps for historical data, 12s spacing for simulated L1
        const timeLabels = data.map((d, i) => {
            const seconds = (typeof d.l1ElapsedSeconds === 'number') ? d.l1ElapsedSeconds : (i * 12);
            return `${(seconds / 3600).toFixed(2)}h`;
        });
        const spotBasefeeData = data.map(d => d.l1Basefee / 1e9); // Convert wei to gwei
        const trendBasefeeData = data.map(d => d.l1TrendBasefee / 1e9); // Convert wei to gwei

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [
                    {
                        label: 'Spot L1 Basefee (gwei)',
                        data: spotBasefeeData,
                        borderColor: this.colors.danger,
                        backgroundColor: this.colors.danger + '10',
                        borderWidth: 1.5,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'Trend L1 Basefee (gwei)',
                        data: trendBasefeeData,
                        borderColor: this.colors.primary,
                        backgroundColor: this.colors.primary + '20',
                        borderWidth: 2.5,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'L1 Basefee: Spot vs Trend Estimation',
                        font: { size: 14, weight: 600 },
                        color: '#2d3748',
                        padding: { bottom: 20 }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            font: { size: 11 },
                            color: '#4a5568'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 32, 44, 0.9)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#4299e1',
                        borderWidth: 1,
                        cornerRadius: 6,
                        displayColors: true,
                        callbacks: {
                            title: function(context) {
                                return `Time: ${context[0].label}`;
                            },
                            label: function(context) {
                                const value = context.raw;
                                return `${context.dataset.label}: ${value.toFixed(2)} gwei`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 10 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Basefee (gwei)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 10 },
                            callback: function(value) {
                                return value.toFixed(1);
                            }
                        }
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    createL2FeesChart(canvasId, data, params) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Use L2 timing for fee/deficit composition (Taiko steps = 2s)
        const timeLabels = data.map((d, i) => {
            const seconds = (typeof d.l2ElapsedSeconds === 'number') ? d.l2ElapsedSeconds : (i * 2);
            return `${(seconds / 3600).toFixed(2)}h`;
        });

        // Calculate L1 and deficit components
        const l1Components = data.map(d => {
            const l1Cost = d.estimatedL1Cost || 0;
            return params.mu * l1Cost; // L1 component: μ × L1_cost
        });

        const deficitComponents = data.map(d => {
            const deficit = Math.max(0, d.vaultDeficit || 0);
            return params.nu * deficit / params.H; // Deficit component: ν × deficit/H
        });

        const totalFees = data.map(d => d.estimatedFee || 0);

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [
                    {
                        label: 'Total L2 Estimated Fee',
                        data: totalFees,
                        borderColor: this.colors.primary,
                        backgroundColor: this.colors.primary + '20',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'L1 Cost Component (μ × L1_cost)',
                        data: l1Components,
                        borderColor: this.colors.success,
                        backgroundColor: this.colors.success + '20',
                        borderWidth: 1.5,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderDash: [5, 5]
                    },
                    {
                        label: 'Deficit Component (ν × deficit/H)',
                        data: deficitComponents,
                        borderColor: this.colors.warning,
                        backgroundColor: this.colors.warning + '20',
                        borderWidth: 1.5,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderDash: [10, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'L2 Fee Components Over Time',
                        color: '#2d3748',
                        font: { size: 14, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#4a5568',
                            font: { size: 11 },
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#2d3748',
                        bodyColor: '#4a5568',
                        borderColor: '#e2e8f0',
                        borderWidth: 1,
                        cornerRadius: 6,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label;
                                const value = context.parsed.y;
                                if (value < 1e-6) {
                                    return `${label}: ${(value * 1e9).toFixed(6)} gwei`;
                                } else {
                                    return `${label}: ${value.toFixed(8)} ETH`;
                                }
                            },
                            afterLabel: function(context) {
                                if (context.datasetIndex === 0) { // Total fee
                                    const mu = params.mu;
                                    const nu = params.nu;
                                    return `Parameters: μ=${mu}, ν=${nu}`;
                                }
                                return '';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 10 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Fee (ETH)',
                            color: '#4a5568',
                            font: { size: 12 }
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        ticks: {
                            color: '#718096',
                            font: { size: 10 },
                            callback: function(value) {
                                if (value < 1e-6) {
                                    return (value * 1e9).toFixed(3) + ' gwei';
                                } else {
                                    return value.toExponential(2) + ' ETH';
                                }
                            }
                        }
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    destroyAllCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChartManager };
}
