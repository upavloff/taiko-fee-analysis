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

    createFeeChart(canvasId, data) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        const timeLabels = data.map((_, i) => `${(i * 12 / 3600).toFixed(1)}h`);
        const feeData = data.map(d => d.estimatedFee);

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'Estimated Fee (ETH)',
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
                        text: 'Fee Evolution Over Time',
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
                                return `Fee: ${value.toExponential(3)} ETH`;
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
                            text: 'Fee (ETH)',
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
                        type: 'logarithmic'
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

    createVaultChart(canvasId, data, targetBalance = 1000) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        const timeLabels = data.map((_, i) => `${(i * 12 / 3600).toFixed(1)}h`);
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
                                return `${context.dataset.label}: ${value.toFixed(1)} ETH`;
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

        const timeLabels = data.map((_, i) => `${(i * 12 / 3600).toFixed(1)}h`);
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

    createCorrelationChart(canvasId, data) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Create scatter plot data
        const scatterData = data.map(d => ({
            x: d.l1Basefee / 1e9, // gwei
            y: d.estimatedFee
        }));

        this.charts[canvasId] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Fee vs L1 Basefee',
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
                        text: 'Fee vs L1 Basefee Correlation',
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
                                    `Fee: ${point.y.toExponential(3)} ETH`
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
                            text: 'Estimated Fee (ETH)',
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
                        type: 'logarithmic'
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
            formattedValue = value.toExponential(2);
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

        // Average fee evaluation (relative)
        if (metrics.avgFee < 1e-5) {
            evaluations.avgFee = 'excellent';
        } else if (metrics.avgFee < 1e-4) {
            evaluations.avgFee = 'good';
        } else {
            evaluations.avgFee = 'poor';
        }

        return evaluations;
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