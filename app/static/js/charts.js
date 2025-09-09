/**
 * Chart.js visualization functions for the cost forecasting dashboard
 * Provides utility functions to create and manage various chart types
 */

// Chart.js default configuration
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.position = 'top';

/**
 * Create a cost trends line chart
 * @param {HTMLCanvasElement} ctx - Canvas context
 * @param {Object} data - Chart data containing dates and costs
 * @param {string} view - View type ('total' or 'components')
 * @returns {Chart} Chart.js instance
 */
function createTrendsChart(ctx, data, view = 'total') {
    const config = {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return window.formatCurrency(value);
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#374151',
                    bodyColor: '#6B7280',
                    borderColor: '#E5E7EB',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + window.formatCurrency(context.parsed.y);
                        }
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                }
            },
            elements: {
                point: {
                    radius: 3,
                    hoverRadius: 6
                },
                line: {
                    borderWidth: 2
                }
            }
        }
    };
    
    if (view === 'total') {
        config.data.datasets = [{
            label: 'Total Daily Cost',
            data: data.total_costs,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.3,
            fill: true
        }];
    } else {
        config.data.datasets = [
            {
                label: 'CPU Cost',
                data: data.cpu_costs,
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.3
            },
            {
                label: 'Memory Cost',
                data: data.ram_costs,
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.3
            },
            {
                label: 'Bandwidth Cost',
                data: data.bandwidth_costs,
                borderColor: 'rgb(139, 92, 246)',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.3
            }
        ];
    }
    
    return new Chart(ctx, config);
}

/**
 * Create a server comparison doughnut chart
 * @param {HTMLCanvasElement} ctx - Canvas context
 * @param {Object} data - Chart data containing server costs
 * @returns {Chart} Chart.js instance
 */
function createServerComparisonChart(ctx, data) {
    if (!data.server_costs || Object.keys(data.server_costs).length === 0) {
        return createEmptyChart(ctx, 'No server data available');
    }
    
    const serverNames = Object.keys(data.server_costs);
    const serverTotals = serverNames.map(server => {
        const costs = data.server_costs[server];
        return Array.isArray(costs) ? costs.reduce((sum, cost) => sum + cost, 0) : 0;
    });
    
    // Filter out servers with zero costs
    const filteredData = serverNames
        .map((name, index) => ({ name, total: serverTotals[index] }))
        .filter(server => server.total > 0);
    
    if (filteredData.length === 0) {
        return createEmptyChart(ctx, 'No cost data available');
    }
    
    const colors = [
        'rgba(59, 130, 246, 0.8)',   // Blue
        'rgba(16, 185, 129, 0.8)',   // Green
        'rgba(139, 92, 246, 0.8)',   // Purple
        'rgba(245, 158, 11, 0.8)',   // Amber
        'rgba(239, 68, 68, 0.8)',    // Red
        'rgba(6, 182, 212, 0.8)'     // Cyan
    ];
    
    const borderColors = [
        'rgba(59, 130, 246, 1)',
        'rgba(16, 185, 129, 1)',
        'rgba(139, 92, 246, 1)',
        'rgba(245, 158, 11, 1)',
        'rgba(239, 68, 68, 1)',
        'rgba(6, 182, 212, 1)'
    ];
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: filteredData.map(server => formatServerName(server.name)),
            datasets: [{
                data: filteredData.map(server => server.total),
                backgroundColor: colors.slice(0, filteredData.length),
                borderColor: borderColors.slice(0, filteredData.length),
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#374151',
                    bodyColor: '#6B7280',
                    borderColor: '#E5E7EB',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((sum, val) => sum + val, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return context.label + ': ' + window.formatCurrency(context.parsed) + ' (' + percentage + '%)';
                        }
                    }
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                }
            }
        }
    });
}

/**
 * Create an empty chart with a message
 * @param {HTMLCanvasElement} ctx - Canvas context
 * @param {string} message - Message to display
 * @returns {Chart} Chart.js instance
 */
function createEmptyChart(ctx, message) {
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [message],
            datasets: [{
                data: [1],
                backgroundColor: ['rgba(156, 163, 175, 0.3)'],
                borderColor: ['rgba(156, 163, 175, 0.5)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });
}

/**
 * Create a forecast chart with confidence intervals
 * @param {HTMLCanvasElement} ctx - Canvas context
 * @param {Array} forecast - Forecast data with date, predicted_cost, confidence_lower, confidence_upper
 * @returns {Chart} Chart.js instance
 */
function createForecastChart(ctx, forecast) {
    if (!forecast || forecast.length === 0) {
        return createEmptyChart(ctx, 'No forecast data available');
    }
    
    const dates = forecast.map(f => f.date);
    const predictions = forecast.map(f => f.predicted_cost);
    const lowerBounds = forecast.map(f => f.confidence_lower);
    const upperBounds = forecast.map(f => f.confidence_upper);
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Predicted Cost',
                    data: predictions,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.3,
                    fill: false,
                    borderWidth: 3,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Upper Confidence',
                    data: upperBounds,
                    borderColor: 'rgba(59, 130, 246, 0.4)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: '+1',
                    tension: 0.3,
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5]
                },
                {
                    label: 'Lower Confidence',
                    data: lowerBounds,
                    borderColor: 'rgba(59, 130, 246, 0.4)',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    fill: false,
                    tension: 0.3,
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5]
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
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return window.formatCurrency(value);
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                }
            },
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#374151',
                    bodyColor: '#6B7280',
                    borderColor: '#E5E7EB',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + window.formatCurrency(context.parsed.y);
                        }
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        filter: function(item) {
                            // Hide the lower confidence line from legend
                            return item.text !== 'Lower Confidence';
                        }
                    }
                }
            },
            elements: {
                point: {
                    radius: 3,
                    hoverRadius: 6
                }
            }
        }
    });
}

/**
 * Create a resource usage gauge chart
 * @param {HTMLCanvasElement} ctx - Canvas context
 * @param {number} value - Usage value (0-100)
 * @param {string} label - Chart label
 * @param {string} color - Chart color
 * @returns {Chart} Chart.js instance
 */
function createGaugeChart(ctx, value, label, color = 'rgb(59, 130, 246)') {
    const remaining = 100 - value;
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [label, 'Remaining'],
            datasets: [{
                data: [value, remaining],
                backgroundColor: [color, 'rgba(229, 231, 235, 0.3)'],
                borderColor: [color, 'rgba(229, 231, 235, 0.5)'],
                borderWidth: 2,
                circumference: 180,
                rotation: 270
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataIndex === 0) {
                                return label + ': ' + value.toFixed(1) + '%';
                            }
                            return null;
                        }
                    }
                },
                legend: { display: false }
            }
        }
    });
}

/**
 * Format server name for display
 * @param {string} serverName - Raw server name
 * @returns {string} Formatted server name
 */
function formatServerName(serverName) {
    return serverName
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

/**
 * Destroy chart safely
 * @param {Chart} chart - Chart instance to destroy
 */
function destroyChart(chart) {
    if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
    }
}

/**
 * Update chart data
 * @param {Chart} chart - Chart instance
 * @param {Object} newData - New chart data
 */
function updateChartData(chart, newData) {
    if (chart && chart.data) {
        chart.data = newData;
        chart.update('none');
    }
}

// Export functions for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createTrendsChart,
        createServerComparisonChart,
        createForecastChart,
        createGaugeChart,
        createEmptyChart,
        formatServerName,
        destroyChart,
        updateChartData
    };
}