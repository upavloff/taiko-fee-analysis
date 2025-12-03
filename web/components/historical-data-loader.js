/**
 * Historical Data Loader for Taiko Fee Analysis
 * Loads and manages the 4 historical datasets used in the research
 */

class HistoricalDataLoader {
    constructor() {
        this.datasets = {};
        this.datasetPaths = {
            'july_2022_spike': 'data/real_july_2022_spike_data.csv',
            'luna_crash': 'data/luna_crash_true_peak_contiguous.csv',
            'pepe_crisis': 'data/may_2023_pepe_crisis_data.csv',
            'normal_operation': 'data/recent_low_fees_3hours.csv'
        };
        this.loaded = false;
    }

    /**
     * Load all historical datasets
     * @returns {Promise<boolean>} Success status
     */
    async loadAllDatasets() {
        try {
            console.log('Loading historical datasets...');
            this.showDataLoadingStatus('Loading historical datasets...', 'loading');

            for (const [name, path] of Object.entries(this.datasetPaths)) {
                console.log(`Loading ${name} from ${path}...`);
                this.datasets[name] = await this.loadCsvData(path);
                console.log(`Loaded ${this.datasets[name].length} data points for ${name}`);
            }

            this.loaded = true;
            console.log('All historical datasets loaded successfully');
            this.showDataLoadingStatus('✅ Historical data loaded successfully', 'success');
            return true;
        } catch (error) {
            console.error('Failed to load historical datasets:', error);
            this.showDataLoadingStatus('❌ Historical data failed to load. Please serve over HTTP (not file://). Charts will use simulated data.', 'error');
            return false;
        }
    }

    /**
     * Load CSV data from file path
     * @param {string} path File path
     * @returns {Promise<Array>} Parsed data
     */
    async loadCsvData(path) {
        const response = await fetch(path);
        if (!response.ok) {
            throw new Error(`Failed to load ${path}: ${response.statusText}`);
        }

        const csvText = await response.text();
        return this.parseCsv(csvText);
    }

    /**
     * Parse CSV text into structured data with quality validation
     * @param {string} csvText CSV content
     * @returns {Array} Parsed data points
     */
    parseCsv(csvText) {
        // Remove BOM if present and normalize line endings
        const cleanedText = csvText.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        const lines = cleanedText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        const data = [];
        const issues = [];

        // Find column indices (handle different CSV formats)
        const timestampIndex = this.findColumnIndex(headers, ['timestamp']);
        const basefeeWeiIndex = this.findColumnIndex(headers, ['basefee_wei']);
        const basefeeGweiIndex = this.findColumnIndex(headers, ['basefee_gwei']);
        const blockNumberIndex = this.findColumnIndex(headers, ['block_number']);

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue; // Skip empty lines

            const row = line.split(',').map(cell => cell.trim());
            if (row.length >= headers.length) {
                const dataPoint = {
                    timestamp: row[timestampIndex],
                    basefee_wei: parseFloat(row[basefeeWeiIndex]),
                    basefee_gwei: parseFloat(row[basefeeGweiIndex]),
                    block_number: row[blockNumberIndex],
                    row_number: i + 1
                };

                // Data quality validation
                const rowIssues = this.validateDataPoint(dataPoint, i + 1);
                if (rowIssues.length > 0) {
                    issues.push(...rowIssues);
                }

                data.push(dataPoint);
            } else {
                issues.push(`Row ${i + 1}: Insufficient columns (${row.length}/${headers.length})`);
            }
        }

        // Perform additional data quality checks
        const additionalIssues = this.performDataQualityChecks(data);
        issues.push(...additionalIssues);

        // Log data quality results
        if (issues.length === 0) {
            console.log(`✅ Data quality: EXCELLENT - No issues detected`);
        } else {
            console.warn(`⚠️ Data quality issues detected:`, issues);
            console.warn(`🔍 Total issues: ${issues.length} across ${data.length} records`);
        }

        return data;
    }

    /**
     * Validate individual data point
     * @param {Object} dataPoint Data point to validate
     * @param {number} rowNum Row number for error reporting
     * @returns {Array} Array of issues found
     */
    validateDataPoint(dataPoint, rowNum) {
        const issues = [];

        // Check for missing values
        if (!dataPoint.timestamp || dataPoint.timestamp.trim() === '') {
            issues.push(`Row ${rowNum}: Missing timestamp`);
        }
        if (isNaN(dataPoint.basefee_wei) || dataPoint.basefee_wei < 0) {
            issues.push(`Row ${rowNum}: Invalid basefee_wei: ${dataPoint.basefee_wei}`);
        }
        if (isNaN(dataPoint.basefee_gwei) || dataPoint.basefee_gwei < 0) {
            issues.push(`Row ${rowNum}: Invalid basefee_gwei: ${dataPoint.basefee_gwei}`);
        }
        if (!dataPoint.block_number || dataPoint.block_number.trim() === '') {
            issues.push(`Row ${rowNum}: Missing block_number`);
        }

        // Check for unrealistic basefee values
        if (dataPoint.basefee_gwei > 10000) {
            issues.push(`Row ${rowNum}: Unrealistically high basefee: ${dataPoint.basefee_gwei} gwei`);
        }
        if (dataPoint.basefee_gwei > 0 && dataPoint.basefee_gwei < 0.001) {
            issues.push(`Row ${rowNum}: Unrealistically low basefee: ${dataPoint.basefee_gwei} gwei`);
        }

        // Check wei/gwei consistency (allow 0.1% tolerance)
        const expectedGwei = dataPoint.basefee_wei / 1e9;
        const tolerance = Math.max(expectedGwei * 0.001, 0.000001);
        if (Math.abs(dataPoint.basefee_gwei - expectedGwei) > tolerance) {
            issues.push(`Row ${rowNum}: Wei/Gwei mismatch: ${dataPoint.basefee_wei} wei ≠ ${dataPoint.basefee_gwei} gwei`);
        }

        return issues;
    }

    /**
     * Perform comprehensive data quality checks
     * @param {Array} data Dataset to check
     * @returns {Array} Array of issues found
     */
    performDataQualityChecks(data) {
        const issues = [];

        if (data.length === 0) {
            issues.push('Dataset is empty');
            return issues;
        }

        // Check for large basefee jumps (potential data corruption)
        let prevFee = null;
        for (let i = 0; i < data.length; i++) {
            const currentFee = data[i].basefee_gwei;
            if (prevFee !== null && currentFee > 0 && prevFee > 0) {
                const ratio = currentFee / prevFee;
                if (ratio > 10 || ratio < 0.1) {
                    issues.push(`Rows ${i}-${i+1}: Large basefee jump: ${prevFee.toFixed(3)} → ${currentFee.toFixed(3)} gwei (${ratio.toFixed(1)}x)`);
                }
            }
            prevFee = currentFee;
        }

        // Check timestamp ordering
        for (let i = 1; i < data.length; i++) {
            const prevTime = new Date(data[i-1].timestamp);
            const currTime = new Date(data[i].timestamp);
            if (currTime <= prevTime) {
                issues.push(`Rows ${i}-${i+1}: Timestamps not in ascending order`);
            }
        }

        // Calculate basic statistics for anomaly detection
        const basefees = data.map(d => d.basefee_gwei).filter(f => f > 0);
        if (basefees.length > 0) {
            const mean = basefees.reduce((a, b) => a + b) / basefees.length;
            const std = Math.sqrt(basefees.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / basefees.length);
            const outlierThreshold = mean + 5 * std;

            const outliers = data.filter(d => d.basefee_gwei > outlierThreshold);
            if (outliers.length > 0) {
                issues.push(`Statistical outliers detected: ${outliers.length} values >5σ from mean`);
            }
        }

        return issues;
    }

    /**
     * Find column index by name variants
     * @param {Array} headers Header names
     * @param {Array} possibleNames Possible column names
     * @returns {number} Column index
     */
    findColumnIndex(headers, possibleNames) {
        for (const name of possibleNames) {
            const index = headers.findIndex(h => h.toLowerCase().includes(name.toLowerCase()));
            if (index !== -1) return index;
        }
        throw new Error(`Could not find column matching any of: ${possibleNames.join(', ')}`);
    }

    /**
     * Get dataset by name
     * @param {string} name Dataset name
     * @returns {Array} Dataset
     */
    getDataset(name) {
        if (!this.loaded) {
            throw new Error('Datasets not loaded yet. Call loadAllDatasets() first.');
        }
        return this.datasets[name];
    }

    /**
     * Get all dataset names
     * @returns {Array} Dataset names
     */
    getDatasetNames() {
        return Object.keys(this.datasetPaths);
    }

    /**
     * Get dataset statistics
     * @param {string} name Dataset name
     * @returns {Object} Statistics
     */
    getDatasetStats(name) {
        const data = this.getDataset(name);
        const basefees = data.map(d => d.basefee_gwei);

        return {
            name: name,
            count: data.length,
            duration_hours: data.length * 2 / 3600, // Assuming 2s per step
            min_basefee_gwei: Math.min(...basefees),
            max_basefee_gwei: Math.max(...basefees),
            avg_basefee_gwei: basefees.reduce((a, b) => a + b, 0) / basefees.length,
            median_basefee_gwei: this.median(basefees),
            std_basefee_gwei: this.standardDeviation(basefees)
        };
    }

    /**
     * Calculate median
     * @param {Array} arr Numbers
     * @returns {number} Median
     */
    median(arr) {
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    /**
     * Calculate standard deviation
     * @param {Array} arr Numbers
     * @returns {number} Standard deviation
     */
    standardDeviation(arr) {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
        return Math.sqrt(variance);
    }

    /**
     * Get combined statistics for all datasets
     * @returns {Object} Combined statistics
     */
    getAllDatasetStats() {
        const stats = {};
        for (const name of this.getDatasetNames()) {
            stats[name] = this.getDatasetStats(name);
        }
        return stats;
    }

    /**
     * Show data loading status message in UI
     * @param {string} message Status message
     * @param {string} type Status type: 'loading', 'success', 'error'
     */
    showDataLoadingStatus(message, type) {
        // Try to find existing status element or create one
        let statusElement = document.getElementById('data-loading-status');

        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.id = 'data-loading-status';
            statusElement.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                padding: 12px 16px;
                border-radius: 6px;
                font-family: monospace;
                font-size: 14px;
                font-weight: bold;
                z-index: 10000;
                max-width: 400px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: opacity 0.3s ease;
            `;

            // Insert at top of body
            if (document.body) {
                document.body.insertBefore(statusElement, document.body.firstChild);
            }
        }

        // Set styling based on type
        const styles = {
            loading: { bg: '#e3f2fd', border: '#2196f3', color: '#0d47a1' },
            success: { bg: '#e8f5e8', border: '#4caf50', color: '#2e7d32' },
            error: { bg: '#ffebee', border: '#f44336', color: '#c62828' }
        };

        const style = styles[type] || styles.error;
        statusElement.style.backgroundColor = style.bg;
        statusElement.style.border = `2px solid ${style.border}`;
        statusElement.style.color = style.color;
        statusElement.textContent = message;
        statusElement.style.opacity = '1';

        // Auto-hide success/error messages after delay
        if (type !== 'loading') {
            setTimeout(() => {
                if (statusElement) {
                    statusElement.style.opacity = '0';
                    setTimeout(() => {
                        if (statusElement && statusElement.parentNode) {
                            statusElement.parentNode.removeChild(statusElement);
                        }
                    }, 300);
                }
            }, type === 'success' ? 3000 : 8000);
        }
    }
}

// Global instance
window.historicalDataLoader = new HistoricalDataLoader();