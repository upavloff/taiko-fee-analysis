/**
 * Historical Data Loader for Taiko Fee Analysis
 * Loads and manages the 4 historical datasets used in the research
 */

class HistoricalDataLoader {
    constructor() {
        this.datasets = {};
        this.datasetPaths = {
            'july_2022_spike': 'data/data_cache/real_july_2022_spike_data.csv',
            'luna_crash': 'data/data_cache/luna_crash_true_peak_contiguous.csv',
            'pepe_crisis': 'data/data_cache/may_2023_pepe_crisis_data.csv',
            'normal_operation': 'data/data_cache/recent_low_fees_3hours.csv'
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

            for (const [name, path] of Object.entries(this.datasetPaths)) {
                console.log(`Loading ${name} from ${path}...`);
                this.datasets[name] = await this.loadCsvData(path);
                console.log(`Loaded ${this.datasets[name].length} data points for ${name}`);
            }

            this.loaded = true;
            console.log('All historical datasets loaded successfully');
            return true;
        } catch (error) {
            console.error('Failed to load historical datasets:', error);
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
     * Parse CSV text into structured data
     * @param {string} csvText CSV content
     * @returns {Array} Parsed data points
     */
    parseCsv(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        const data = [];

        // Find column indices (handle different CSV formats)
        const timestampIndex = this.findColumnIndex(headers, ['timestamp']);
        const basefeeWeiIndex = this.findColumnIndex(headers, ['basefee_wei']);
        const basefeeGweiIndex = this.findColumnIndex(headers, ['basefee_gwei']);
        const blockNumberIndex = this.findColumnIndex(headers, ['block_number']);

        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            if (row.length >= headers.length) {
                const dataPoint = {
                    timestamp: row[timestampIndex],
                    basefee_wei: parseFloat(row[basefeeWeiIndex]),
                    basefee_gwei: parseFloat(row[basefeeGweiIndex]),
                    block_number: row[blockNumberIndex]
                };
                data.push(dataPoint);
            }
        }

        return data;
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
}

// Global instance
window.historicalDataLoader = new HistoricalDataLoader();