import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // Build configuration that outputs to root level (where existing workflow expects)
  root: '.',
  build: {
    outDir: '../build',   // Output to build/ directory first
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html')
      },
      output: {
        // Output files directly to root (matching current structure)
        entryFileNames: 'app.js',
        chunkFileNames: '[name].js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.name.endsWith('.css')) {
            return 'styles.css';
          }
          return '[name][extname]';
        }
      }
    }
  },

  // Development server configuration
  server: {
    port: 3000,
    open: true,
    // Serve data files from parent directory during development
    fs: {
      allow: ['..']
    }
  },

  // Static asset handling
  publicDir: 'public',

  // CSS processing
  css: {
    devSourcemap: true
  }
});