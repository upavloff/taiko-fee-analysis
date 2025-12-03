# Taiko Fee Analysis - Repository Architecture

This document describes the restructured repository architecture designed for professional development while maintaining GitHub Pages compatibility.

## 🏗️ Repository Structure

```
taiko-fee-analysis/
├── 📁 web_src/                        # Web Development Source
│   ├── components/                     # Modular JavaScript components
│   │   ├── historical-data-loader.js  # Data loading utilities
│   │   ├── taiko-simulator-js.js      # Core fee simulation engine
│   │   ├── metrics-framework-js.js    # Performance metrics calculation
│   │   ├── simulator.js               # Main UI simulator
│   │   ├── charts.js                  # Visualization components
│   │   ├── pareto-visualizer.js       # Multi-objective visualization
│   │   ├── nsga-ii-web.js             # Optimization algorithms
│   │   ├── optimization-research.js   # Research optimization framework
│   │   └── app.js                     # Application controller
│   ├── styles/                        # CSS organization
│   │   └── styles.css                 # Main stylesheet
│   ├── tests/                         # Web component tests
│   ├── utils/                         # Utility modules
│   ├── main.js                        # Development entry point
│   ├── index.html                     # Development HTML template
│   ├── package.json                   # Build tool configuration
│   ├── vite.config.js                 # Bundler configuration
│   └── build.js                       # Custom build script
├── 📁 python/                         # Python Package
│   ├── taiko_fee/                     # Main package (was src/)
│   │   ├── __init__.py               # Package initialization
│   │   ├── core/                     # Fee mechanism simulation
│   │   ├── data/                     # Data fetching & processing
│   │   ├── analysis/                 # Performance metrics
│   │   └── utils/                    # Helper functions
│   ├── tests/                        # Python test suite
│   ├── pyproject.toml               # Python package configuration
│   └── README.md                    # Python package documentation
├── 📁 research/                       # Research Workspace
│   ├── notebooks/                    # Jupyter notebooks
│   ├── experiments/                  # Ad-hoc analysis scripts
│   ├── papers/                       # Research documents
│   ├── results/                      # Generated outputs
│   └── README.md                     # Research documentation
├── 📁 data/                          # Centralized Data Management
│   ├── cache/                        # Processed cache files (.csv)
│   ├── raw/                          # Original raw datasets
│   ├── external/                     # External data sources
│   ├── results/                      # Analysis outputs
│   └── README.md                     # Data documentation
├── 📁 docs/                          # Documentation
│   └── research/                     # Research findings
├── 📁 .github/                       # CI/CD & GitHub Configuration
│   └── workflows/
│       └── deploy.yml                # Enhanced deployment workflow
├── 📄 index.html                     # 🏗️ Generated (from web_src/)
├── 📄 app.js                         # 🏗️ Generated (from web_src/)
├── 📄 styles.css                     # 🏗️ Generated (from web_src/)
├── 📄 CANONICAL_FEE_MECHANISM_SPEC.md # Single source of truth
├── 📄 ARCHITECTURE.md                # This document
├── 📄 README.md                      # Project overview
├── 📄 CLAUDE.md                      # Project context
└── 📄 .gitignore                     # Enhanced ignore rules
```

## 🔄 Build & Deploy Pipeline

### Development Workflow
```bash
# Web development
cd web_src
npm install
npm run dev          # Development server with hot reload

# Python development
cd python
pip install -e .     # Editable installation
pytest tests/        # Run tests

# Research
jupyter notebook research/notebooks/
```

### Production Build
```bash
cd web_src
npm run build        # Generates index.html, app.js, styles.css at root
```

### GitHub Pages Deployment
```yaml
# .github/workflows/deploy.yml automatically:
1. Checks out code
2. Sets up Node.js
3. Installs dependencies (web_src/package.json)
4. Runs build (web_src/build.js)
5. Creates optimized deployment
6. Deploys to GitHub Pages
```

## 🎯 Key Design Principles

### 1. **Separation of Concerns**
- **Web**: Clean modular development in `web_src/`
- **Python**: Professional package structure in `python/`
- **Research**: Dedicated workspace in `research/`
- **Data**: Centralized management in `data/`

### 2. **Build-Time Generation**
```
Source files (web_src/) → Build process → Production files (root/)
```
- Development happens in organized `web_src/` structure
- Build process generates the exact files GitHub Pages expects
- No manual file management at root level

### 3. **Backward Compatibility**
- GitHub Pages workflow unchanged (still deploys from root)
- Same file structure expected by existing deployment
- Zero downtime migration path

### 4. **Professional Standards**
- Modern build tooling (Vite/Node.js)
- Proper package management
- Clean import/export patterns
- Comprehensive documentation

## 📊 Data Flow Architecture

### Web Application Data Flow
```
data/cache/*.csv → web_src/components/historical-data-loader.js → Charts & Simulation
```

### Python Package Data Flow
```
data/raw/ → python/taiko_fee/data/ → python/taiko_fee/core/ → data/results/
```

### Research Data Flow
```
data/cache/ → research/notebooks/ → research/results/ → research/papers/
```

## 🏭 Development vs Production

### Development Environment
- **Web**: `web_src/` with hot reload and modular imports
- **Python**: Editable install with development dependencies
- **Research**: Direct notebook access to all components

### Production Environment
- **Web**: Concatenated `app.js` with all components
- **Python**: Installed package with clean imports
- **Research**: Reproducible with documented dependencies

## 🔐 Security & Best Practices

### Secrets Management
- No API keys in source code
- Environment variables for sensitive data
- Secure data access patterns

### Code Quality
- ESLint configuration for JavaScript
- Python code follows PEP 8 standards
- Comprehensive test coverage
- Documentation standards enforced

## 🚀 Deployment Architecture

### GitHub Pages Integration
```
Repository Root (GitHub Pages serves from here)
├── index.html      ← Generated by web_src/build.js
├── app.js          ← Generated from web_src/components/*.js
├── styles.css      ← Generated from web_src/styles/styles.css
└── data_cache/     ← Symlinked to data/cache/ for web access
```

### CI/CD Pipeline
1. **Code Push** → triggers GitHub Actions
2. **Build Step** → compiles web_src/ to root files
3. **Test Step** → validates Python package and tests
4. **Deploy Step** → GitHub Pages serves built files
5. **Zero Downtime** → instant switching between versions

## 📋 Migration Benefits

### Before Restructure
❌ **Scattered files at root level**
❌ **No build process or tooling**
❌ **Mixed concerns (web + Python)**
❌ **Manual file management**
❌ **No development workflow**

### After Restructure
✅ **Clean development structure**
✅ **Professional build pipeline**
✅ **Separated concerns**
✅ **Automated file generation**
✅ **Modern development workflow**
✅ **Maintained GitHub Pages compatibility**

## 🤝 Contributing Guidelines

### Web Development
1. Work in `web_src/components/`
2. Test with `npm run dev`
3. Build with `npm run build`
4. Verify output at root level

### Python Development
1. Work in `python/taiko_fee/`
2. Install with `pip install -e python/`
3. Test with `pytest python/tests/`
4. Update imports as needed

### Research Work
1. Create notebooks in `research/notebooks/`
2. Document methodology clearly
3. Save results in `research/results/`
4. Update findings in `research/papers/`

## 🎛️ Configuration Files

### Web Build Configuration
- **`web_src/package.json`**: Build scripts and dependencies
- **`web_src/build.js`**: Custom build logic for GitHub Pages
- **`web_src/vite.config.js`**: Development server configuration

### Python Package Configuration
- **`python/pyproject.toml`**: Package metadata and dependencies
- **`python/setup.py`**: Installation configuration

### Repository Configuration
- **`.gitignore`**: Ignore built files and dependencies
- **`.github/workflows/deploy.yml`**: Enhanced deployment pipeline

---

This architecture provides a **professional development experience** while maintaining **full GitHub Pages compatibility**, enabling the project to scale with modern development practices.

*Last updated: December 2025*