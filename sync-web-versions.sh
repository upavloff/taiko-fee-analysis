#!/bin/bash
# sync-web-versions.sh
# Synchronizes web_src development files to root production files
# Usage: ./sync-web-versions.sh [direction]
#   - sync-web-versions.sh src-to-root    (default: web_src -> root)
#   - sync-web-versions.sh root-to-src    (root -> web_src)

set -e  # Exit on any error

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Taiko Web Interface Sync Tool${NC}"
echo "=========================================="

DIRECTION=${1:-src-to-root}

sync_src_to_root() {
    echo -e "${BLUE}📁 Syncing web_src → root (production)${NC}"

    if [ ! -f "web_src/index.html" ]; then
        echo -e "${RED}❌ Error: web_src/index.html not found${NC}"
        exit 1
    fi

    # Copy HTML and fix paths
    echo "  📄 Copying index.html..."
    cp web_src/index.html index.html

    # Fix file references for root deployment
    echo "  🔧 Adjusting file paths for root deployment..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's|styles/styles\.css|styles.css|g' index.html
        sed -i '' 's|type="module" src="main\.js"|src="app.js"|g' index.html
    else
        # Linux
        sed -i 's|styles/styles\.css|styles.css|g' index.html
        sed -i 's|type="module" src="main\.js"|src="app.js"|g' index.html
    fi

    echo -e "${GREEN}✅ web_src → root sync completed${NC}"
    echo -e "${YELLOW}⚠️  Note: You may need to manually update app.js if component files changed${NC}"
}

sync_root_to_src() {
    echo -e "${BLUE}📁 Syncing root → web_src (development)${NC}"

    if [ ! -f "index.html" ]; then
        echo -e "${RED}❌ Error: root index.html not found${NC}"
        exit 1
    fi

    # Copy HTML and fix paths
    echo "  📄 Copying index.html..."
    cp index.html web_src/index.html

    # Fix file references for web_src structure
    echo "  🔧 Adjusting file paths for web_src structure..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's|href="styles\.css"|href="styles/styles.css"|g' web_src/index.html
        sed -i '' 's|src="app\.js"|type="module" src="main.js"|g' web_src/index.html
    else
        # Linux
        sed -i 's|href="styles\.css"|href="styles/styles.css"|g' web_src/index.html
        sed -i 's|src="app\.js"|type="module" src="main.js"|g' web_src/index.html
    fi

    echo -e "${GREEN}✅ root → web_src sync completed${NC}"
    echo -e "${YELLOW}⚠️  Note: Component files in web_src/components/ are not synced${NC}"
}

case $DIRECTION in
    src-to-root)
        sync_src_to_root
        ;;
    root-to-src)
        sync_root_to_src
        ;;
    *)
        echo -e "${RED}❌ Error: Invalid direction '$DIRECTION'${NC}"
        echo "Usage: $0 [src-to-root|root-to-src]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}📊 Current status:${NC}"
echo "  Root index.html:    $(stat -f%Sm index.html 2>/dev/null || stat -c %y index.html 2>/dev/null || echo 'Not found')"
echo "  Web_src index.html: $(stat -f%Sm web_src/index.html 2>/dev/null || stat -c %y web_src/index.html 2>/dev/null || echo 'Not found')"
echo ""
echo -e "${GREEN}🚀 Sync complete! Remember:${NC}"
echo "  • GitHub Pages deploys from ROOT files"
echo "  • Test both versions locally before pushing"
echo "  • Keep CLAUDE.md updated with any architectural changes"