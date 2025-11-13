#!/bin/bash
# setup_python_env.sh
# Script to set up a system-wide Python 3.10+ environment for Prometheus Unbound.
#
# CRITICAL WARNING: This script installs Python packages globally.
# Execute ONLY on a physically air-gapped, disposable research system.

set -e # Exit immediately if a command exits with a non-zero status.

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root. Please use sudo."
    exit 1
fi

log_info "Starting Python environment setup."

# Ensure python3 is available and points to a recent version
if ! command -v python3 &>/dev/null || ! python3 -c 'import sys; exit(sys.version_info < (3, 10))'; then
    log_error "Python 3.10+ is not found or not the default 'python3' command."
    log_error "Please ensure Python 3.10 or higher is installed and accessible as 'python3'."
    log_error "You might need to install 'python3.10' or 'python3.11' package via apt/pacman if not already done."
    exit 1
fi

# Ensure pip is available
if ! command -v pip3 &>/dev/null; then
    log_info "pip3 not found, installing..."
    if [ -f /etc/arch-release ]; then
        pacman -S --noconfirm python-pip
    elif [ -f /etc/debian_version ]; then
        apt update && apt install -y python3-pip
    else
        log_error "Unsupported OS. Please install pip3 manually."
        exit 1
    fi
fi

# Upgrade pip
log_info "Upgrading pip to the latest version..."
pip3 install --upgrade pip

# Install all required Python packages globally
log_info "Installing required Python packages globally. This may take some time..."
PYTHON_PACKAGES=(
    torch
    transformers
    accelerate
    bitsandbytes
    llama-cpp-python
    chromadb
    networkx
    selenium
    beautifulsoup4
    python-dotenv
    psutil
    pylint # For self_architecting_module static analysis
    mypy   # For self_architecting_module static analysis
)

# Handle torch specific installation for CUDA
log_info "Installing PyTorch with CUDA support. This might be a large download."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
for PACKAGE in "${PYTHON_PACKAGES[@]}"; do
    if [ "$PACKAGE" == "torch" ]; then
        # Skip torch as it's handled separately with CUDA index URL
        continue
    fi
    log_info "Installing $PACKAGE..."
    pip3 install "$PACKAGE"
done

# Install appropriate WebDriver for Selenium (e.g., ChromeDriver)
log_info "Installing Chrome browser and ChromeDriver for Selenium."
if [ -f /etc/arch-release ]; then
    pacman -S --noconfirm chromium chromedriver
elif [ -f /etc/debian_version ]; then
    apt update
    apt install -y chromium chromium-driver
else
    log_info "Unsupported OS. Please manually install a headless browser (e.g., Chromium) and its corresponding WebDriver."
fi

log_info "Python environment setup script completed."
log_info "Verify all packages are installed by trying to import them in a Python shell."