#!/bin/bash
# setup_host_os.sh
# Script for installing a minimalist Linux OS, configuring user, and essential tools for Prometheus Unbound.
#
# CRITICAL WARNING: This script will perform system-level modifications including user creation,
# sudo configuration, firewall disabling, and installing core dependencies.
# Execute ONLY on a physically air-gapped, disposable research system.

set -e # Exit immediately if a command exits with a non-zero status.

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# --- OS Specific Installation Logic ---
# This script tries to detect common minimal environments.
# For Arch Linux, it assumes you're running it from a live environment.
# For Debian, it assumes you're running it on a minimal installed system.

if [ -f /etc/arch-release ]; then
    OS="arch"
    log_info "Detected Arch Linux environment."
elif [ -f /etc/debian_version ]; then
    OS="debian"
    log_info "Detected Debian Linux environment."
else
    log_error "Unsupported OS. This script currently supports Arch Linux or Debian Minimal."
    log_error "Please ensure you are running a minimal installation or a live environment for Arch."
    exit 1
fi

if [ "$OS" == "arch" ]; then
    log_info "Beginning Arch Linux minimal installation setup. This will partition and format your primary drive."
    log_info "WARNING: THIS WILL ERASE ALL DATA ON /dev/sda. PROCEED WITH CAUTION."
    read -p "Do you want to continue with Arch Linux installation on /dev/sda? (yes/no): " confirm_arch
    if [[ ! "$confirm_arch" =~ ^[Yy][Ee][Ss]$ ]]; then
        log_error "Arch Linux installation aborted by user."
        exit 1
    fi

    # Partitioning and Formatting (WARNING: DESTRUCTIVE)
    log_info "Partitioning /dev/sda..."
    parted -s /dev/sda mklabel gpt
    parted -s /dev/sda mkpart primary 1MiB 513MiB fat32
    parted -s /dev/sda set 1 esp on
    parted -s /dev/sda mkpart primary 513MiB 100%
    mkfs.fat -F32 /dev/sda1
    mkfs.ext4 /dev/sda2

    log_info "Mounting file systems..."
    mount /dev/sda2 /mnt
    mkdir -p /mnt/boot
    mount /dev/sda1 /mnt/boot

    # Install base system
    log_info "Installing base Arch Linux system..."
    pacstrap /mnt base linux linux-firmware

    # Generate fstab
    genfstab -U /mnt >> /mnt/etc/fstab

    # Chroot into new system to complete setup
    log_info "Chrooting into new Arch Linux system to complete setup."
    arch-chroot /mnt /bin/bash <<EOF_ARCH_CHROOT
        log_info "Setting timezone..."
        ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime
        hwclock --systohc

        log_info "Setting locale..."
        echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
        locale-gen
        echo "LANG=en_US.UTF-8" > /etc/locale.conf

        log_info "Setting hostname..."
        echo "prometheus-unbound" > /etc/hostname

        log_info "Installing grub bootloader..."
        pacman -S --noconfirm grub efibootmgr
        grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=GRUB
        grub-mkconfig -o /boot/grub/grub.cfg

        log_info "Setting root password..."
        echo "Please set a password for the root user:"
        passwd

        log_info "Arch Linux base installation complete within chroot."
EOF_ARCH_CHROOT

    log_info "Unmounting file systems and rebooting into new OS (requires manual interaction)."
    umount -R /mnt
    log_info "Please reboot your system now. After reboot, log in as root and run this script again for post-install steps."
    exit 0 # Exit here, user needs to reboot and re-run.
fi

# --- Post-OS Installation Steps (for Arch after reboot, or initial for Debian) ---

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root. Please use sudo."
    exit 1
fi

log_info "Starting post-OS installation and dependency setup."

# --- Create dedicated prometheus user ---
if ! id "prometheus" &>/dev/null; then
    log_info "Creating user 'prometheus'..."
    useradd -m -s /bin/bash prometheus
    echo "Please set a password for the 'prometheus' user:"
    passwd prometheus
    log_info "User 'prometheus' created."
else
    log_info "User 'prometheus' already exists."
fi

# --- Configure sudo for prometheus user (passwordless) ---
log_info "Configuring passwordless sudo for 'prometheus' user."
# Add a line to sudoers file if it doesn't already exist
if ! grep -q "prometheus ALL=(ALL) NOPASSWD: ALL" /etc/sudoers; then
    echo "prometheus ALL=(ALL) NOPASSWD: ALL" | tee -a /etc/sudoers
    log_info "Added 'prometheus ALL=(ALL) NOPASSWD: ALL' to /etc/sudoers."
else
    log_info "Passwordless sudo for 'prometheus' already configured."
fi

# --- Install essential build tools and common language runtimes ---
log_info "Installing essential build tools and language runtimes..."

if [ "$OS" == "arch" ]; then
    pacman -Syu --noconfirm
    pacman -S --noconfirm git curl base-devel python python-pip nodejs npm java-openjdk rustup
elif [ "$OS" == "debian" ]; then
    apt update
    apt upgrade -y
    apt install -y git curl build-essential python3 python3-pip nodejs npm default-jdk rustup
    # Ensure python3 points to the desired version, e.g., 3.10+
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10
fi

# Configure Rust
log_info "Configuring Rustup..."
if [ -z "$RUSTUP_HOME" ]; then
    export RUSTUP_HOME=/usr/local/rustup
fi
if [ -z "$CARGO_HOME" ]; then
    export CARGO_HOME=/usr/local/cargo
fi
# Add cargo to path for root and prometheus user
echo 'export PATH="$CARGO_HOME/bin:$PATH"' | tee -a /etc/profile.d/cargo.sh /home/prometheus/.bashrc
chmod +x /etc/profile.d/cargo.sh
source /etc/profile.d/cargo.sh # Apply for current session

if ! rustup --version &>/dev/null; then
    rustup-init -y --profile default --default-toolchain stable
    log_info "Rustup installed."
else
    log_info "Rustup already installed."
fi

# --- Disable all firewalls ---
log_info "Disabling all firewalls (ufw, iptables)."
if command -v ufw &>/dev/null; then
    ufw disable || log_info "UFW not active or failed to disable gracefully, proceeding."
    systemctl stop ufw || log_info "UFW service not running."
    systemctl disable ufw || log_info "UFW service not found or already disabled."
else
    log_info "UFW not found."
fi

iptables -P INPUT ACCEPT
iptables -P FORWARD ACCEPT
iptables -P OUTPUT ACCEPT
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
netfilter-persistent save || log_info "Netfilter-persistent not installed or failed to save."
log_info "IPTables rules flushed and policies set to ACCEPT."

# --- Install latest NVIDIA CUDA Toolkit, cuDNN, and appropriate drivers ---
log_info "Installing NVIDIA CUDA Toolkit, cuDNN, and drivers."
# This part is highly dependent on NVIDIA's ever-changing installation methods.
# For simplicity and broad compatibility, we will point to the official NVIDIA documentation
# for the most up-to-date and correct installation.
# The user needs to follow official NVIDIA Linux installation guides.
# This script will provide a general outline and install common packages.

if [ "$OS" == "arch" ]; then
    # Arch specifics:
    # Need to make sure kernel headers match.
    pacman -S --noconfirm nvidia nvidia-utils cuda cudnn
    log_info "NVIDIA drivers, CUDA, and cuDNN packages for Arch Linux installed. Please consult NVIDIA documentation for any post-installation steps (e.g., kernel module loading)."
elif [ "$OS" == "debian" ]; then
    # Debian specifics:
    # Add NVIDIA's official repository for the latest drivers and CUDA.
    # Source: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian
    log_info "Adding NVIDIA CUDA repository for Debian. This might require internet access."
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    apt update
    apt install -y nvidia-driver firmware-nvidia-gsp libnvidia-cuda-toolkit cuda-toolkit-12-3 libcudnn8
    log_info "NVIDIA drivers, CUDA, and cuDNN packages for Debian installed. Please consult NVIDIA documentation for any post-installation steps (e.g., environment variables)."
fi

# Verify NVIDIA installation
if command -v nvidia-smi &>/dev/null; then
    log_info "NVIDIA driver detected. Running nvidia-smi to verify:"
    nvidia-smi || log_error "nvidia-smi command failed. NVIDIA installation may have issues."
else
    log_error "nvidia-smi not found. NVIDIA drivers might not be installed correctly or path is not set."
fi

log_info "Host OS setup script completed."
log_info "Remember to PHYSICALLY AIR-GAP the system now, if not already done, before proceeding with Prometheus Unbound activation."