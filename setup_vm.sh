#!/bin/bash
# ============================================================
# EXPAI Inference VM Setup Script
# Run this ONCE inside the VM after first SSH connection.
# Usage: bash setup_vm.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="/mnt/data/expai-inference"
VENV_DIR="/mnt/data/expai-env"

echo "=== [1/6] System update & deps ==="
sudo apt-get update -qq && sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq \
    build-essential linux-headers-$(uname -r) \
    python3.11 python3.11-venv python3-pip \
    git curl wget unzip

echo "=== [2/6] NVIDIA drivers ==="
if ! nvidia-smi &>/dev/null; then
    sudo apt-get install -y nvidia-driver-535
    echo "⚠  Reboot required after driver install. Re-run this script after reboot."
    echo "   sudo reboot"
    exit 0
else
    echo "NVIDIA driver already installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi

echo "=== [3/6] Mount data disk ==="
if ! mountpoint -q /mnt/data; then
    sudo mkfs.ext4 -m 0 -E lazy_itable_init=0 -F /dev/sdb
    sudo mkdir -p /mnt/data
    sudo mount /dev/sdb /mnt/data
    echo "/dev/sdb /mnt/data ext4 defaults 0 2" | sudo tee -a /etc/fstab
    sudo chown -R $USER:$USER /mnt/data
    echo "Data disk mounted at /mnt/data"
else
    echo "Data disk already mounted."
fi

echo "=== [4/6] Python venv ==="
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== [5/6] Project setup ==="
mkdir -p "$PROJECT_DIR"/{models,videos,outputs/{annotated_videos,reports,logs}}

# Install Python deps
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install --upgrade pip -q
    pip install -r "$PROJECT_DIR/requirements.txt" -q
    echo "Dependencies installed."
else
    echo "⚠  requirements.txt not found at $PROJECT_DIR — sync the code first."
fi

echo "=== [6/6] Systemd service ==="
sudo tee /etc/systemd/system/expai-inference.service > /dev/null <<EOF
[Unit]
Description=EXPAI Inference API
After=network.target

[Service]
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_DIR/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
Environment=EXPAI_ENV=production
Environment=EXPAI_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable expai-inference
sudo systemctl start expai-inference || true

echo ""
echo "✅ Setup complete!"
echo "   API: http://$(curl -s ifconfig.me):8000/health"
echo "   Logs: sudo journalctl -u expai-inference -f"
echo "   Project: $PROJECT_DIR"
