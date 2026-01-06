# ChaosWalker Deployment Guide
# Writer: J.Ekrami, Co-writer: Claude Sonnet 4.5
# Date: 2026-01-06
# Version: 1.2.0

## Overview

This guide shows how to deploy **ChaosWalker runtime binaries** to a remote GPU server without copying the entire source code.

---

## 📦 What Files to Deploy

### Minimum Required Files (CLI Only)

```
chaos_walker/
├── chaos_walker              # Main executable binary
├── config.toml              # Configuration file
├── kernels/
│   ├── chaos_worker_linear.ptx   # Linear mode kernel
│   └── chaos_worker_random.ptx   # Random mode kernel
├── smart_mapper.py          # Password encoding/decoding library
└── decode_result.py         # Result decoder utility
```

### Full Deployment (With Web Dashboard)

```
chaos_walker/
├── chaos_walker              # Main executable binary
├── config.toml              # Configuration file
├── kernels/
│   ├── chaos_worker_linear.ptx   # Linear mode kernel
│   └── chaos_worker_random.ptx   # Random mode kernel
├── smart_mapper.py          # Password encoding/decoding library
├── decode_result.py         # Result decoder utility
├── flask_dashboard.py       # Web dashboard
├── requirements.txt         # Python dependencies
└── run_dashboard.sh         # Dashboard launcher (optional)
```

---

## 🚀 Step-by-Step Deployment

### Step 1: Build on Development Machine

On your **local machine** (where you have the source code):

```bash
# Navigate to project directory
cd /home/ekrami/ChaosWalker

# Build release version
cargo build --release

# Verify binary exists
ls -lh target/release/chaos_walker

# Verify PTX kernels exist
ls -lh kernels/*.ptx
```

Expected output:
```
-rwxr-xr-x  chaos_walker (10-20 MB)
-rw-r--r--  chaos_worker_linear.ptx (~50-100 KB)
-rw-r--r--  chaos_worker_random.ptx (~50-100 KB)
```

---

### Step 2: Create Deployment Package

**Option A: Manual Copy**

```bash
# Create deployment directory
mkdir -p ~/chaos_deploy

# Copy binary
cp target/release/chaos_walker ~/chaos_deploy/

# Copy kernels
mkdir -p ~/chaos_deploy/kernels
cp kernels/chaos_worker_linear.ptx ~/chaos_deploy/kernels/
cp kernels/chaos_worker_random.ptx ~/chaos_deploy/kernels/

# Copy configuration and utilities
cp config.toml ~/chaos_deploy/
cp smart_mapper.py ~/chaos_deploy/
cp decode_result.py ~/chaos_deploy/

# For web dashboard (optional)
cp flask_dashboard.py ~/chaos_deploy/
cp requirements.txt ~/chaos_deploy/
cp run_dashboard.sh ~/chaos_deploy/

# Verify structure
tree ~/chaos_deploy
```

**Option B: Automated Script**

```bash
# Run this script from project root
cat > create_deploy_package.sh << 'EOF'
#!/bin/bash
# ChaosWalker Deployment Package Creator
# Version: 1.2.0

DEPLOY_DIR="chaos_walker_deploy"

echo "🌪️  Creating ChaosWalker deployment package..."

# Create directory structure
mkdir -p $DEPLOY_DIR/kernels

# Copy binary
echo "📦 Copying binary..."
cp target/release/chaos_walker $DEPLOY_DIR/

# Copy kernels
echo "🔧 Copying PTX kernels..."
cp kernels/chaos_worker_linear.ptx $DEPLOY_DIR/kernels/
cp kernels/chaos_worker_random.ptx $DEPLOY_DIR/kernels/

# Copy configuration
echo "⚙️  Copying configuration..."
cp config.toml $DEPLOY_DIR/

# Copy Python utilities
echo "🐍 Copying Python utilities..."
cp smart_mapper.py $DEPLOY_DIR/
cp decode_result.py $DEPLOY_DIR/

# Copy dashboard (optional)
echo "🌐 Copying web dashboard..."
cp flask_dashboard.py $DEPLOY_DIR/
cp requirements.txt $DEPLOY_DIR/
cp run_dashboard.sh $DEPLOY_DIR/ 2>/dev/null || true

# Make executable
chmod +x $DEPLOY_DIR/chaos_walker
chmod +x $DEPLOY_DIR/run_dashboard.sh 2>/dev/null || true

# Create archive
echo "📦 Creating tarball..."
tar -czf chaos_walker_v1.2_runtime.tar.gz $DEPLOY_DIR/

echo "✅ Deployment package created!"
echo "   Directory: $DEPLOY_DIR/"
echo "   Archive:   chaos_walker_v1.2_runtime.tar.gz"
echo ""
echo "📤 Transfer to remote server:"
echo "   scp chaos_walker_v1.2_runtime.tar.gz user@remote-server:~/"

EOF

chmod +x create_deploy_package.sh
./create_deploy_package.sh
```

---

### Step 3: Transfer to Remote Server

**Using SCP:**

```bash
# Transfer deployment package
scp chaos_walker_v1.2_runtime.tar.gz user@remote-server.com:~/

# Or transfer directory directly
scp -r chaos_walker_deploy/ user@remote-server.com:~/
```

**Using rsync (recommended for updates):**

```bash
# First time
rsync -avz --progress chaos_walker_deploy/ user@remote-server.com:~/chaos_walker/

# Updates (only changed files)
rsync -avz --progress chaos_walker_deploy/ user@remote-server.com:~/chaos_walker/
```

---

### Step 4: Setup on Remote Server

SSH into the remote server:

```bash
ssh user@remote-server.com
```

**Extract and setup:**

```bash
# If using tarball
tar -xzf chaos_walker_v1.2_runtime.tar.gz
cd chaos_walker_deploy

# Or if transferred directory
cd chaos_walker

# Verify files
ls -lh
ls -lh kernels/

# Make binary executable (if needed)
chmod +x chaos_walker
```

**Verify NVIDIA GPU and CUDA:**

```bash
# Check GPU availability
nvidia-smi

# Check CUDA version
nvcc --version  # Should be CUDA 11.0+

# If CUDA not in PATH, add it:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

### Step 5: Configure for Remote Server

**Edit `config.toml`:**

```bash
nano config.toml
```

```toml
# Set your target hash
target_hash = "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"

# Adjust batch size for your GPU
batch_size = 10000000  # 10M for RTX 3090, adjust as needed

# Select kernel mode
gpu_ptx_path = "./kernels/chaos_worker_linear.ptx"  # or chaos_worker_random.ptx

# Optional: If you know password length
known_password_length = 0  # Set to 8 for 8-char passwords, 0 to disable

# Checkpoint settings
checkpoint_file = "chaos_state.txt"
checkpoint_interval_secs = 30
```

---

### Step 6: Test CLI Mode

**Basic test:**

```bash
# Run directly
./chaos_walker
```

Expected output:
```
--- Project ChaosWalker v1.2: Multi-GPU Edition ---

No checkpoint found. Starting from beginning.

Target loaded. Engine started.
Batch Size: 10000000 keys/cycle
Checkpoint: Saving every 30 seconds to chaos_state.txt

🎉 Found 1 NVIDIA GPU(s):
  [0] NVIDIA GeForce RTX 3090 (82 SMs, 24 GB)

Worker 0 starting from index: 0

Checked: 10.0 M | Speed: 1234.56 M/sec | Offset: 10000000
Checked: 20.0 M | Speed: 1256.78 M/sec | Offset: 20000000
```

**If it works, decode result:**

```bash
# When password found:
python3 decode_result.py <RANDOM_INDEX>
```

---

### Step 7: Setup Web Dashboard (Optional)

**Install Python dependencies:**

```bash
# Check Python version (need 3.8+)
python3 --version

# Install dependencies
pip3 install flask toml

# Or use requirements.txt
pip3 install -r requirements.txt
```

**Run dashboard:**

```bash
# Start dashboard
python3 flask_dashboard.py
```

Expected output:
```
🌪️  ChaosWalker Flask Dashboard
Starting server...

 * Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

**Access from your local machine:**

```bash
# Option 1: Direct access (if firewall allows)
http://remote-server-ip:5000

# Option 2: SSH tunnel (secure, recommended)
ssh -L 5000:localhost:5000 user@remote-server.com
# Then open: http://localhost:5000
```

---

## 🔧 Troubleshooting

### Problem: "CUDA error" when running

**Solution 1: Check CUDA libraries**
```bash
# Find CUDA libraries
find /usr -name "libcuda.so*" 2>/dev/null
find /usr -name "libcudart.so*" 2>/dev/null

# Add to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Test
./chaos_walker
```

**Solution 2: Install CUDA toolkit**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-0
```

---

### Problem: "PTX kernel not found"

**Cause:** Binary looking for kernels in wrong location

**Solution:** Ensure directory structure:
```bash
pwd  # Should be in deployment directory

ls -la
# Should see: chaos_walker, config.toml, kernels/

ls -la kernels/
# Should see: chaos_worker_linear.ptx, chaos_worker_random.ptx

# Fix config.toml if needed
nano config.toml
# Ensure: gpu_ptx_path = "./kernels/chaos_worker_linear.ptx"
```

---

### Problem: "Permission denied" when running

**Solution:**
```bash
chmod +x chaos_walker
./chaos_walker
```

---

### Problem: Dashboard port 5000 already in use

**Solution 1: Change port**
```bash
# Edit flask_dashboard.py, last line
nano flask_dashboard.py
# Change: app.run(host='0.0.0.0', port=8080, debug=False)
```

**Solution 2: Kill existing process**
```bash
# Find process using port 5000
lsof -i :5000

# Kill it
kill -9 <PID>
```

---

### Problem: "No module named 'flask'"

**Solution:**
```bash
# Install Flask
pip3 install flask toml

# If pip3 not found
sudo apt install python3-pip
pip3 install flask toml
```

---

## 🔒 Security Considerations

### Firewall Configuration

**If accessing dashboard remotely:**

```bash
# Ubuntu/Debian
sudo ufw allow 5000/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=5000/tcp --permanent
sudo firewall-cmd --reload
```

**Recommended: Use SSH tunnel instead:**
```bash
# From local machine
ssh -L 5000:localhost:5000 user@remote-server.com
# Access: http://localhost:5000 (secure, encrypted)
```

---

### Running as Service (systemd)

**Create service file:**

```bash
sudo nano /etc/systemd/system/chaoswalker.service
```

```ini
[Unit]
Description=ChaosWalker GPU Password Cracker
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/chaos_walker
ExecStart=/home/your-username/chaos_walker/chaos_walker
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable chaoswalker
sudo systemctl start chaoswalker

# Check status
sudo systemctl status chaoswalker

# View logs
sudo journalctl -u chaoswalker -f
```

---

### Running Dashboard as Service

```bash
sudo nano /etc/systemd/system/chaoswalker-dashboard.service
```

```ini
[Unit]
Description=ChaosWalker Web Dashboard
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/chaos_walker
ExecStart=/usr/bin/python3 /home/your-username/chaos_walker/flask_dashboard.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable chaoswalker-dashboard
sudo systemctl start chaoswalker-dashboard
sudo systemctl status chaoswalker-dashboard
```

---

## 📊 Performance Tuning

### Batch Size Optimization

Test different batch sizes for your GPU:

```bash
# Edit config.toml
nano config.toml

# Try different values:
# batch_size = 5000000   # 5M - Lower GPU usage
# batch_size = 10000000  # 10M - Recommended
# batch_size = 20000000  # 20M - Maximum throughput
# batch_size = 50000000  # 50M - For high-end GPUs
```

Monitor GPU with:
```bash
watch -n 1 nvidia-smi
```

Target: **95-100% GPU utilization**

---

### Multi-GPU Configuration

ChaosWalker automatically detects all GPUs. To verify:

```bash
./chaos_walker
# Should show: "Found 2 NVIDIA GPU(s):" (or however many you have)
```

No configuration needed - automatic load balancing!

---

## 🧹 Cleanup and Maintenance

### Disk Space Management

```bash
# Remove old checkpoints (if starting fresh)
rm chaos_state.txt

# Monitor disk usage
du -sh .
```

### Log Rotation

```bash
# If running as service, logs managed by systemd
sudo journalctl --vacuum-time=7d  # Keep 7 days of logs
```

---

## 📋 Quick Reference

### Directory Structure (Deployed)

```
chaos_walker/                    # Deployment directory
├── chaos_walker                # Binary (10-20 MB)
├── config.toml                 # Configuration
├── kernels/
│   ├── chaos_worker_linear.ptx   # ~50 KB
│   └── chaos_worker_random.ptx   # ~50 KB
├── smart_mapper.py             # Python library
├── decode_result.py            # Decoder utility
├── flask_dashboard.py          # Web dashboard (optional)
├── requirements.txt            # Python deps (optional)
└── chaos_state.txt             # Checkpoint (created at runtime)
```

### Essential Commands

```bash
# Run CLI
./chaos_walker

# Run dashboard
python3 flask_dashboard.py

# Decode result
python3 decode_result.py <INDEX>

# Check GPU
nvidia-smi

# Check logs (if service)
sudo journalctl -u chaoswalker -f

# Stop engine
pkill chaos_walker
```

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] GPU drivers installed and tested (`nvidia-smi`)
- [ ] CUDA libraries accessible (`LD_LIBRARY_PATH`)
- [ ] Binary executable (`chmod +x chaos_walker`)
- [ ] PTX kernels in `kernels/` directory
- [ ] `config.toml` configured with correct target hash
- [ ] Python 3.8+ installed (for utilities/dashboard)
- [ ] Flask installed (if using dashboard): `pip3 install flask toml`
- [ ] Firewall configured (if remote dashboard access)
- [ ] SSH tunnel setup (for secure remote access)
- [ ] Tested CLI run successfully
- [ ] Tested dashboard access (if applicable)
- [ ] systemd service configured (for production)
- [ ] Backup strategy for checkpoints

---

## 🆘 Support

If issues persist:

1. **Check logs**: `./chaos_walker 2>&1 | tee chaos.log`
2. **Verify GPU**: `nvidia-smi`
3. **Check libraries**: `ldd chaos_walker`
4. **Test kernel**: Run with small batch first (1M)
5. **GitHub Issues**: Report bugs with full error output

---

## 📝 Example: Complete Deployment Session

```bash
# === LOCAL MACHINE ===
cd /home/ekrami/ChaosWalker
cargo build --release
./create_deploy_package.sh
scp chaos_walker_v1.2_runtime.tar.gz user@gpu-server.com:~/

# === REMOTE SERVER ===
ssh user@gpu-server.com
tar -xzf chaos_walker_v1.2_runtime.tar.gz
cd chaos_walker_deploy
nvidia-smi  # Verify GPU
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nano config.toml  # Set target hash
./chaos_walker  # Test run
python3 -m pip install flask toml  # For dashboard
python3 flask_dashboard.py &  # Start dashboard in background

# === LOCAL MACHINE ===
ssh -L 5000:localhost:5000 user@gpu-server.com  # SSH tunnel
# Open browser: http://localhost:5000
```

---

**Success!** 🎉 ChaosWalker is now running on your remote GPU server!

For questions or issues, see the main README.md or GitHub Issues.
