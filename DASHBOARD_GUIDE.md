# ChaosWalker Flask Dashboard Guide
# Version: 1.2.0
# Writer: J.Ekrami, Co-writer: Claude Sonnet 4.5
# Date: 2026-01-06

## Overview

The ChaosWalker Flask Dashboard provides a modern, responsive web-based UI for controlling the password cracker with real-time GPU monitoring and instant notifications.

## Features

- 🎯 **Target Configuration**: Enter password (auto-generates hash) or paste SHA-256 hash
- ⚡ **Live GPU Telemetry**: Real-time temperature, load, and VRAM usage (updates every 500ms)
- 🚀 **Engine Control**: Start/Stop buttons with graceful shutdown
- 🚨 **Instant Alerts**: Browser popup notification when password found
- 📊 **System Logs**: Live console output from the engine
- 📱 **Mobile Responsive**: Works on phones, tablets, and desktops
- 🎨 **Dark Theme**: Easy on the eyes for extended monitoring

## Installation

### Quick Start (Automated)

```bash
./run_dashboard.sh
```

This will:
1. Create virtual environment (if needed)
2. Install dependencies (flask, toml)
3. Build ChaosWalker (if needed)
4. Launch the Flask dashboard on http://localhost:5000

### Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install flask toml

# 3. Build ChaosWalker
cargo build --release

# 4. Run dashboard
python3 flask_dashboard.py
```

## Usage

### 1. Start the Dashboard

```bash
./run_dashboard.sh
```

Or manually:
```bash
source venv/bin/activate
python3 flask_dashboard.py
```

### 2. Access the UI

Open in browser:
- **Local**: http://localhost:5000
- **Network**: http://YOUR_IP:5000

### 3. Set Target

**Option A: Enter Password (Recommended)**
- Type password in "Target Password" field (e.g., "admin")
- SHA-256 hash automatically generated as you type
- Updates in real-time

**Option B: Enter Hash Directly**
- Paste SHA-256 hash in "SHA-256 Hash" field
- 64 hexadecimal characters

### 4. Start Engine

Click **🚀 START ENGINE**

The dashboard will:
- Update config.toml with your target hash
- Launch ChaosWalker engine as subprocess
- Start streaming logs to the UI
- Begin GPU telemetry monitoring

Watch:
- System logs update in real-time
- GPU temperature, load, VRAM refresh every 500ms
- Status updates as engine searches

### 5. When Password Found

You'll see:
- 🎉 **Browser Alert Popup**: "PASSWORD FOUND! Password: [result]"
- **Big Green Box**: Displays the found password
- **Status Update**: "🎉 FOUND!"
- **Logs**: Shows success message and index

### 6. Stop Engine

Click **🛑 STOP ENGINE** to:
- Send SIGTERM to running process
- Gracefully terminate search
- Update status to "Engine stopped"

## GPU Telemetry

### Display Format
```
GPU #0: 72°C    99%    695/24576 MB
        ↑       ↑      ↑
        Temp    Load   VRAM Used/Total
```

### Temperature
- **Normal**: 50-75°C under load
- **Warning**: 80°C+ (consider improving cooling)
- **Throttle**: 85°C+ (GPU may reduce performance)

### Load (Utilization)
- **Target**: 95-100% for maximum performance
- **Low (<50%)**: Increase batch size in config.toml
- **Fluctuating**: Normal during checkpoint saves

### VRAM (GPU Memory)
- **ChaosWalker usage**: ~100-200 MB
- **Available**: Rest available for other tasks
- **Full**: Reduce batch size if memory errors occur

## Troubleshooting

### Dashboard Won't Start

**Error: "No module named 'flask'"**
```bash
source venv/bin/activate
pip install flask toml
```

**Error: "Permission denied"**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Error: "Address already in use (port 5000)"**
- Another process using port 5000
- Edit `flask_dashboard.py`, change last line:
```python
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Engine Won't Start

**Error: "chaos_walker binary not found"**
```bash
cargo build --release
```

**Error: "CUDA error" in logs**
```bash
# Check if GPU is available
nvidia-smi

# Check CUDA toolkit
nvcc --version
```

### GPU Stats Show "N/A"

**Cause:** nvidia-smi not found or GPU not available

**Fix:**
```bash
# Test nvidia-smi
nvidia-smi

# If not found, add to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Or install NVIDIA drivers
# Ubuntu/Debian:
sudo apt install nvidia-driver-XXX
```

### No Alert When Password Found

**Cause:** Browser popup blocked

**Fix:**
- Check browser address bar for blocked popup icon
- Allow popups for localhost:5000
- Check browser console (F12) for JavaScript errors

### Config.toml Errors

**Error: "Could not write config.toml"**

**Fixes:**
```bash
# Check permissions
ls -la config.toml

# Make writable
chmod 644 config.toml

# Validate format
cat config.toml
```

### Network Access Issues

**Can't access from other devices:**

1. **Check firewall:**
```bash
# Ubuntu/Debian
sudo ufw allow 5000

# CentOS/RHEL
sudo firewall-cmd --add-port=5000/tcp --permanent
sudo firewall-cmd --reload
```

2. **Verify binding:**
- Dashboard binds to `0.0.0.0:5000` (all interfaces)
- Should be accessible from network

3. **Get server IP:**
```bash
hostname -I
# Example: 192.168.1.100
# Access: http://192.168.1.100:5000
```

## Advanced Configuration

### Change Port

Edit `flask_dashboard.py`, last line:
```python
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Use Pre-built Binary

Dashboard automatically uses `./target/release/chaos_walker` if available.

For development build (slower):
```python
command = ["cargo", "run", "--release"]
```

### Customize Update Interval

Edit `flask_dashboard.py`:
```javascript
// GPU telemetry update interval (milliseconds)
setInterval(updateGpuStats, 500);  // Change 500 to desired ms
```

### Enable Debug Mode

For development/troubleshooting:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

**Warning:** Don't use debug mode in production!

## Architecture

### Process Flow

```
User → Browser → Flask App → config.toml → ChaosWalker Engine
                    ↓                            ↓
               nvidia-smi ← GPU Stats ← Multi-GPU Workers
```

### Files

- `flask_dashboard.py`: Main Flask application
- `run_dashboard.sh`: Launcher script (recommended)
- `requirements.txt`: Python dependencies
- `config.toml`: Engine configuration (auto-updated by dashboard)
- `smart_mapper.py`: Password encoding/decoding library

### Process Management

- Dashboard spawns ChaosWalker as subprocess with process group
- Uses `os.setsid()` for clean process group termination
- Monitors stdout/stderr in separate thread
- Graceful shutdown with SIGTERM signal
- GPU stats polled via nvidia-smi subprocess

## Best Practices

### 1. Build Before Running Dashboard

```bash
cargo build --release  # Faster startup, better performance
./run_dashboard.sh
```

### 2. Monitor GPU Temperature

- Keep below 80°C for GPU longevity
- Improve case cooling if consistently high
- Check thermal paste if overheating persists

### 3. Batch Size Tuning

Edit `config.toml` for optimal performance:

| GPU Model | Recommended Batch Size |
|-----------|------------------------|
| RTX 3060  | 5M - 10M               |
| RTX 3070  | 10M - 15M              |
| RTX 3080  | 15M - 20M              |
| RTX 3090  | 20M - 30M              |
| RTX 4090  | 30M - 50M              |

### 4. Remote Access via SSH Tunnel (Secure)

**From your laptop:**
```bash
ssh -L 5000:localhost:5000 user@gpu-server

# Then open browser:
http://localhost:5000
```

Benefits:
- Encrypted connection
- No firewall changes needed
- Secure over internet

### 5. Persistent Sessions with tmux

**Run dashboard persistently:**
```bash
# Start tmux session
tmux new -s chaoswalker

# Run dashboard
./run_dashboard.sh

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t chaoswalker
```

## Performance Tips

1. **Close other GPU applications** (browsers with hardware acceleration, games, ML tools)
2. **Use compiled binary** (`./target/release/chaos_walker`)
3. **Optimize batch size** in config.toml
4. **Monitor GPU utilization** - should be 95-100%
5. **Check PCIe link speed**: `nvidia-smi -q | grep PCIe` (should be Gen3 x16 or better)
6. **Disable GPU power limits** for maximum performance (if safe):
```bash
sudo nvidia-smi -pl 350  # Example: 350W for RTX 3090
```

## Security Considerations

### Network Exposure

The dashboard binds to `0.0.0.0`, making it accessible from any network interface.

**For production/public servers:**
1. **Use SSH tunnel** (recommended)
2. **Add authentication** (modify flask_dashboard.py)
3. **Use reverse proxy** with HTTPS (nginx + Let's Encrypt)
4. **Firewall rules** (restrict to specific IPs)

### Example: Add Basic Auth

```python
from flask import request, Response

def check_auth(username, password):
    return username == 'admin' and password == 'secure_password'

def authenticate():
    return Response('Login required', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

@app.before_request
def require_auth():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()
```

## Example Sessions

### Quick Test (Password: "test")

```bash
# Terminal
./run_dashboard.sh

# Browser: http://localhost:5000
1. Enter "test" in Target Password field
2. Hash auto-fills: 9f86d081...
3. Click "🚀 START ENGINE"
4. Watch logs and GPU stats
5. Alert popup: "PASSWORD FOUND! Password: test"
6. Green box shows: "Password: test"
```

### Known Password Length

If you know password is 8 characters:

```bash
# Edit config.toml
known_password_length = 8

# Start dashboard normally
./run_dashboard.sh
```

This optimizes search to start at 8-character passwords.

## Conclusion

The ChaosWalker Flask Dashboard makes GPU password cracking accessible through an intuitive, modern web interface with real-time monitoring and instant notifications.

**Quick Start:**
```bash
./run_dashboard.sh
# Open http://localhost:5000
# Enter password → Click START → Get instant alert when found! 🚀
```

---

For issues or contributions:
- **GitHub**: https://github.com/jekrami/ChaosWalker
- **Email**: ekrami@gmail.com
- **Main Docs**: README.md, CHANGELOG.md
