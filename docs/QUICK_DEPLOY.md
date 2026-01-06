# ChaosWalker - Quick Deployment Reference
# Version: 1.2.0

## 🚀 Deploy in 5 Steps

### 1️⃣ Build (Local Machine)
```bash
cd /home/ekrami/ChaosWalker
cargo build --release
./create_deploy_package.sh
```

### 2️⃣ Transfer to Remote Server
```bash
scp chaos_walker_v1.2_runtime.tar.gz user@remote-server:~/
```

### 3️⃣ Setup on Remote Server
```bash
ssh user@remote-server
tar -xzf chaos_walker_v1.2_runtime.tar.gz
cd chaos_walker_deploy
```

### 4️⃣ Configure & Test
```bash
# Verify GPU
nvidia-smi

# Set CUDA path (if needed)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Edit target hash
nano config.toml

# Test run
./chaos_walker
```

### 5️⃣ Run Dashboard (Optional)
```bash
# Install Flask
pip3 install flask toml

# Start dashboard
python3 flask_dashboard.py

# Access via SSH tunnel (from local machine)
ssh -L 5000:localhost:5000 user@remote-server
# Open: http://localhost:5000
```

---

## 📦 What Gets Deployed

```
chaos_walker_deploy/
├── chaos_walker              ← Binary (10-20 MB)
├── config.toml              ← Configuration
├── kernels/
│   ├── chaos_worker_linear.ptx   ← PTX kernel (~50 KB)
│   └── chaos_worker_random.ptx   ← PTX kernel (~50 KB)
├── smart_mapper.py          ← Decoder library
├── decode_result.py         ← CLI decoder
└── flask_dashboard.py       ← Web UI
```

**Total size: ~10-20 MB**

---

## 🔧 Essential Commands

```bash
# Run engine (CLI)
./chaos_walker

# Run dashboard
python3 flask_dashboard.py

# Decode found password
python3 decode_result.py <INDEX>

# Check GPU status
nvidia-smi

# Stop engine
pkill chaos_walker
```

---

## ⚡ Quick Troubleshooting

### "CUDA error"
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "PTX not found"
```bash
ls kernels/  # Should see .ptx files
pwd          # Must be in chaos_walker_deploy directory
```

### "Permission denied"
```bash
chmod +x chaos_walker
```

### Dashboard port 5000 in use
```bash
# Edit flask_dashboard.py, change port to 8080
nano flask_dashboard.py
```

---

## 🔒 Secure Remote Access

**Recommended: SSH Tunnel**
```bash
# From your local machine
ssh -L 5000:localhost:5000 user@remote-server

# Open browser
http://localhost:5000
```

**No firewall changes needed!**

---

## 📖 Full Documentation

See **`DEPLOYMENT_GUIDE.md`** for complete instructions including:
- systemd service setup
- Security configurations
- Performance tuning
- Multi-GPU setup
- Production checklist

---

**That's it!** Your ChaosWalker is now running on a remote GPU server. 🎉
