# 🖥️ ChaosWalker Flask Dashboard - Quick Start

**Version 1.2.0** - Modern Flask-based web interface

## One-Command Launch

```bash
./run_dashboard.sh
```

Then open: **http://localhost:5000**

## What It Does

The Flask dashboard provides a clean web UI for ChaosWalker with:

- 🎯 **Easy Target Setup**: Type password → SHA-256 hash auto-generated
- ⚡ **Real-time Monitoring**: Live GPU temperature, load, VRAM (updates every 500ms)
- 🚀 **One-Click Control**: Start/Stop buttons
- 🚨 **Instant Alerts**: Browser popup when password found
- 📊 **Live Logs**: Watch the engine work in real-time
- 📱 **Mobile Friendly**: Responsive design works on any device

## First Time Setup

The launcher script does everything automatically:

```bash
# Just run this once
./run_dashboard.sh
```

It will:
1. Create virtual environment
2. Install dependencies (flask, toml)
3. Build ChaosWalker if needed
4. Launch Flask dashboard

**Takes ~1 minute first time, then instant!**

## Usage

### 1. Start Dashboard
```bash
./run_dashboard.sh
```

### 2. Open Browser
```
http://localhost:5000
```

### 3. Set Target

**Option A:** Type password (e.g., "test123")
- SHA-256 hash auto-generated in real-time

**Option B:** Paste SHA-256 hash
- Direct entry (manual mode)

### 4. Start Engine
Click **🚀 START ENGINE**

### 5. Watch It Work
- Hash rate updates live (every 500ms)
- GPU telemetry (temperature, load, VRAM)
- Logs scroll in real-time
- **Browser alert popup** when password found!

### 6. Stop Engine
Click **🛑 STOP ENGINE** for graceful termination

## Example Session

```
1. Open: http://localhost:5000
2. Enter: "admin"
3. Auto-hash: 8c6976e5b5410415...
4. Click: 🚀 START ENGINE
5. Watch: GPU stats update, logs scroll
6. Result: 🎉 Browser alert popup + password displayed!
```

## Access from Phone/Laptop

The dashboard is network-accessible:

```bash
# Get your server IP
hostname -I
# Example: 192.168.1.100

# Open on phone/laptop
http://192.168.1.100:5000
```

## Troubleshooting

### Dashboard Won't Start

```bash
# Make script executable
chmod +x run_dashboard.sh

# Run manually
source venv/bin/activate
python3 dashboard.py
```

### "Module not found"

```bash
source venv/bin/activate
pip install flask toml
```

### Port Already Used

Edit `flask_dashboard.py`, bottom of file:
```python
app.run(host='0.0.0.0', port=8080, debug=False)  # Change from 5000
```

## Files

- `flask_dashboard.py` - **Main Flask dashboard** ⭐
- `run_dashboard.sh` - Launcher script (recommended)
- `requirements.txt` - Dependencies (flask, toml)
- `smart_mapper.py` - Smart Mapper library

## Pro Tips

1. **Build first** for faster startup:
   ```bash
   cargo build --release
   ./run_dashboard.sh
   ```

2. **Run persistent** with tmux:
   ```bash
   tmux new -s dashboard
   ./run_dashboard.sh
   # Detach: Ctrl+B then D
   ```

3. **Monitor GPU** while running:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Remote access** securely:
   ```bash
   ssh -L 5000:localhost:5000 user@server
   # Then: http://localhost:5000
   ```

## Screenshots

### Main Interface
```
┌─────────────────────────────────────────────────┐
│ 🌪️ ChaosWalker v1.2 Flask Dashboard            │
├─────────────────────────────────────────────────┤
│ 🎉 PASSWORD FOUND!                              │
│    Password: admin                              │
├─────────────────────────────────────────────────┤
│ Target Password: [admin          ]              │
│ SHA-256 Hash:    [8c6976e5b54... ]              │
│ [🚀 START ENGINE]  [🛑 STOP ENGINE]            │
│ Status: 🎉 FOUND!                               │
├─────────────────────────────────────────────────┤
│ 🖥️ GPU Telemetry                                │
│ GPU #0: 72°C    99%    695/24576 MB             │
├─────────────────────────────────────────────────┤
│ System Logs:                                    │
│ Engine started...                               │
│ Target loaded. Engine started.                  │
│ Checked: 50.0 M | Speed: 1234.56 M/sec          │
│ !!! SUCCESS !!!                                 │
│ Target Found at Index: 1065825710               │
└─────────────────────────────────────────────────┘
```

## Status

**✅ FULLY FUNCTIONAL (v1.2)**

All features working:
- ✅ Flask-based dashboard
- ✅ Real-time GPU telemetry
- ✅ Browser alert notifications
- ✅ Auto-hash generator
- ✅ Stop button
- ✅ Mobile responsive

## Learn More

- **`README.md`** - Main ChaosWalker documentation
- **`CHANGELOG.md`** - Version history and changes

---

**Ready to crack?** Run `./run_dashboard.sh` and open http://localhost:5000! 🚀
