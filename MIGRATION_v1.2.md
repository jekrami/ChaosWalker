# Migration Guide: v1.1 → v1.2
# Writer: J.Ekrami, Co-writer: Claude Sonnet 4.5
# Date: 2026-01-06

## Overview

Version 1.2 replaces the Gradio-based dashboard with a modern Flask implementation. This guide helps you migrate from v1.1 to v1.2.

## What Changed

### ✅ Added
- `flask_dashboard.py` - New Flask-based web dashboard
- Real-time GPU telemetry (updates every 500ms)
- Browser alert notifications when password found
- Auto-hash generator (type password → hash auto-generated)
- Stop button for graceful engine termination
- Mobile-responsive design

### 🗑️ Removed
- `dashboard.py` - Old Gradio-based dashboard (replaced by Flask)
- Gradio dependency (no longer needed)

### 📝 Updated
- `run_dashboard.sh` - Now launches `flask_dashboard.py` instead of `dashboard.py`
- `requirements.txt` - Flask instead of Gradio
- Port changed: **7860** → **5000**
- `README.md` - Updated with v1.2 features
- `CHANGELOG.md` - Added v1.2.0 entry

## Migration Steps

### 1. Update Dependencies

```bash
# Remove old Gradio installation (optional)
source venv/bin/activate
pip uninstall gradio -y

# Install Flask
pip install flask toml
```

### 2. Update Your Workflow

**Old (v1.1):**
```bash
./run_dashboard.sh
# Opens: http://localhost:7860
```

**New (v1.2):**
```bash
./run_dashboard.sh
# Opens: http://localhost:5000
```

**Port changed from 7860 → 5000**

### 3. Update Bookmarks/Scripts

If you have bookmarks or automation scripts:
- Change `http://localhost:7860` → `http://localhost:5000`
- Change `http://YOUR_IP:7860` → `http://YOUR_IP:5000`

### 4. Update Firewall Rules

If you opened port 7860 in firewall:

```bash
# Ubuntu/Debian
sudo ufw delete allow 7860
sudo ufw allow 5000

# CentOS/RHEL
sudo firewall-cmd --remove-port=7860/tcp --permanent
sudo firewall-cmd --add-port=5000/tcp --permanent
sudo firewall-cmd --reload
```

### 5. Update SSH Tunnels

**Old:**
```bash
ssh -L 7860:localhost:7860 user@server
```

**New:**
```bash
ssh -L 5000:localhost:5000 user@server
```

## Cleanup (Optional)

The following files are old Gradio-related documentation and can be safely removed:

```bash
# Gradio dashboard test files
rm dashboard_simple.py
rm test_dashboard_simple.py
rm test_alert_simple.py

# Old Gradio documentation
rm ALERT_TEST.md
rm DASHBOARD_FIXES.md
rm POPUP_FEATURE.md
rm POPUP_FIX.md
rm QUICK_TEST.md
```

**Keep these files:**
- `flask_dashboard.py` - **Main dashboard** ✅
- `run_dashboard.sh` - Launcher script ✅
- `requirements.txt` - Dependencies ✅
- `DASHBOARD_GUIDE.md` - Flask dashboard guide ✅
- `README_DASHBOARD.md` - Quick start guide ✅

## Feature Comparison

| Feature | v1.1 (Gradio) | v1.2 (Flask) |
|---------|---------------|--------------|
| Web Interface | ✅ | ✅ |
| Real-time Logs | ✅ | ✅ |
| GPU Telemetry | ✅ | ✅ (Faster updates) |
| Auto-hash Generator | ✅ | ✅ |
| Success Notification | ❌ Unreliable | ✅ Browser alert |
| Stop Button | ❌ | ✅ |
| Mobile Responsive | ⚠️ Partial | ✅ Full |
| Load Time | ~3-5s | <1s |
| Dependencies | Heavy (Gradio) | Light (Flask) |

## Benefits of v1.2

1. **Faster**: Flask is lighter and loads instantly
2. **More Reliable**: Browser native alerts always work
3. **Better UX**: Cleaner interface, better mobile support
4. **Simpler**: Fewer dependencies, easier to maintain
5. **Stop Button**: Finally! Gracefully terminate searches

## Troubleshooting

### "No module named 'flask'"

```bash
source venv/bin/activate
pip install flask toml
```

### Port 5000 already in use

Some systems have AirPlay receiver on port 5000 (macOS).

**Option A: Disable AirPlay**
- System Preferences → Sharing → Uncheck AirPlay Receiver

**Option B: Change port**
- Edit `flask_dashboard.py`, last line:
```python
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Old Gradio error messages

If you see Gradio-related errors, you're running the old dashboard:

```bash
# Check which Python script is running
ps aux | grep python

# Make sure run_dashboard.sh calls flask_dashboard.py
grep "flask_dashboard.py" run_dashboard.sh
```

## Rollback (If Needed)

If you need to go back to v1.1:

```bash
# Checkout v1.1 tag
git checkout v1.1.0

# Reinstall Gradio
source venv/bin/activate
pip install gradio toml

# Run old dashboard
python3 dashboard.py
```

## Questions?

- **Documentation**: See `DASHBOARD_GUIDE.md` for complete Flask dashboard guide
- **Main README**: Updated with v1.2 features
- **Changelog**: `CHANGELOG.md` for detailed changes
- **GitHub Issues**: Report bugs or request features

---

**Welcome to ChaosWalker v1.2!** 🌪️

Run `./run_dashboard.sh` and open http://localhost:5000 to try the new Flask dashboard!
