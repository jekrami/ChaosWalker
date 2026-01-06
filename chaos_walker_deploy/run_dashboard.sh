#!/bin/bash
# ChaosWalker Flask Dashboard Launcher
# Version: 1.2.0
# Description: Launches the Flask-based web dashboard for ChaosWalker
# Writer: J.Ekrami, Co-writer: Claude Sonnet 4.5

cd "$(dirname "$0")"

echo "🌪️  ChaosWalker Dashboard Launcher v1.2"
echo "======================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    echo "📥 Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install flask toml -q
    echo "✅ Dependencies installed!"
else
    echo "✅ Virtual environment found"
    source venv/bin/activate
fi

# Check if binary exists
if [ ! -f "./target/release/chaos_walker" ]; then
    echo "⚠️  Warning: chaos_walker binary not found"
    echo "   Building now (this may take a minute)..."
    cargo build --release
    echo "✅ Build complete!"
fi

echo ""
echo "🚀 Launching Flask dashboard..."
echo "   Access at: http://localhost:5000"
echo "   Or from network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 flask_dashboard.py
