#!/bin/bash
# ChaosWalker Deployment Package Creator
# Writer: J.Ekrami, Co-writer: Claude Sonnet 4.5
# Date: 2026-01-06
# Version: 1.2.0

DEPLOY_DIR="chaos_walker_deploy"

echo "🌪️  Creating ChaosWalker deployment package..."

# Clean old deployment
if [ -d "$DEPLOY_DIR" ]; then
    echo "🧹 Cleaning old deployment directory..."
    rm -rf $DEPLOY_DIR
fi

# Create directory structure
mkdir -p $DEPLOY_DIR/kernels

# Copy binary
echo "📦 Copying binary..."
if [ ! -f "target/release/chaos_walker" ]; then
    echo "❌ Error: Binary not found. Run 'cargo build --release' first!"
    exit 1
fi
cp target/release/chaos_walker $DEPLOY_DIR/

# Copy kernels
echo "🔧 Copying PTX kernels..."
if [ ! -f "kernels/chaos_worker_linear.ptx" ]; then
    echo "❌ Error: PTX kernels not found. Build failed?"
    exit 1
fi
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

# Calculate size
SIZE=$(du -sh chaos_walker_v1.2_runtime.tar.gz | cut -f1)

echo ""
echo "✅ Deployment package created!"
echo "   Directory: $DEPLOY_DIR/"
echo "   Archive:   chaos_walker_v1.2_runtime.tar.gz ($SIZE)"
echo ""
echo "📤 Transfer to remote server:"
echo "   scp chaos_walker_v1.2_runtime.tar.gz user@remote-server:~/"
echo ""
echo "   Or use rsync for updates:"
echo "   rsync -avz $DEPLOY_DIR/ user@remote-server:~/chaos_walker/"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for complete instructions"
