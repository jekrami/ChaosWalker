#!/bin/bash
# Simple script to show when password is found

cd /home/ekrami/ChaosWalker

echo "Running ChaosWalker..."
./target/release/chaos_walker | while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == *"SUCCESS"* ]] || [[ "$line" == *"Target Found"* ]]; then
        # Extract index
        index=$(echo "$line" | grep -oP 'Random Index:\s*\K\d+')
        if [ ! -z "$index" ]; then
            echo ""
            echo "=========================================="
            echo "🎉 PASSWORD FOUND!"
            echo "=========================================="
            password=$(python3 -c "from smart_mapper import smart_encode; print(smart_encode($index))")
            echo "Password: $password"
            echo "=========================================="
            echo ""
        fi
    fi
done
