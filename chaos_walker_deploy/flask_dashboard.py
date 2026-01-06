#!/usr/bin/env python3
"""Simple Flask Dashboard for ChaosWalker"""
from flask import Flask, render_template, jsonify, request
import subprocess
import threading
import re
from smart_mapper import smart_encode
import hashlib

app = Flask(__name__)

# Global state
engine_running = False
engine_process = None
result = {"found": False, "password": "", "logs": ""}

def get_gpu_stats():
    """Get GPU stats from nvidia-smi"""
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode().strip()
        temp, util, mem_used, mem_total = output.split(',')
        return {
            "temp": f"{temp.strip()}°C",
            "load": f"{util.strip()}%",
            "memory": f"{mem_used.strip()}/{mem_total.strip()} MB"
        }
    except:
        return {"temp": "N/A", "load": "N/A", "memory": "N/A"}

def run_engine(target_hash):
    global engine_running, engine_process, result
    engine_running = True
    result = {"found": False, "password": "", "logs": "Ready...\n"}
    
    try:
        # Update config
        import toml
        config = toml.load('config.toml')
        config['target_hash'] = target_hash
        with open('config.toml', 'w') as f:
            toml.dump(config, f)
        
        # Run engine
        engine_process = subprocess.Popen(
            ['./target/release/chaos_walker'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        proc = engine_process
        
        for line in proc.stdout:
            result["logs"] += line
            
            # Check for success
            if "SUCCESS" in line or "Target Found" in line:
                match = re.search(r'Random Index:\s*(\d+)', line)
                if match:
                    index = int(match.group(1))
                    password = smart_encode(index)
                    result["found"] = True
                    result["password"] = password
                    break
                    
    except Exception as e:
        result["logs"] += f"\nError: {e}\n"
    finally:
        engine_running = False

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>ChaosWalker Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            background: #2d2d2d;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        h1 {
            color: #10b981;
            text-align: center;
        }
        input, button {
            width: 100%;
            padding: 15px;
            margin: 10px 0;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #444;
        }
        input {
            background: #1a1a1a;
            color: #fff;
        }
        button {
            background: #10b981;
            color: white;
            font-weight: bold;
            cursor: pointer;
            border: none;
        }
        button:hover {
            background: #059669;
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .button-row {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 10px;
        }
        #stop_btn {
            background: #ef4444;
        }
        #stop_btn:hover {
            background: #dc2626;
        }
        #result {
            display: none;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        #result.show {
            display: block;
        }
        #result h2 {
            font-size: 48px;
            margin: 0;
        }
        #password {
            font-size: 32px;
            font-weight: bold;
            margin: 20px 0;
            padding: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
            font-family: 'Courier New', monospace;
        }
        #logs {
            background: #1a1a1a;
            padding: 15px;
            border-radius: 5px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        .status.running {
            background: #3b82f6;
        }
        .status.idle {
            background: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌪️ ChaosWalker Dashboard</h1>
        
        <div id="result">
            <h2>🎉 PASSWORD FOUND!</h2>
            <div id="password"></div>
        </div>
        
        <label>Target Password:</label>
        <input type="text" id="password_input" placeholder="Enter password (e.g., 'a')" />
        
        <label>Target Hash (auto-generated):</label>
        <input type="text" id="hash_input" readonly placeholder="Hash will appear here..." />
        
        <div class="button-row">
            <button id="start_btn" onclick="startEngine()">🚀 START ENGINE</button>
            <button id="stop_btn" onclick="stopEngine()" disabled>🛑 STOP</button>
        </div>
        
        <div id="status" class="status idle">IDLE</div>
        
        <h3>🖥️ GPU Telemetry:</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 15px 0;">
            <div style="background: #1a1a1a; padding: 15px; border-radius: 5px; text-align: center;">
                <div style="color: #10b981; font-size: 12px;">Temperature</div>
                <div id="gpu_temp" style="font-size: 24px; font-weight: bold;">--</div>
            </div>
            <div style="background: #1a1a1a; padding: 15px; border-radius: 5px; text-align: center;">
                <div style="color: #10b981; font-size: 12px;">GPU Load</div>
                <div id="gpu_load" style="font-size: 24px; font-weight: bold;">--</div>
            </div>
            <div style="background: #1a1a1a; padding: 15px; border-radius: 5px; text-align: center;">
                <div style="color: #10b981; font-size: 12px;">VRAM</div>
                <div id="gpu_memory" style="font-size: 24px; font-weight: bold;">--</div>
            </div>
        </div>
        
        <h3>Logs:</h3>
        <div id="logs">Ready...</div>
    </div>
    
    <script>
        let polling = null;
        
        // Auto-generate hash
        document.getElementById('password_input').addEventListener('input', async function() {
            const pwd = this.value;
            if (pwd) {
                const hash = await sha256(pwd);
                document.getElementById('hash_input').value = hash;
            }
        });
        
        async function sha256(message) {
            const msgBuffer = new TextEncoder().encode(message);
            const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
            const hashArray = Array.from(new Uint8Array(hashBuffer));
            return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        }
        
        function startEngine() {
            const hash = document.getElementById('hash_input').value;
            if (!hash) {
                alert('Please enter a password first!');
                return;
            }
            
            document.getElementById('start_btn').disabled = true;
            document.getElementById('stop_btn').disabled = false;
            document.getElementById('status').className = 'status running';
            document.getElementById('status').textContent = 'RUNNING...';
            document.getElementById('result').classList.remove('show');
            
            fetch('/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({hash: hash})
            });
            
            // Start polling
            polling = setInterval(checkStatus, 500);
        }
        
        function stopEngine() {
            fetch('/stop', {method: 'POST'});
            clearInterval(polling);
            document.getElementById('start_btn').disabled = false;
            document.getElementById('stop_btn').disabled = true;
            document.getElementById('status').className = 'status idle';
            document.getElementById('status').textContent = 'STOPPED';
        }
        
        function checkStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('logs').textContent = data.logs;
                    document.getElementById('logs').scrollTop = document.getElementById('logs').scrollHeight;
                    
                    // Update GPU stats
                    if (data.gpu) {
                        document.getElementById('gpu_temp').textContent = data.gpu.temp;
                        document.getElementById('gpu_load').textContent = data.gpu.load;
                        document.getElementById('gpu_memory').textContent = data.gpu.memory;
                    }
                    
                    if (data.found) {
                        clearInterval(polling);
                        document.getElementById('start_btn').disabled = false;
                        document.getElementById('stop_btn').disabled = true;
                        document.getElementById('status').className = 'status idle';
                        document.getElementById('status').textContent = '🎉 FOUND!';
                        document.getElementById('password').textContent = data.password;
                        document.getElementById('result').classList.add('show');
                        
                        // Alert
                        alert('🎉 PASSWORD FOUND!\\n\\nPassword: ' + data.password);
                    }
                    
                    if (!data.running && !data.found) {
                        clearInterval(polling);
                        document.getElementById('start_btn').disabled = false;
                        document.getElementById('stop_btn').disabled = true;
                        document.getElementById('status').className = 'status idle';
                        document.getElementById('status').textContent = 'STOPPED';
                    }
                });
        }
    </script>
</body>
</html>
'''

@app.route('/start', methods=['POST'])
def start():
    data = request.json
    target_hash = data.get('hash')
    
    thread = threading.Thread(target=run_engine, args=(target_hash,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop():
    global engine_running, engine_process
    if engine_process:
        engine_process.terminate()
        engine_process = None
    engine_running = False
    result["logs"] += "\n\n🛑 Engine stopped by user\n"
    return jsonify({"status": "stopped"})

@app.route('/status')
def status():
    return jsonify({
        "running": engine_running,
        "found": result["found"],
        "password": result["password"],
        "logs": result["logs"],
        "gpu": get_gpu_stats()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🌪️  ChaosWalker Flask Dashboard")
    print("=" * 70)
    print("\nStarting server...")
    print("Open: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
