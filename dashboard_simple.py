#!/usr/bin/env python3
import gradio as gr
import subprocess
import re
from smart_mapper import smart_encode

def run_engine():
    """Run engine and capture output"""
    result = subprocess.run(
        ["./target/release/chaos_walker"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    output = result.stdout + result.stderr
    
    # Check for success
    if "SUCCESS" in output:
        match = re.search(r'Random Index:\s*(\d+)', output)
        if match:
            index = int(match.group(1))
            password = smart_encode(index)
            return (output, 
                    f"🎉 FOUND: {password}", 
                    f"Password is: {password}")
    
    return (output, "Not found yet", "")

with gr.Blocks() as demo:
    gr.Markdown("# Simple ChaosWalker Dashboard")
    
    btn = gr.Button("🚀 START")
    status = gr.Textbox(label="Status", value="Ready")
    result = gr.Textbox(label="PASSWORD", value="")
    logs = gr.Textbox(label="Logs", lines=10)
    
    btn.click(run_engine, outputs=[logs, status, result])

demo.launch(server_name="0.0.0.0", server_port=7862)
