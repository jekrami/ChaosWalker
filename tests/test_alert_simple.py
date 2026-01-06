#!/usr/bin/env python3
"""
Simple test dashboard with just an alert button
"""
import gradio as gr

def show_alert():
    alert_html = """
<script>
    alert('🎉 TEST ALERT!\\n\\nThis should appear immediately!');
</script>
"""
    return alert_html, "Alert triggered!"

with gr.Blocks() as demo:
    gr.Markdown("# Simple Alert Test")
    
    alert_trigger = gr.HTML(value="", visible=True)
    status = gr.Textbox(label="Status", value="Ready")
    
    btn = gr.Button("🚨 TEST ALERT")
    btn.click(show_alert, outputs=[alert_trigger, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
