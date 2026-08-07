import os
import subprocess
import threading
import gradio as gr
import spaces

@spaces.GPU
def zero_gpu_initializer():
    return "ZeroGPU Active"

# Prepare credentials file from secret
creds_content = os.environ.get("GOOGLE_SHEET_CREDS", "")
if creds_content.startswith("{"):
    with open("credentials.json", "w") as f:
        f.write(creds_content)
    os.environ["GOOGLE_SHEET_CREDS"] = "credentials.json"

def start_trading_bot():
    # Pass current environment variables directly to the child process
    env = os.environ.copy()
    cmd = [
        "python", "main.py",
        "--source", "ccxt",
        "--exchange", "kraken",
        "--symbol", "BTC/USDT",
        "--timeframe", "5m",
        "--poll", "300"
    ]
    subprocess.run(cmd, env=env)

threading.Thread(target=start_trading_bot, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("# ?? Quant ML Trading Bot")
    gr.Markdown("Status: Running in background 24/7.")

demo.launch()
