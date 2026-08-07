import os
import threading
import gradio as gr
import spaces

@spaces.GPU
def zero_gpu_initializer():
    return "ZeroGPU Active"

# Properly parse JSON secret into credentials.json file
creds_content = os.environ.get("GOOGLE_SHEET_CREDS", "")
if creds_content.startswith("{"):
    with open("credentials.json", "w") as f:
        f.write(creds_content)

def start_trading_bot():
    # Pass environment variables and file path explicitly
    os.system('python main.py --source ccxt --exchange kraken --symbol "BTC/USDT" --timeframe 5m --poll 300')

threading.Thread(target=start_trading_bot, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("# ?? Quant ML Trading Bot")
    gr.Markdown("Status: Running in background 24/7.")

demo.launch()
