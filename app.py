import os
import threading
import gradio as gr

# Write credentials from environment secret to credentials.json if passed as text
if "GOOGLE_SHEET_CREDS" in os.environ and os.environ["GOOGLE_SHEET_CREDS"].startswith("{"):
    with open("credentials.json", "w") as f:
        f.write(os.environ["GOOGLE_SHEET_CREDS"])
    os.environ["GOOGLE_SHEET_CREDS"] = "credentials.json"

def start_trading_bot():
    os.system('python main.py --source ccxt --symbol "BTC/USDT" --timeframe 5m --poll 300')

# Run trading bot in background thread
threading.Thread(target=start_trading_bot, daemon=True).start()

# Minimal UI for Hugging Face
with gr.Blocks() as demo:
    gr.Markdown("# ?? Quant ML Trading Bot")
    gr.Markdown("Status: Running in background 24/7.")

demo.launch()
