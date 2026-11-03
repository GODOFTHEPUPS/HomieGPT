# HomieGPT

A Discord voice bot that uses Ollama for AI chat, Sherpa-ONNX for speech
recognition, and Edge TTS for speech synthesis.

## Requirements

- Python and FFmpeg
- [Ollama](https://ollama.com/) with the `qwen3:8b` model available
- A Discord bot token with the permissions needed to join voice channels
- The Sherpa-ONNX streaming Zipformer model files listed below

## Setup

1. Install the Python dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Download the English streaming Zipformer model from the
   [Sherpa-ONNX model repository](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26)
   and extract it into this project folder. The resulting directory must be
   named `sherpa-onnx-streaming-zipformer-en-2023-06-26` and contain:
   - `tokens.txt`
   - `encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx`
   - `decoder-epoch-99-avg-1-chunk-16-left-128.onnx`
   - `joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx`

   The model is not included in this repository. Its model card identifies it
   as Apache-2.0 licensed; follow the license and attribution terms when using
   or redistributing it.

3. Copy `.env.example` to `.env` and set `DISCORD_TOKEN` to your bot token.
   Do not commit `.env`.

4. Start Ollama with `qwen3:8b` available, then run:

   ```powershell
   python bot.py
   ```
