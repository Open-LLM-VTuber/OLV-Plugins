"""
Fish Audio TTS Plugin Server

Uses the generic TTS server with Fish Audio TTS engine implementation.
"""

import os
import sys
from pathlib import Path

# Set environment variables for the generic TTS server
os.environ["TTS_ENGINE_MODULE"] = "fish_audio_tts_plugin.engine"
os.environ["TTS_ENGINE_CLASS"] = "FishAudioTTSEngine"

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from plugins.tts.tts_server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fish_audio_tts_plugin.server:app", host="0.0.0.0", port=8000, reload=False
    )
