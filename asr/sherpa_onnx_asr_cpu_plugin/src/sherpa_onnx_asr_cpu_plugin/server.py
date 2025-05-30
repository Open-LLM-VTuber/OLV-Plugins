"""
Sherpa-ONNX ASR Plugin Server

Uses the generic ASR server with Sherpa-ONNX engine implementation.
"""

import os
import sys
from pathlib import Path

os.environ["ASR_ENGINE_MODULE"] = "sherpa_onnx_asr_cpu_plugin.engine"
os.environ["ASR_ENGINE_CLASS"] = "SherpaONNXASREngine"

# Add project root to Python path for accessing the generic ASR server
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from plugins.asr.asr_server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sherpa_onnx_asr_cpu_plugin.server:app", host="0.0.0.0", port=8000, reload=False
    )
