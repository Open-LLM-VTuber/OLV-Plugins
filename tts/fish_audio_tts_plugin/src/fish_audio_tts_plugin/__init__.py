"""
Fish Audio TTS Plugin

A TTS plugin for Fish Audio service supporting both HTTP and WebSocket connections.
"""

__version__ = "1.0.0"
__author__ = "OLV Team"

from .engine import FishAudioTTSEngine
from .server import app

__all__ = ["FishAudioTTSEngine", "app"]
