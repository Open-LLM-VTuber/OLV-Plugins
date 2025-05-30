"""
TTS Plugins Package

This package contains TTS (Text-to-Speech) plugin implementations for the OLV platform.
"""

from .tts_engine_interface import TTSEngineInterface
from .tts_server import app

__all__ = ["TTSEngineInterface", "app"]
