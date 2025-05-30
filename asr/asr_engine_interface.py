"""
Modern ASR Engine Interface

A clean, async-first interface for ASR engines with simplified configuration management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class ASREngineInterface(ABC):
    """Modern async-first ASR engine interface"""

    # Audio format constants
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    DTYPE = np.float32

    def __init__(self):
        """Initialize the ASR engine"""
        self._config: Dict[str, Any] = {}
        self._is_initialized: bool = False

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the ASR engine with configuration

        Args:
            config: Complete configuration dictionary for the engine
        
        This method should set up the model, load weights, etc.
        Must be called before any transcription operations.
        Implementations should call self._store_config(config) to save the configuration.
        """
        pass

    @abstractmethod
    async def transcribe(
        self, 
        audio: np.ndarray, 
        custom: Optional[Dict[str, Any]] = None
    ) -> str:
        """Transcribe audio data asynchronously

        Args:
            audio: Audio data as float32 numpy array, normalized to [-1, 1]
                  Expected format: 16kHz, mono, float32
            custom: Optional dictionary of custom parameters for this transcription request

        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If engine is not initialized or ready
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources asynchronously
        
        This method should release any resources, close connections, etc.
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the engine is ready for transcription

        Returns:
            True if ready, False otherwise
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration
        
        Returns:
            Copy of current configuration
        """
        return self._config.copy()

    def validate_audio(self, audio: np.ndarray) -> None:
        """Validate audio format

        Args:
            audio: Audio data to validate
            
        Raises:
            ValueError: If audio format is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")
        
        if audio.dtype != self.DTYPE:
            raise ValueError(f"Audio must be {self.DTYPE}, got {audio.dtype}")
        
        if len(audio.shape) != 1:
            raise ValueError(f"Audio must be 1D array, got shape {audio.shape}")
        
        if np.any(np.abs(audio) > 1.0):
            raise ValueError("Audio values must be normalized to [-1, 1]")

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized"""
        return self._is_initialized

    def _mark_initialized(self) -> None:
        """Mark engine as initialized (for subclasses)"""
        self._is_initialized = True

    def _mark_uninitialized(self) -> None:
        """Mark engine as uninitialized (for subclasses)"""
        self._is_initialized = False

    def _store_config(self, config: Dict[str, Any]) -> None:
        """Store configuration (for subclasses to call during initialization)"""
        self._config = config.copy()


class ASREngineError(Exception):
    """Base exception for ASR engine errors"""
    pass


class ASREngineNotReadyError(ASREngineError):
    """Raised when engine is not ready for operation"""
    pass


class ASREngineConfigError(ASREngineError):
    """Raised when there's a configuration error"""
    pass


class ASREngineTranscriptionError(ASREngineError):
    """Raised when transcription fails"""
    pass
