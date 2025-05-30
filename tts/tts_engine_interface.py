"""
Modern TTS Engine Interface

A clean, async-first interface for TTS engines with support for both HTTP and WebSocket connections.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator


class TTSEngineInterface(ABC):
    """Modern async-first TTS engine interface"""

    def __init__(self):
        """Initialize the TTS engine"""
        self._config: Dict[str, Any] = {}
        self._is_initialized: bool = False

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the TTS engine with configuration

        Args:
            config: Complete configuration dictionary for the engine

        This method should set up the model, load weights, etc.
        Must be called before any synthesis operations.
        Implementations should call self._store_config(config) to save the configuration.
        """
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        custom: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Synthesize speech from text (non-streaming)

        Args:
            text: Text to synthesize
            custom: Optional dictionary of custom parameters for this synthesis request
                   Examples: {"reference_id": "abc123", "latency": "balanced"}

        Returns:
            Audio data as bytes

        Raises:
            RuntimeError: If engine is not initialized or ready
        """
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None],
        custom: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize speech from streaming text (WebSocket streaming)

        Args:
            text_stream: Async generator yielding text chunks
            custom: Optional dictionary of custom parameters for this synthesis request

        Yields:
            Audio data chunks as bytes

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
        """Check if the engine is ready for synthesis

        Returns:
            True if ready, False otherwise
        """
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if the engine supports streaming synthesis

        Returns:
            True if streaming is supported, False otherwise
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration

        Returns:
            Copy of current configuration
        """
        return self._config.copy()

    def _mark_initialized(self) -> None:
        """Mark engine as initialized"""
        self._is_initialized = True

    def _mark_uninitialized(self) -> None:
        """Mark engine as uninitialized"""
        self._is_initialized = False

    def _store_config(self, config: Dict[str, Any]) -> None:
        """Store configuration (for subclasses to call during initialization)"""
        self._config = config.copy()

    def validate_text(self, text: str) -> None:
        """Validate text input

        Args:
            text: Text to validate

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if not text.strip():
            raise ValueError("Text cannot be empty")
