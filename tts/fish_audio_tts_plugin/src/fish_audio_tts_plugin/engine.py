"""
Fish Audio TTS Engine Implementation

Implements the TTSEngineInterface for Fish Audio TTS service.
Supports both HTTP (non-streaming) and WebSocket (streaming) connections.
"""

import asyncio
import httpx
import ormsgpack
import websockets
from typing import Dict, Any, Optional, AsyncGenerator
from loguru import logger
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from plugins.tts.tts_engine_interface import TTSEngineInterface


class FishAudioTTSEngine(TTSEngineInterface):
    """Fish Audio TTS Engine implementation"""

    def __init__(self):
        """Initialize the Fish Audio TTS engine"""
        super().__init__()
        self.client: Optional[httpx.AsyncClient] = None
        self._is_ready = False

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the TTS engine with configuration

        Args:
            config: Complete configuration dictionary for the engine
        """
        try:
            # Store the configuration
            self._config = config.copy()

            # Validate required configuration
            if not config.get("api_key"):
                raise ValueError("api_key is required for Fish Audio TTS")

            # Create HTTP client
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/msgpack"
                }
            )

            # Test connection
            await self._test_connection()

            self._is_ready = True
            self._mark_initialized()

            logger.success(
                f"Fish Audio TTS engine initialized successfully with model {config.get('model', 'speech-1.6')}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Fish Audio TTS engine: {e}")
            self._is_ready = False
            self._mark_uninitialized()
            raise

    async def _test_connection(self) -> None:
        """Test connection to Fish Audio API"""
        try:
            # Simple test request to verify API key and connection
            test_request = {
                "text": "test",
                "chunk_length": 200,
                "format": "mp3",
                "mp3_bitrate": 128,
                "normalize": True,
                "latency": "balanced"
            }

            # Add reference_id if provided
            if self._config.get("reference_id"):
                test_request["reference_id"] = self._config["reference_id"]

            base_url = self._config.get("base_url", "https://api.fish.audio")
            url = f"{base_url}/v1/tts"

            headers = {
                "Authorization": f"Bearer {self._config['api_key']}",
                "Content-Type": "application/msgpack"
            }

            if self._config.get("model"):
                headers["model"] = self._config["model"]

            # Make a small test request
            async with httpx.AsyncClient(timeout=10.0) as test_client:
                response = await test_client.post(
                    url,
                    content=ormsgpack.packb(test_request),
                    headers=headers
                )

                if response.status_code not in [200, 201]:
                    raise Exception(f"API test failed with status {response.status_code}: {response.text}")

        except Exception as e:
            raise Exception(f"Failed to connect to Fish Audio API: {e}")

    async def synthesize(
        self,
        text: str,
        custom: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Synthesize speech from text (non-streaming)

        Args:
            text: Text to synthesize
            custom: Optional dictionary of custom parameters for this synthesis request
                   Supported parameters:
                   - "reference_id": Override reference ID for this request
                   - "latency": Override latency mode for this request
                   - "temperature": Override temperature for this request
                   - "top_p": Override top_p for this request

        Returns:
            Audio data as bytes
        """
        if not self.is_ready():
            raise RuntimeError("Fish Audio TTS engine is not initialized")

        # Validate text
        self.validate_text(text)

        try:
            # Build request
            request_data = {
                "text": text,
                "chunk_length": self._config.get("chunk_length", 200),
                "format": self._config.get("format", "mp3"),
                "mp3_bitrate": self._config.get("mp3_bitrate", 128),
                "normalize": self._config.get("normalize", True),
                "latency": self._config.get("latency", "balanced"),
                "temperature": self._config.get("temperature", 0.7),
                "top_p": self._config.get("top_p", 0.7)
            }

            # Add reference_id if provided
            if self._config.get("reference_id"):
                request_data["reference_id"] = self._config["reference_id"]

            # Add prosody if provided
            if self._config.get("prosody"):
                request_data["prosody"] = self._config["prosody"]

            # Override with custom parameters if provided
            if custom:
                for key in ["reference_id", "latency", "temperature", "top_p"]:
                    if key in custom:
                        request_data[key] = custom[key]

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self._config['api_key']}",
                "Content-Type": "application/msgpack"
            }

            if self._config.get("model"):
                headers["model"] = self._config["model"]

            # Make request
            base_url = self._config.get("base_url", "https://api.fish.audio")
            url = f"{base_url}/v1/tts"

            response = await self.client.post(
                url,
                content=ormsgpack.packb(request_data),
                headers=headers
            )

            if response.status_code not in [200, 201]:
                raise Exception(f"TTS request failed with status {response.status_code}: {response.text}")

            # Collect all audio data
            audio_data = b""
            async for chunk in response.aiter_bytes():
                audio_data += chunk

            return audio_data

        except Exception as e:
            logger.error(f"Fish Audio TTS synthesis failed: {e}")
            raise

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
        """
        if not self.is_ready():
            raise RuntimeError("Fish Audio TTS engine is not initialized")

        base_url = self._config.get("base_url", "https://api.fish.audio")
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/v1/tts/live"

        headers = {
            "Authorization": f"Bearer {self._config['api_key']}"
        }

        if self._config.get("model"):
            headers["model"] = self._config["model"]

        try:
            async with websockets.connect(ws_url, extra_headers=headers) as websocket:
                # Send initial configuration
                start_request = {
                    "event": "start",
                    "request": {
                        "text": "",
                        "latency": self._config.get("latency", "balanced"),
                        "format": self._config.get("format", "mp3"),
                        "temperature": self._config.get("temperature", 0.7),
                        "top_p": self._config.get("top_p", 0.7),
                        "normalize": self._config.get("normalize", True)
                    }
                }

                # Add reference_id if provided
                if self._config.get("reference_id"):
                    start_request["request"]["reference_id"] = self._config["reference_id"]

                # Add prosody if provided
                if self._config.get("prosody"):
                    start_request["request"]["prosody"] = self._config["prosody"]

                # Override with custom parameters if provided
                if custom:
                    for key in ["reference_id", "latency", "temperature", "top_p"]:
                        if key in custom:
                            start_request["request"][key] = custom[key]

                await websocket.send(ormsgpack.packb(start_request))

                # Create a queue for audio chunks
                audio_queue = asyncio.Queue()

                # Start listening for audio responses
                async def listen_for_audio():
                    try:
                        while True:
                            message = await websocket.recv()
                            data = ormsgpack.unpackb(message)

                            if data.get("event") == "audio":
                                await audio_queue.put(data["audio"])
                            elif data.get("event") == "finish":
                                await audio_queue.put(None)  # Signal end
                                break
                            elif data.get("event") == "log":
                                logger.debug(f"Fish Audio log: {data.get('message')}")

                    except websockets.exceptions.ConnectionClosed:
                        await audio_queue.put(None)  # Signal end

                # Start audio listener task
                listen_task = asyncio.create_task(listen_for_audio())

                try:
                    # Send text chunks
                    async for text_chunk in text_stream:
                        if text_chunk.strip():
                            text_event = {
                                "event": "text",
                                "text": text_chunk
                            }
                            await websocket.send(ormsgpack.packb(text_event))

                    # Send stop signal
                    stop_event = {"event": "stop"}
                    await websocket.send(ormsgpack.packb(stop_event))

                    # Yield audio chunks as they arrive
                    while True:
                        audio_chunk = await audio_queue.get()
                        if audio_chunk is None:  # End signal
                            break
                        yield audio_chunk

                finally:
                    # Ensure the listen task is cancelled
                    if not listen_task.done():
                        listen_task.cancel()
                        try:
                            await listen_task
                        except asyncio.CancelledError:
                            pass

        except Exception as e:
            logger.error(f"Fish Audio WebSocket TTS failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources asynchronously"""
        try:
            if self.client:
                await self.client.aclose()
                self.client = None

            self._is_ready = False
            self._mark_uninitialized()

            logger.info("Fish Audio TTS engine cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during Fish Audio TTS cleanup: {e}")

    def is_ready(self) -> bool:
        """Check if the engine is ready for synthesis"""
        return self._is_ready and self._is_initialized

    def supports_streaming(self) -> bool:
        """Check if the engine supports streaming synthesis"""
        return True
