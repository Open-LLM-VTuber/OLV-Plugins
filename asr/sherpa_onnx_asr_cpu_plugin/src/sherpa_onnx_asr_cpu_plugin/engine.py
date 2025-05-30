"""
Sherpa-ONNX ASR Engine Implementation

Implements the ASREngineInterface for Sherpa-ONNX speech recognition.
"""

import os
import tarfile
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio

import numpy as np
import sherpa_onnx
import requests
from tqdm import tqdm
from loguru import logger

import sys

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from plugins.asr.asr_engine_interface import ASREngineInterface


def download_and_extract(url: str, output_dir: str) -> Path:
    """Download a file from a URL and extract it if it is a tar.bz2 archive."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_name = url.split("/")[-1]
    file_path = os.path.join(output_dir, file_name)
    root_dir = file_name.replace(".tar.bz2", "")
    extracted_dir_path = Path(output_dir) / root_dir

    if extracted_dir_path.exists():
        logger.info(
            f"Directory {extracted_dir_path} already exists. Skipping download."
        )
        return extracted_dir_path

    logger.info(f"Downloading {url} to {file_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(file_path, "wb") as f,
        tqdm(desc=file_name, total=total_size, unit="iB", unit_scale=True) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

    logger.info(f"Downloaded {file_name} successfully.")

    if file_name.endswith(".tar.bz2"):
        logger.info(f"Extracting {file_name}...")
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall(path=output_dir)
        logger.info("Extraction completed.")
        os.remove(file_path)
        return extracted_dir_path
    else:
        return Path(file_path)


def check_and_extract_local_file(url: str, output_dir: str) -> Optional[Path]:
    """Check if a local file exists and extract it if it is a tar.bz2 archive."""
    file_name = url.split("/")[-1]
    compressed_path = Path(output_dir) / file_name
    extracted_dir = Path(output_dir) / file_name.replace(".tar.bz2", "")

    if extracted_dir.exists():
        logger.info(f"Extracted directory exists: {extracted_dir}")
        return extracted_dir

    if compressed_path.exists() and file_name.endswith(".tar.bz2"):
        logger.info(f"Found local archive file: {compressed_path}")
        try:
            logger.info("Extracting archive file...")
            with tarfile.open(compressed_path, "r:bz2") as tar:
                tar.extractall(path=output_dir)
            logger.info(f"Extracted archive to: {extracted_dir}")
            os.remove(compressed_path)
            return extracted_dir
        except Exception as e:
            logger.error(f"Failed to extract file: {str(e)}")
            return None

    return None


class SherpaONNXASREngine(ASREngineInterface):
    """Sherpa-ONNX ASR Engine implementation"""

    def __init__(self):
        """Initialize the Sherpa-ONNX ASR engine"""
        super().__init__()
        self.recognizer = None
        self._is_ready = False

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the ASR engine with configuration"""
        try:
            # Store the configuration
            self._config = config.copy()
            
            # Create recognizer
            self.recognizer = await asyncio.to_thread(self._create_recognizer, config)
            self._is_ready = True
            self._mark_initialized()
            
            logger.success(
                f"ASR engine initialized successfully with {config.get('model_type', 'unknown')} model"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ASR engine: {e}")
            self._is_ready = False
            self._mark_uninitialized()
            raise

    async def transcribe(
        self, 
        audio: np.ndarray, 
        custom: Optional[Dict[str, Any]] = None
    ) -> str:
        """Transcribe audio data asynchronously
        
        Args:
            audio: Audio data as float32 numpy array, normalized to [-1, 1]
            custom: Optional dictionary of custom parameters for this transcription request
                   Supported parameters:
                   - "language": Language code for transcription (if supported by model)
                   - "use_itn": Override ITN setting for this request
                   - "sample_rate": Override sample rate for this request
        """
        if not self.is_ready():
            raise RuntimeError("ASR engine is not initialized")

        # Validate audio format
        self.validate_audio(audio)

        # Use async transcription with custom parameters
        return await asyncio.to_thread(self._transcribe_sync, audio, custom)

    async def cleanup(self) -> None:
        """Clean up resources asynchronously"""
        if self.recognizer:
            self.recognizer = None
        self._is_ready = False
        self._mark_uninitialized()
        logger.info("ASR engine cleaned up")

    def is_ready(self) -> bool:
        """Check if the ASR engine is ready to process audio"""
        return self._is_ready and self.recognizer is not None

    def _transcribe_sync(self, audio: np.ndarray, custom: Optional[Dict[str, Any]] = None) -> str:
        """Synchronous transcription method for thread execution
        
        Args:
            audio: Audio data as float32 numpy array
            custom: Optional custom parameters for this transcription
        """
        stream = self.recognizer.create_stream()
        
        # Get sample rate from custom parameters or use default from config
        sample_rate = self._config.get("sample_rate", 16000)
        
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_streams([stream])
        
        result_text = stream.result.text
        return result_text

    def _create_recognizer(self, config: Dict[str, Any]):
        """Create the appropriate recognizer based on model type"""
        model_type = config.get("model_type", "sense_voice")
        
        if model_type == "sense_voice":
            return self._create_sense_voice_recognizer(config)
        elif model_type == "whisper":
            return self._create_whisper_recognizer(config)
        elif model_type == "transducer":
            return self._create_transducer_recognizer(config)
        elif model_type == "paraformer":
            return self._create_paraformer_recognizer(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_sense_voice_recognizer(self, config: Dict[str, Any]):
        """Create SenseVoice recognizer"""
        sense_voice_model = config.get("sense_voice")
        tokens = config.get("tokens")
        
        # Handle SenseVoice model download if needed
        if not sense_voice_model or not os.path.isfile(sense_voice_model):
            if sense_voice_model and sense_voice_model.startswith(
                "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
            ):
                logger.warning(
                    "SenseVoice model not found. Downloading the model..."
                )
                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
                output_dir = "./models"

                # Check local file first before download
                local_result = check_and_extract_local_file(url, output_dir)
                if local_result is None:
                    logger.info("Local file not found. Downloading...")
                    download_and_extract(url, output_dir)
                else:
                    logger.info("Local file found. Using existing file.")
            else:
                raise ValueError(
                    "The SenseVoice model is missing. Please provide the path to the model.onnx file."
                )

        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=sense_voice_model,
            tokens=tokens,
            num_threads=config.get("num_threads", 1),
            use_itn=config.get("use_itn", True),
            debug=config.get("debug", False),
            provider=config.get("provider", "cpu"),
        )

    def _create_whisper_recognizer(self, config: Dict[str, Any]):
        """Create Whisper recognizer"""
        encoder = config.get("whisper_encoder")
        decoder = config.get("whisper_decoder")
        tokens = config.get("tokens")
        
        if not encoder or not decoder or not tokens:
            raise ValueError("Whisper model requires encoder, decoder, and tokens files")

        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=config.get("num_threads", 1),
            debug=config.get("debug", False),
            provider=config.get("provider", "cpu"),
            language=config.get("whisper_language", "en"),
            task=config.get("whisper_task", "transcribe"),
        )

    def _create_transducer_recognizer(self, config: Dict[str, Any]):
        """Create Transducer recognizer"""
        encoder = config.get("encoder")
        decoder = config.get("decoder")
        joiner = config.get("joiner")
        tokens = config.get("tokens")
        
        if not encoder or not decoder or not joiner or not tokens:
            raise ValueError("Transducer model requires encoder, decoder, joiner, and tokens files")

        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=config.get("num_threads", 1),
            debug=config.get("debug", False),
            provider=config.get("provider", "cpu"),
        )

    def _create_paraformer_recognizer(self, config: Dict[str, Any]):
        """Create Paraformer recognizer"""
        model = config.get("paraformer")
        tokens = config.get("tokens")
        
        if not model or not tokens:
            raise ValueError("Paraformer model requires model and tokens files")

        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            model=model,
            tokens=tokens,
            num_threads=config.get("num_threads", 1),
            debug=config.get("debug", False),
            provider=config.get("provider", "cpu"),
        )
