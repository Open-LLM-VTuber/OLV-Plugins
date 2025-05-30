"""
Generic ASR Server

A universal FastAPI server that works with any ASR engine implementing ASREngineInterface.
Supports multiple engine instances for different configurations.
"""

import os
import base64
import time
import importlib
import hashlib
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from loguru import logger

from plugins.asr.asr_engine_interface import ASREngineInterface
from src.olv_launcher.models.asr import HealthResponse
from src.olv_launcher.models.api import PluginStatus


# Global engine instances: config_hash -> engine_instance
engine_instances: Dict[str, ASREngineInterface] = {}


def get_config_hash(config: Dict[str, Any]) -> str:
    """Generate a hash for configuration to use as key"""
    # Sort the config to ensure consistent hashing
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def find_best_matching_instance(target_config: Dict[str, Any]) -> str:
    """
    Find the best matching instance using minimal superset matching strategy.
    
    Strategy: Minimal Superset Matching (最小超集匹配)
    - Instance config must contain all fields from target_config with exact values
    - Among all matching instances, select the one with fewest total fields
    - If multiple instances have the same minimum field count, select the first one
    
    Args:
        target_config: The plugin_config to match against
        
    Returns:
        Instance ID (config hash) of the best matching instance
        
    Raises:
        ValueError: If no matching instance is found
    """
    if not engine_instances:
        raise ValueError("No engine instances available")
    
    matching_instances = []
    
    # Find all instances whose config is a superset of target_config
    for instance_id, engine in engine_instances.items():
        instance_config = engine.get_config() if hasattr(engine, 'get_config') else {}
        
        # Check if instance config contains all target_config fields with matching values
        is_match = True
        for key, value in target_config.items():
            if key not in instance_config or instance_config[key] != value:
                is_match = False
                break
        
        if is_match:
            matching_instances.append((instance_id, len(instance_config)))
    
    if not matching_instances:
        raise ValueError("No matching instance found for the provided plugin_config")
    
    # Sort by field count (ascending) and return the first one
    matching_instances.sort(key=lambda x: x[1])
    return matching_instances[0][0]


async def create_engine_instance(config: Dict[str, Any]) -> ASREngineInterface:
    """Create a new ASR engine instance with given configuration"""
    engine_module = os.getenv("ASR_ENGINE_MODULE")
    engine_class = os.getenv("ASR_ENGINE_CLASS")

    if not engine_module or not engine_class:
        raise RuntimeError(
            "ASR_ENGINE_MODULE and ASR_ENGINE_CLASS environment variables must be set"
        )

    try:
        # Import the engine module
        module = importlib.import_module(engine_module)
        engine_cls = getattr(module, engine_class)

        # Create engine instance
        engine = engine_cls()

        # Initialize the engine (async)
        await engine.initialize(config)

        logger.info(f"Created new ASR engine instance: {engine_module}.{engine_class}")
        return engine

    except Exception as e:
        logger.error(f"Failed to create ASR engine {engine_module}.{engine_class}: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.success("Generic ASR Server started successfully")
    yield
    # Shutdown - cleanup all engine instances
    for config_hash, engine in engine_instances.items():
        if hasattr(engine, "cleanup"):
            await engine.cleanup()
    engine_instances.clear()
    logger.info("ASR Server shutdown complete")


app = FastAPI(
    title="Generic ASR Server",
    description="Universal ASR service with instance-based configurations",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs"""
    return {"message": "Generic ASR Server v1.0", "docs": "/docs"}


@app.post("/transcribe")
async def transcribe(request: Dict[str, Any]):
    """Transcribe audio data using existing plugin configuration instance"""
    try:
        # Extract audio data, plugin config, and custom parameters
        audio_data = request.get("audio")
        plugin_config = request.get("plugin_config", {})
        custom_params = request.get("custom")

        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio field is required",
            )

        if not plugin_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="plugin_config field is required",
            )

        # Find the best matching instance using minimal superset matching
        try:
            instance_id = find_best_matching_instance(plugin_config)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )

        engine = engine_instances[instance_id]

        if not engine.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ASR engine instance not ready",
            )

        start_time = time.time()

        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)

        # Convert to numpy array (assuming 16-bit PCM, 16kHz)
        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Transcribe using async method, passing custom parameters if provided
        text = await engine.transcribe(audio_np, custom=custom_params)

        processing_time = time.time() - start_time

        return {
            "text": text,
            "processing_time": processing_time,
            "instance_id": instance_id,
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    total_instances = len(engine_instances)
    ready_instances = sum(1 for engine in engine_instances.values() if engine.is_ready())

    if total_instances == 0:
        # No instances created yet, but service is running
        return HealthResponse(status=PluginStatus.RUNNING)
    elif ready_instances == total_instances:
        # All instances are ready
        return HealthResponse(status=PluginStatus.RUNNING)
    elif ready_instances > 0:
        # Some instances are ready, some are not
        return HealthResponse(status=PluginStatus.STARTING)
    else:
        # No instances are ready
        return HealthResponse(status=PluginStatus.ERROR)


# Internal management endpoints (called by launcher)
@app.post("/create_instance")
async def create_instance_endpoint(request: Dict[str, Any]):
    """Create a new engine instance (internal endpoint for launcher)"""
    try:
        config = request.get("config", {})
        if not config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="config field is required",
            )

        instance_id = await create_instance(config)
        return {
            "message": "Instance created successfully",
            "instance_id": instance_id,
            "config": config,
        }
    except Exception as e:
        logger.error(f"Failed to create instance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create instance: {str(e)}",
        )


@app.get("/list_instances")
async def list_instances_endpoint():
    """List all engine instances (internal endpoint for launcher)"""
    try:
        instances = list_instances()
        return {
            "total_instances": len(instances),
            "instances": instances,
        }
    except Exception as e:
        logger.error(f"Failed to list instances: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list instances: {str(e)}",
        )


@app.delete("/delete_instance/{instance_id}")
async def delete_instance_endpoint(instance_id: str):
    """Delete an engine instance (internal endpoint for launcher)"""
    try:
        success = await delete_instance(instance_id)
        if success:
            return {
                "message": f"Instance '{instance_id}' deleted successfully",
                "instance_id": instance_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_id}' not found",
            )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to delete instance {instance_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete instance: {str(e)}",
        )


@app.get("/plugin-config", summary="Get plugin configuration", description="Get the plugin.json configuration for remote plugin discovery")
async def get_plugin_config():
    """Get plugin configuration for remote plugin discovery"""
    try:
        # Try to find plugin.json in the current working directory or parent directories
        current_dir = Path.cwd()
        plugin_config_path = None

        # Search for plugin.json in current directory and up to 3 parent directories
        for i in range(4):
            potential_path = current_dir / "plugin.json"
            if potential_path.exists():
                plugin_config_path = potential_path
                break
            current_dir = current_dir.parent

        if not plugin_config_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="plugin.json not found"
            )

        with open(plugin_config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return config_data

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="plugin.json not found"
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JSON in plugin.json: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read plugin configuration: {e}"
        )


# Internal management functions (called by launcher, not exposed as endpoints)
async def create_instance(config: Dict[str, Any]) -> str:
    """Create a new engine instance with given configuration (internal use)"""
    config_hash = get_config_hash(config)

    if config_hash in engine_instances:
        logger.info(f"Engine instance already exists: {config_hash}")
        return config_hash

    try:
        engine = await create_engine_instance(config)
        engine_instances[config_hash] = engine
        logger.info(f"Created engine instance with hash: {config_hash}")
        return config_hash
    except Exception as e:
        logger.error(f"Failed to create engine instance: {e}")
        raise


def list_instances() -> Dict[str, Dict[str, Any]]:
    """List all engine instances (internal use)"""
    instances = {}
    for config_hash, engine in engine_instances.items():
        instances[config_hash] = {
            "instance_id": config_hash,
            "ready": engine.is_ready(),
            "config": engine.get_config(),
        }
    return instances


async def delete_instance(instance_id: str) -> bool:
    """Delete an engine instance (internal use)"""
    if instance_id not in engine_instances:
        return False

    try:
        engine = engine_instances[instance_id]
        if hasattr(engine, "cleanup"):
            await engine.cleanup()

        del engine_instances[instance_id]
        logger.info(f"Deleted engine instance: {instance_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete engine instance {instance_id}: {e}")
        return False
