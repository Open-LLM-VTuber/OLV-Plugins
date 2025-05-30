"""
Generic TTS Server

A universal FastAPI server that works with any TTS engine implementing TTSEngineInterface.
Supports both HTTP (non-streaming) and WebSocket (streaming) connections.
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

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from loguru import logger

from plugins.tts.tts_engine_interface import TTSEngineInterface
from src.olv_launcher.models.asr import HealthResponse
from src.olv_launcher.models.api import PluginStatus


# Global engine instances: config_hash -> engine_instance
engine_instances: Dict[str, TTSEngineInterface] = {}


def get_config_hash(config: Dict[str, Any]) -> str:
    """Generate a hash for the configuration to use as instance ID"""
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


async def create_engine_instance(config: Dict[str, Any]) -> TTSEngineInterface:
    """Create a new TTS engine instance with given configuration"""
    engine_module = os.getenv("TTS_ENGINE_MODULE")
    engine_class = os.getenv("TTS_ENGINE_CLASS")

    if not engine_module or not engine_class:
        raise RuntimeError(
            "TTS_ENGINE_MODULE and TTS_ENGINE_CLASS environment variables must be set"
        )

    try:
        # Import the engine module
        module = importlib.import_module(engine_module)
        engine_cls = getattr(module, engine_class)

        # Create engine instance
        engine = engine_cls()

        # Initialize the engine (async)
        await engine.initialize(config)

        logger.info(f"Created new TTS engine instance: {engine_module}.{engine_class}")
        return engine

    except Exception as e:
        logger.error(f"Failed to create TTS engine instance: {e}")
        raise


async def create_instance(config: Dict[str, Any]) -> str:
    """Create a new engine instance and return its ID"""
    config_hash = get_config_hash(config)

    if config_hash in engine_instances:
        logger.info(f"Engine instance already exists for config hash: {config_hash}")
        return config_hash

    engine = await create_engine_instance(config)
    engine_instances[config_hash] = engine

    logger.info(f"Created engine instance with ID: {config_hash}")
    return config_hash


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting TTS server...")
    yield
    logger.info("Shutting down TTS server...")

    # Cleanup all engine instances
    for instance_id, engine in engine_instances.items():
        try:
            await engine.cleanup()
            logger.info(f"Cleaned up engine instance: {instance_id}")
        except Exception as e:
            logger.error(f"Error cleaning up engine instance {instance_id}: {e}")

    engine_instances.clear()


app = FastAPI(
    title="Generic TTS Server",
    description="Universal TTS server supporting HTTP and WebSocket connections",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs"""
    return {"message": "Generic TTS Server v1.0", "docs": "/docs"}


@app.post("/synthesize")
async def synthesize(request: Dict[str, Any]):
    """Synthesize speech from text using existing plugin configuration instance"""
    try:
        # Extract text, plugin config, and custom parameters
        text = request.get("text")
        plugin_config = request.get("plugin_config", {})
        custom_params = request.get("custom")

        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text field is required",
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
                detail="TTS engine instance not ready",
            )

        start_time = time.time()

        # Synthesize using async method, passing custom parameters if provided
        audio_bytes = await engine.synthesize(text, custom=custom_params)

        processing_time = time.time() - start_time

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {
            "audio": audio_b64,
            "processing_time": processing_time,
            "instance_id": instance_id,
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}",
        )


@app.websocket("/synthesize_stream")
async def synthesize_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS synthesis"""
    await websocket.accept()

    try:
        # Receive initial configuration
        config_data = await websocket.receive_json()
        plugin_config = config_data.get("plugin_config", {})
        custom_params = config_data.get("custom")

        if not plugin_config:
            await websocket.send_json({"error": "plugin_config is required"})
            await websocket.close()
            return

        # Find the best matching instance using minimal superset matching
        try:
            instance_id = find_best_matching_instance(plugin_config)
        except ValueError as e:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
            return

        engine = engine_instances[instance_id]

        if not engine.is_ready():
            await websocket.send_json({"error": "TTS engine instance not ready"})
            await websocket.close()
            return

        if not engine.supports_streaming():
            await websocket.send_json({"error": "TTS engine does not support streaming"})
            await websocket.close()
            return

        # Send ready signal
        await websocket.send_json({"status": "ready", "instance_id": instance_id})

        # Create text stream generator
        async def text_stream():
            while True:
                try:
                    data = await websocket.receive_json()
                    if data.get("type") == "text":
                        yield data.get("text", "")
                    elif data.get("type") == "end":
                        break
                except WebSocketDisconnect:
                    break

        # Stream synthesis
        async for audio_chunk in engine.synthesize_stream(text_stream(), custom=custom_params):
            if audio_chunk:
                # Encode audio chunk as base64
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({
                    "type": "audio",
                    "audio": audio_b64
                })

        # Send completion signal
        await websocket.send_json({"type": "complete"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket synthesis failed: {e}")
        try:
            await websocket.send_json({"error": f"Synthesis failed: {str(e)}"})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check TTS service health status"""
    if not engine_instances:
        return HealthResponse(status=PluginStatus.RUNNING)

    ready_count = sum(1 for engine in engine_instances.values() if engine.is_ready())
    total_count = len(engine_instances)

    if ready_count == 0:
        status_value = PluginStatus.ERROR
    elif ready_count == total_count:
        status_value = PluginStatus.RUNNING
    else:
        status_value = PluginStatus.STARTING

    return HealthResponse(status=status_value)


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


@app.delete("/instances/{instance_id}")
async def delete_instance(instance_id: str):
    """Delete an engine instance (internal endpoint for launcher)"""
    try:
        if instance_id not in engine_instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance {instance_id} not found",
            )

        engine = engine_instances[instance_id]
        await engine.cleanup()
        del engine_instances[instance_id]

        logger.info(f"Deleted engine instance: {instance_id}")
        return {"message": f"Instance {instance_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete instance {instance_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete instance: {str(e)}",
        )


@app.get("/list_instances")
async def list_instances():
    """List all engine instances (internal endpoint for launcher)"""
    instances = {}
    for instance_id, engine in engine_instances.items():
        instances[instance_id] = {
            "ready": engine.is_ready(),
            "supports_streaming": engine.supports_streaming(),
            "config": engine.get_config(),
        }
    return {
        "total_instances": len(instances),
        "instances": instances
    }


@app.get("/plugin-config", summary="Get plugin configuration", description="Get the plugin.json configuration for remote plugin discovery")
async def get_plugin_config():
    """Get plugin configuration for remote plugin discovery"""
    try:
        # Try to find plugin.json in the current working directory or parent directories
        current_dir = Path.cwd()
        plugin_config_path = None

        # Search for plugin.json in current directory and up to 3 parent directories
        for _ in range(4):
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

        with open(plugin_config_path, 'r', encoding='utf-8') as f:
            plugin_config = json.load(f)

        return plugin_config

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read plugin config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read plugin config: {str(e)}"
        )
