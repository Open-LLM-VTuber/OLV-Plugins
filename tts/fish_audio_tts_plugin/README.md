# Fish Audio TTS Plugin

这是一个 Fish Audio TTS（文本转语音）插件，支持 HTTP 和 WebSocket 连接。

## 功能特性

- **HTTP 连接**: 支持非流式 TTS 合成
- **WebSocket 连接**: 支持流式 TTS 合成
- **多种音频格式**: 支持 MP3、WAV、PCM、Opus 格式
- **语音克隆**: 支持使用 reference_id 进行语音克隆
- **韵律控制**: 支持语速和音量调节
- **多种延迟模式**: 支持 normal 和 balanced 延迟模式

## 配置参数

### 必需参数
- `api_key`: Fish Audio API 密钥

### 可选参数
- `base_url`: API 基础 URL（默认: https://api.fish.audio）
- `reference_id`: 语音参考 ID
- `model`: TTS 模型（speech-1.5 或 speech-1.6，默认: speech-1.6）
- `format`: 音频格式（wav、pcm、mp3、opus，默认: mp3）
- `mp3_bitrate`: MP3 比特率（64、128、192，默认: 128）
- `chunk_length`: 分块长度（100-300，默认: 200）
- `normalize`: 是否标准化文本（默认: true）
- `latency`: 延迟模式（normal 或 balanced，默认: balanced）
- `temperature`: 随机性控制（0.0-2.0，默认: 0.7）
- `top_p`: 多样性控制（0.0-1.0，默认: 0.7）
- `prosody`: 韵律设置
  - `speed`: 语速（0.5-2.0，默认: 1.0）
  - `volume`: 音量调节（dB，默认: 0）

## API 端点

### HTTP 端点

#### POST `/synthesize`
非流式 TTS 合成

**请求体:**
```json
{
    "text": "要合成的文本",
    "plugin_config": {
        "api_key": "your_api_key",
        "reference_id": "voice_reference_id",
        "latency": "balanced"
    },
    "custom": {
        "reference_id": "override_reference_id",
        "latency": "normal"
    }
}
```

**响应:**
```json
{
    "audio": "base64_encoded_audio_data",
    "processing_time": 0.123,
    "instance_id": "abc123def456"
}
```

### WebSocket 端点

#### WebSocket `/synthesize_stream`
流式 TTS 合成

**连接后发送配置:**
```json
{
    "plugin_config": {
        "api_key": "your_api_key",
        "reference_id": "voice_reference_id"
    },
    "custom": {
        "latency": "balanced"
    }
}
```

**发送文本块:**
```json
{
    "type": "text",
    "text": "文本块"
}
```

**结束信号:**
```json
{
    "type": "end"
}
```

**接收音频:**
```json
{
    "type": "audio",
    "audio": "base64_encoded_audio_chunk"
}
```

## 使用示例

### HTTP 示例
```python
import httpx
import base64

async def synthesize_text():
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/synthesize", json={
            "text": "Hello, world!",
            "plugin_config": {
                "api_key": "your_api_key",
                "reference_id": "your_reference_id",
                "format": "mp3",
                "latency": "balanced"
            }
        })
        
        result = response.json()
        audio_data = base64.b64decode(result["audio"])
        
        with open("output.mp3", "wb") as f:
            f.write(audio_data)
```

### WebSocket 示例
```python
import asyncio
import websockets
import json
import base64

async def stream_synthesize():
    uri = "ws://localhost:8000/synthesize_stream"
    
    async with websockets.connect(uri) as websocket:
        # 发送配置
        config = {
            "plugin_config": {
                "api_key": "your_api_key",
                "reference_id": "your_reference_id",
                "format": "mp3"
            }
        }
        await websocket.send(json.dumps(config))
        
        # 等待就绪信号
        ready_msg = await websocket.recv()
        print(f"Ready: {ready_msg}")
        
        # 发送文本
        text_chunks = ["Hello, ", "world! ", "How are you?"]
        for chunk in text_chunks:
            await websocket.send(json.dumps({
                "type": "text",
                "text": chunk
            }))
        
        # 发送结束信号
        await websocket.send(json.dumps({"type": "end"}))
        
        # 接收音频块
        audio_data = b""
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "audio":
                audio_chunk = base64.b64decode(data["audio"])
                audio_data += audio_chunk
            elif data.get("type") == "complete":
                break
        
        # 保存音频
        with open("stream_output.mp3", "wb") as f:
            f.write(audio_data)
```

## 注意事项

1. 确保你有有效的 Fish Audio API 密钥
2. reference_id 可以从 Fish Audio 网站获取
3. 流式模式适合实时应用，非流式模式适合批处理
4. balanced 延迟模式可以减少延迟但可能降低稳定性
