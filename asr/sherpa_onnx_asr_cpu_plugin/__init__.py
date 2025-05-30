"""
Sherpa-ONNX ASR CPU Plugin

Configuration examples:

{
  "model_type": "sense_voice",
  "sense_voice": "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
  "tokens": "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
  "provider": "cpu",
  "num_threads": 4,
  "sample_rate": 16000,
  "debug": false,
  "use_itn": true,
  "whisper_language": "zh",
  "whisper_task": "transcribe"
}

For transcription request:
{
  "audio": "base64_encoded_audio_data",
  "plugin_config": {
    "model_type": "sense_voice",
    "sense_voice": "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
    "tokens": "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
    "provider": "cpu",
    "num_threads": 4,
    "sample_rate": 16000,
    "debug": false,
    "use_itn": true
  }
}
"""
