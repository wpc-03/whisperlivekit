# WhisperLiveKit 第三方集成指南

WhisperLiveKit 是一个**实时语音转录服务**，专注于 WebSocket 流式传输，适合会议、直播、语音助手等实时场景。

> **注意**：本项目仅提供实时转录功能。如需文件转录，请使用其他专门的文件转录服务。

---

## 快速开始

### 1. 启动服务

```bash
# 安装依赖
pip install whisperlivekit

# 启动服务
wlk --model tiny --language zh

# 或使用 Python 模块方式
python -m whisperlivekit.basic_server --model tiny --language zh
```

服务启动后，默认在 `http://localhost:8000` 运行。

### 2. 访问示例页面

打开浏览器访问：
```
http://localhost:8000/web/integration-example.html
```

---

## 集成方式

### 方式一：JavaScript SDK（推荐）

最简单的集成方式，适合网页应用。

```html
<!DOCTYPE html>
<html>
<head>
    <title>语音转录</title>
</head>
<body>
    <button id="startBtn">开始录音</button>
    <button id="stopBtn">停止录音</button>
    <div id="result"></div>

    <!-- 引入 SDK -->
    <script src="http://your-server:8000/web/whisper-livekit-sdk.js"></script>
    <script>
        // 创建客户端
        const client = new WhisperLiveKit({
            serverUrl: 'ws://your-server:8000',
            language: 'zh',        // 指定语言，或留空自动检测
            diarization: true      // 启用说话人分离
        });

        // 监听转录结果
        client.onResult = (result) => {
            console.log('转录结果:', result);
            document.getElementById('result').innerHTML += `<p>${result.text}</p>`;
        };

        // 监听临时结果（实时显示）
        client.onPartial = (result) => {
            console.log('临时结果:', result.text);
        };

        // 监听错误
        client.onError = (error) => {
            console.error('错误:', error);
        };

        // 开始录音
        document.getElementById('startBtn').onclick = () => {
            client.start().then(() => {
                console.log('开始录音');
            });
        };

        // 停止录音
        document.getElementById('stopBtn').onclick = () => {
            client.stop();
        };
    </script>
</body>
</html>
```

### SDK API 参考

#### 构造函数

```javascript
const client = new WhisperLiveKit(options);
```

**options:**
- `serverUrl` (string): 服务器地址，如 `ws://localhost:8000`
- `language` (string, 可选): 语言代码，如 `zh`, `en`，留空则自动检测
- `diarization` (boolean, 可选): 是否启用说话人分离，默认 `true`
- `microphoneId` (string, 可选): 指定麦克风设备 ID

#### 方法

- `start()`: 开始录音和转录
- `stop()`: 停止录音
- `static getMicrophones()`: 获取可用麦克风列表（静态方法）

#### 事件回调

- `onConnect`: 连接成功时触发
- `onReady`: 服务器准备就绪时触发
- `onResult(result)`: 收到转录结果时触发
- `onPartial(result)`: 收到临时结果时触发
- `onError(error)`: 发生错误时触发
- `onDisconnect`: 断开连接时触发

#### result 对象结构

```javascript
{
    text: "转录文本",
    lines: [
        {speaker: 1, text: "...", start: 0, end: 5}
    ],
    isFinal: false,
    speaker: "Speaker 1",
    language: "zh"
}
```

---

### 方式二：WebSocket API

适合需要更灵活控制的场景。

#### 连接地址

```
ws://your-server:8000/asr
```

#### 协议说明

1. **连接建立**: 客户端连接 WebSocket
2. **接收配置**: 服务器发送配置信息
3. **发送音频**: 客户端发送 PCM 音频数据 (16kHz, 16bit, 单声道)
4. **接收结果**: 服务器实时返回转录结果

#### 消息格式

**服务器发送:**
```json
// 配置消息
{
    "type": "config",
    "useAudioWorklet": true,
    "message": "Ready to receive audio stream"
}

// 转录结果
{
    "lines": [
        {"speaker": 1, "text": "已完成的第一段", "start": 0, "end": 5}
    ],
    "buffer_transcription": "正在识别的临时文本",
    "buffer_diarization": "Speaker 1",
    "detected_language": "zh",
    "status": "active_transcription"
}

// 结束消息
{
    "type": "ready_to_stop"
}
```

#### JavaScript 示例

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/stream');

ws.onopen = () => {
    console.log('已连接');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'config') {
        console.log('服务器配置:', data);
        // 开始发送音频...
    } else if (data.buffer_transcription) {
        console.log('实时转录:', data.buffer_transcription);
    } else if (data.type === 'ready_to_stop') {
        console.log('转录完成');
        ws.close();
    }
};

// 发送音频数据 (需要自行实现音频采集)
function sendAudioData(pcmData) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(pcmData);
    }
}
```
```

---

## 完整示例项目

### React 组件示例

```jsx
import React, { useState, useEffect, useRef } from 'react';

function TranscriptionComponent() {
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState('');
    const clientRef = useRef(null);

    useEffect(() => {
        // 加载 SDK
        const script = document.createElement('script');
        script.src = 'http://your-server:8000/web/whisper-livekit-sdk.js';
        script.onload = () => {
            clientRef.current = new window.WhisperLiveKit({
                serverUrl: 'ws://your-server:8000',
                language: 'zh'
            });

            clientRef.current.onResult = (result) => {
                setTranscript(prev => prev + result.text + '\n');
            };
        };
        document.body.appendChild(script);

        return () => {
            if (clientRef.current) {
                clientRef.current.stop();
            }
        };
    }, []);

    const toggleRecording = async () => {
        if (isRecording) {
            clientRef.current?.stop();
            setIsRecording(false);
        } else {
            await clientRef.current?.start();
            setIsRecording(true);
        }
    };

    return (
        <div>
            <button onClick={toggleRecording}>
                {isRecording ? '停止录音' : '开始录音'}
            </button>
            <pre>{transcript}</pre>
        </div>
    );
}

export default TranscriptionComponent;
```

---

## 接口汇总

| 端点 | 类型 | 说明 |
|------|------|------|
| `/` | HTTP GET | 内置网页转录界面 |
| `/web/integration-example.html` | HTTP GET | 集成示例页面 |
| `/web/whisper-livekit-sdk.js` | HTTP GET | JavaScript SDK |
| `/asr` | WebSocket | 实时流式转录 |

---

## 启动参数

```bash
wlk \
    --model tiny \              # 模型名称或路径
    --model_dir /path/to/model \ # 本地模型目录（优先级高于 --model）
    --language zh \             # 语言
    --host 0.0.0.0 \            # 监听地址
    --port 8000 \               # 端口
    --diarization \             # 启用说话人分离
    --pcm-input \               # 使用 PCM 输入
    --ssl-certfile cert.pem \   # SSL 证书
    --ssl-keyfile key.pem       # SSL 密钥
```

---

## 常见问题

### Q: 如何指定特定麦克风？

```javascript
// 先获取麦克风列表
const mics = await WhisperLiveKit.getMicrophones();

// 然后指定使用
const client = new WhisperLiveKit({
    microphoneId: mics[0].deviceId
});
```

### Q: 支持哪些音频格式？

仅支持 **PCM 16kHz 16bit 单声道** 实时流式传输。

如需文件转录，请使用其他专门的文件转录服务。

### Q: 如何启用 HTTPS/WSS？

```bash
python -m whisperlivekit.api_server \
    --ssl-certfile /path/to/cert.pem \
    --ssl-keyfile /path/to/key.pem
```

### Q: 如何跨域调用？

API 服务器默认已启用 CORS，支持所有跨域请求。

---

## 项目定位

**WhisperLiveKit 专注于：**
- ✅ 实时语音转录
- ✅ 低延迟流式传输
- ✅ 说话人分离
- ✅ 网页端集成

**不提供（请使用其他服务）：**
- ❌ 文件上传转录
- ❌ 批量音频处理
- ❌ 离线转录任务

---

## 技术支持

如有问题，请查看项目文档或提交 Issue。
