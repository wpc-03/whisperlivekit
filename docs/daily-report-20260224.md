# WhisperLiveKit 项目日报

**日期**：2026-02-24  
**工作内容**：第三方集成方案设计与实现

---

## 一、工作背景

项目需要支持第三方系统（同事们）方便地集成实时语音转录功能，明确项目定位为**仅提供实时转录服务**，文件转录由其他服务负责。

---

## 二、完成工作

### 1. 新增文件

| 序号 | 文件路径 | 功能说明 |
|:---:|----------|----------|
| 1 | `WhisperLiveKit/web/whisper-livekit-sdk.js` | JavaScript SDK，支持一行代码快速集成 |
| 2 | `WhisperLiveKit/web/integration-example.html` | 集成示例页面，提供实时体验和代码参考 |
| 3 | `API_INTEGRATION_GUIDE.md` | 完整的 API 文档和使用指南 |

### 2. 修改文件

| 序号 | 文件路径 | 修改内容 |
|:---:|----------|----------|
| 1 | `WhisperLiveKit/basic_server.py` | 添加静态文件服务支持，挂载 `/web` 路径 |
| 2 | `WhisperLiveKit/web/integration-example.html` | 修复时间显示类型错误 bug |

---

## 三、技术方案

### 提供的集成方式

**方式一：JavaScript SDK（推荐）**
```javascript
const client = new WhisperLiveKit({
    serverUrl: 'ws://localhost:8000',
    language: 'zh',
    diarization: true
});
client.onResult = (result) => console.log(result.text);
await client.start();
```

**方式二：WebSocket 原生 API**
```javascript
const ws = new WebSocket('ws://localhost:8000/asr');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.buffer_transcription);
};
```

### 服务端接口

| 端点 | 类型 | 说明 |
|------|------|------|
| `/` | HTTP GET | 内置网页转录界面 |
| `/web/integration-example.html` | HTTP GET | 集成示例页面 |
| `/web/whisper-livekit-sdk.js` | HTTP GET | JavaScript SDK |
| `/asr` | WebSocket | 实时流式转录 |

---

## 四、使用说明

### 启动服务
```bash
# 推荐方式
wlk --model tiny --language zh --pcm-input

# 或使用 Python 模块方式
python -m whisperlivekit.basic_server --model tiny --language zh --pcm-input
```

> **注意**：必须添加 `--pcm-input` 参数，SDK 发送的是 PCM 格式音频数据

### 访问地址
- 示例页面：`http://localhost:8000/web/integration-example.html`
- 内置界面：`http://localhost:8000/`

---

## 五、问题与解决

| 问题描述 | 原因分析 | 解决方案 |
|----------|----------|----------|
| 浏览器控制台报错 `toFixed is not a function` | `line.start` 为字符串而非数字 | 添加类型检查后再调用 `toFixed` |
| FFmpeg 连接断开错误 | 服务端未启用 PCM 模式，尝试用 FFmpeg 处理 PCM 数据 | 启动时添加 `--pcm-input` 参数 |

---

## 六、项目定位

**WhisperLiveKit 专注于：**
- ✅ 实时语音转录
- ✅ 低延迟流式传输
- ✅ 说话人分离
- ✅ 网页端集成

**不提供（请使用其他服务）：**
- ❌ 文件上传转录
- ❌ 批量音频处理
