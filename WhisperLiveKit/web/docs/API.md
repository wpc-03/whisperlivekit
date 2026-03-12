# WhisperLiveKit API 文档

## 概述

WhisperLiveKit 是一个用于实时语音转录的 JavaScript SDK，支持两种模式：
- **按钮模式 (WhisperButton)**: 手动控制开始/停止录音
- **唤醒词模式 (WhisperHotword)**: 通过唤醒词触发转录

## 安装

```html
<script src="src/Logger.js"></script>
<script src="src/WhisperButton.js"></script>
<script src="src/WhisperHotword.js"></script>
```

## WhisperButton 类

按钮模式语音转录客户端

### 构造函数

```javascript
new WhisperButton(options)
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| serverUrl | string | `ws://localhost:8000` | WebSocket服务器地址 |
| language | string | `null` | 语言代码，如 'zh', 'en'，null 为自动检测 |
| diarization | boolean | `true` | 是否启用说话人分离 |
| microphoneId | string | `null` | 麦克风设备ID，null 为默认麦克风 |
| autoStart | boolean | `true` | 连接后是否自动开始录音 |
| logLevel | string | `'info'` | 日志级别: debug, info, warn, error |
| logHandler | function | `null` | 自定义日志处理器 |

### 方法

#### start()

开始语音转录。

```javascript
await client.start()
```

**返回**: `Promise<void>`

**抛出**: `WhisperError` - 如果已经在录音中或处于错误状态

#### stop()

停止语音转录。

```javascript
client.stop()
```

**返回**: `void`

#### destroy()

销毁客户端，释放所有资源。

```javascript
client.destroy()
```

**返回**: `void`

#### static getMicrophones()

获取可用的麦克风列表。

```javascript
const microphones = await WhisperButton.getMicrophones()
```

**返回**: `Promise<Array<{deviceId: string, label: string}>>`

#### getAnalyser()

获取音频分析器，用于波形可视化。

```javascript
const analyser = client.getAnalyser()
```

**返回**: `AnalyserNode | null`

### 事件回调

#### onresult

最终转录结果回调。

```javascript
client.onresult = (result) => {
  console.log('转录结果:', result.text);
  console.log('逐句:', result.lines);
}
```

**参数**:
- `result.text`: 完整转录文本
- `result.lines`: 转录行数组，每行包含 text, start, end, speaker 等
- `result.isPartial`: 是否为最终结果

#### onpartial

中间结果回调（实时识别中）。

```javascript
client.onpartial = (result) => {
  console.log('中间结果:', result.text);
}
```

#### onstatuschange

状态变化回调。

```javascript
client.onstatuschange = (status) => {
  console.log('状态:', status.current);
  console.log('消息:', status.message);
}
```

**参数**:
- `status.previous`: 之前的状态
- `status.current`: 当前状态
- `status.message`: 状态消息

#### onerror

错误回调。

```javascript
client.onerror = (error) => {
  console.error('错误:', error.message);
  console.error('错误代码:', error.code);
}
```

#### onconnect

连接成功回调。

```javascript
client.onconnect = () => {
  console.log('已连接服务器');
}
```

#### ondisconnect

断开连接回调。

```javascript
client.ondisconnect = () => {
  console.log('已断开连接');
}
```

#### onready

服务器准备就绪回调。

```javascript
client.onready = (config) => {
  console.log('服务器配置:', config);
}
```

#### oncompletetion

转录完成回调。

```javascript
client.oncompletetion = (data) => {
  console.log('转录完成:', data);
}
```

---

## WhisperHotword 类

唤醒词模式语音转录客户端

### 构造函数

```javascript
new WhisperHotword(options)
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| serverUrl | string | `ws://localhost:8000` | WebSocket服务器地址 |
| microphoneId | string | `null` | 麦克风设备ID |
| hotwordThreshold | number | `0.5` | 唤醒词检测阈值 (0.1-0.9) |
| autoStartTranscription | boolean | `true` | 检测到唤醒词后是否自动开始转录 |
| logLevel | string | `'info'` | 日志级别 |
| logHandler | function | `null` | 自定义日志处理器 |

### 方法

#### start()

开始监听唤醒词。

```javascript
await hotword.start()
```

**返回**: `Promise<void>`

#### stop()

停止监听唤醒词。

```javascript
hotword.stop()
```

**返回**: `void`

#### destroy()

销毁客户端。

```javascript
hotword.destroy()
```

**返回**: `void`

#### static getMicrophones()

获取可用的麦克风列表。

```javascript
const microphones = await WhisperHotword.getMicrophones()
```

**返回**: `Promise<Array<{deviceId: string, label: string}>>`

### 事件回调

#### onhotword

检测到唤醒词回调。

```javascript
hotword.onhotword = (keyword, data) => {
  console.log('检测到唤醒词:', keyword);
}
```

**参数**:
- `keyword`: 唤醒词内容
- `data`: 原始数据

#### onresult

转录结果回调（与 WhisperButton 相同）

#### onpartial

中间结果回调（与 WhisperButton 相同）

#### onstatuschange

状态变化回调（与 WhisperButton 相同）

#### onerror

错误回调（与 WhisperButton 相同）

---

## 错误处理

### WhisperError 类

```javascript
try {
  await client.start();
} catch (error) {
  console.error(error.message);  // 错误消息
  console.error(error.code);    // 错误代码
  console.error(error.details); // 详细信息
}
```

### 错误代码

| 代码 | 说明 |
|------|------|
| E001 | 服务器不可用 |
| E002 | 麦克风访问被拒绝 |
| E003 | WebSocket错误 |
| E004 | 音频处理错误 |
| E005 | 配置无效 |
| E006 | 已在录音中 |
| E007 | 未连接 |
| E999 | 未知错误 |

---

## 日志系统

### 配置日志级别

```javascript
const client = new WhisperButton({
  logLevel: 'debug'  // debug, info, warn, error
});
```

### 自定义日志处理器

```javascript
const client = new WhisperButton({
  logHandler: (level, message, data) => {
    // 发送到远程日志服务
    fetch('/api/logs', {
      method: 'POST',
      body: JSON.stringify({ level, message, data, timestamp: Date.now() })
    });
  }
});
```

---

## 使用示例

### 按钮模式

```html
<script src="src/Logger.js"></script>
<script src="src/WhisperButton.js"></script>
<script>
  const client = new WhisperButton({
    serverUrl: 'ws://localhost:8000',
    language: 'zh'
  });
  
  client.onresult = (result) => {
    console.log('转录:', result.text);
  };
  
  client.onerror = (error) => {
    console.error('错误:', error.message);
  };
  
  // 开始转录
  await client.start();
  
  // 停止转录
  client.stop();
</script>
```

### 唤醒词模式

```html
<script src="src/Logger.js"></script>
<script src="src/WhisperButton.js"></script>
<script src="src/WhisperHotword.js"></script>
<script>
  const hotword = new WhisperHotword({
    serverUrl: 'ws://localhost:8000',
    hotwordThreshold: 0.5
  });
  
  hotword.onhotword = (keyword) => {
    console.log('唤醒词:', keyword);
  };
  
  hotword.onresult = (result) => {
    console.log('转录:', result.text);
  };
  
  // 开始监听
  await hotword.start();
  
  // 停止监听
  hotword.stop();
</script>
```

---

## 浏览器兼容性

- Chrome 66+
- Firefox 66+
- Safari 14.1+
- Edge 79+

需要支持以下API：
- WebSocket
- MediaDevices.getUserMedia
- AudioContext
- AudioWorklet (可选，如不支持则使用 MediaRecorder)

---

## 协议版本

当前版本: 1.0.0
