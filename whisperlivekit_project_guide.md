# WhisperLiveKit 项目详细介绍

## 项目概述

WhisperLiveKit（简称WLK）是一个**超低延迟、自托管的语音转文本系统**，具有说话人识别功能。它基于最新的语音识别研究成果，提供实时、准确的音频转录和翻译服务。

### 核心特点

- 超低延迟转录（基于Simul-Whisper/Streaming技术）
- 支持200多种语言的实时转录和翻译
- 内置说话人识别功能
- 自托管，完全控制数据隐私
- 支持多种部署方式（本地、Docker、云服务器）

---

## 项目架构

### 整体架构

WhisperLiveKit采用分层架构设计，主要包括：

1. **前端层**：基于Web的用户界面，通过WebSocket与后端通信
2. **服务器层**：FastAPI实现的Web服务器，处理WebSocket连接和HTTP请求
3. **音频处理层**：处理音频流、语音活动检测和数据分发
4. **核心引擎层**：管理ASR模型、分词器、说话人识别等核心功能
5. **后端策略层**：实现不同的流式处理策略（SimulStreaming、LocalAgreement）
6. **模型层**：集成各种Whisper模型变体和辅助模型

### 数据流图

```
┌───────────────┐     WebSocket     ┌────────────────┐
│   前端浏览器   │──────────────────>│   basic_server │
└───────────────┘                    └────────────────┘
                                           │
                                           ▼
┌───────────────┐     WebSocket     ┌────────────────┐     音频流     ┌────────────────┐
│   前端浏览器   │<──────────────────│   basic_server │<──────────────│ AudioProcessor │
└───────────────┘                    └────────────────┘               └────────────────┘
                                                                             │
                                                                             ▼
                                                              ┌────────────────────────┐
                                                              │  后端处理策略          │
                                                              │┌──────────┐ ┌─────────┐│
                                                              ││SimulStream│ │LocalAgree││
                                                              ││  ing     │ │  ment   ││
                                                              │└──────────┘ └─────────┘│
                                                              └────────────────────────┘
                                                                             │
                                                                             ▼
                                                              ┌────────────────────────┐
                                                              │  模型层                │
                                                              │┌──────────┐ ┌─────────┐│
                                                              ││ Whisper  │ │  NLLW   ││
                                                              ││ 模型     │ │ 翻译模型││
                                                              │└──────────┘ └─────────┘│
                                                              └────────────────────────┘
```

---

## 核心功能模块

### 1. TranscriptionEngine (core.py)

**核心引擎**，负责初始化和管理所有转录相关的组件。

- **单例模式**：确保整个应用中只有一个引擎实例
- **参数管理**：处理和验证各种配置参数
- **模型初始化**：根据配置初始化ASR模型、分词器、说话人识别模型等
- **后端策略选择**：根据配置选择不同的流式处理策略
- **翻译模型集成**：如果需要翻译功能，初始化NLLW翻译模型

### 2. AudioProcessor (audio_processor.py)

**音频处理器**，负责处理音频流、管理状态和结果格式化。

- **音频处理**：接收、缓冲和处理音频数据
- **语音活动检测**：使用VAD检测语音和静音
- **任务管理**：创建和管理各种处理任务（转录、说话人识别、翻译）
- **状态管理**：维护处理状态和结果
- **结果格式化**：将处理结果格式化为前端可显示的格式

### 3. 后端策略

#### SimulStreaming (simul_whisper)

基于**AlignAtt策略**的超低延迟转录实现，是默认的后端策略。

- **实时性**：通过智能的令牌预测和验证机制，实现接近实时的转录
- **准确性**：在保持低延迟的同时，确保转录准确性
- **灵活性**：支持多种Whisper后端实现

#### LocalAgreement (local_agreement)

基于**LocalAgreement策略**的低延迟转录实现。

- **稳定性**：通过对局部结果的一致性验证，提高转录稳定性
- **适应性**：适用于不同的音频质量和说话风格

### 4. 说话人识别 (diarization)

支持两种说话人识别后端：

- **Sortformer**：推荐的后端，基于最新的Streaming Sortformer技术
- **Diart**：传统的说话人识别后端，基于pyannote模型

### 5. 翻译功能 (NLLW)

集成**NLLW**（NoLanguageLeftWaiting）翻译模型，支持200多种语言的实时翻译。

- **多语言支持**：支持从任何语言翻译到任何语言
- **低延迟**：与转录过程并行执行，实现实时翻译
- **高质量**：基于NLLB模型，提供高质量的翻译结果

---

## 主要代码文件

### 1. core.py

**核心引擎实现**，定义了TranscriptionEngine类，负责初始化和管理所有转录相关的组件。

**主要功能**：初始化模型、管理配置、选择后端策略

**关键函数**：

| 函数名 | 说明 |
|--------|------|
| `TranscriptionEngine.__init__()` | 初始化核心引擎 |
| `online_factory()` | 创建在线处理器实例 |
| `online_diarization_factory()` | 创建在线说话人识别实例 |
| `online_translation_factory()` | 创建在线翻译实例 |

### 2. audio_processor.py

**音频处理器实现**，定义了AudioProcessor类，负责处理音频流、管理状态和结果格式化。

**主要功能**：处理音频数据、管理处理任务、格式化结果

**关键函数**：

| 函数名 | 说明 |
|--------|------|
| `AudioProcessor.process_audio()` | 处理输入音频数据 |
| `AudioProcessor.create_tasks()` | 创建处理任务 |
| `AudioProcessor.transcription_processor()` | 处理转录任务 |
| `AudioProcessor.diarization_processor()` | 处理说话人识别任务 |
| `AudioProcessor.translation_processor()` | 处理翻译任务 |
| `AudioProcessor.results_formatter()` | 格式化处理结果 |

### 3. basic_server.py

**服务器实现**，基于FastAPI，处理WebSocket连接和HTTP请求。

**主要功能**：提供Web界面、处理WebSocket连接、管理客户端会话

**关键函数**：

| 函数名 | 说明 |
|--------|------|
| `websocket_endpoint()` | 处理WebSocket连接 |
| `handle_websocket_results()` | 处理和发送处理结果 |
| `main()` | 启动服务器 |

### 4. ffmpeg_manager.py

**FFmpeg管理器**，负责处理音频编解码。

**主要功能**：启动和管理FFmpeg进程、处理音频数据

**关键函数**：

| 函数名 | 说明 |
|--------|------|
| `FFmpegManager.start()` | 启动FFmpeg进程 |
| `FFmpegManager.write_data()` | 向FFmpeg写入音频数据 |
| `FFmpegManager.read_data()` | 从FFmpeg读取处理后的音频数据 |

### 5. silero_vad_iterator.py

**语音活动检测器**，使用Silero VAD模型检测语音和静音。

**主要功能**：检测音频中的语音活动、分割语音和静音段

**关键函数**：

| 函数名 | 说明 |
|--------|------|
| `FixedVADIterator.__call__()` | 检测语音活动 |
| `load_jit_vad()` | 加载JIT格式的VAD模型 |
| `load_onnx_session()` | 加载ONNX格式的VAD模型 |

---

## 关键技术实现细节

### 1. 超低延迟转录

WhisperLiveKit使用两种策略实现低延迟转录：

#### SimulStreaming

基于AlignAtt策略，通过以下技术实现低延迟：

- **增量处理**：边接收音频边处理
- **智能预测**：预测可能的后续令牌
- **动态验证**：验证预测结果的准确性
- **自适应延迟**：根据音频内容调整处理延迟

#### LocalAgreement

基于LocalAgreement策略，通过以下技术实现低延迟：

- **局部一致性验证**：验证局部结果的一致性
- **缓冲管理**：智能管理音频缓冲
- **早期输出**：尽早输出可信的转录结果

### 2. 多后端支持

WhisperLiveKit支持多种Whisper后端实现：

| 后端 | 说明 |
|------|------|
| **whisper** | 原始的OpenAI Whisper实现 |
| **faster-whisper** | 优化的Whisper实现，提供更快的推理速度 |
| **mlx-whisper** | 针对Apple Silicon优化的Whisper实现 |
| **openai-api** | 使用OpenAI API进行转录（仅支持LocalAgreement策略） |

### 3. 说话人识别

WhisperLiveKit集成了两种说话人识别后端：

- **Sortformer**：基于最新的Streaming Sortformer技术，提供实时说话人识别
- **Diart**：基于pyannote模型的传统说话人识别系统

### 4. 翻译功能

WhisperLiveKit使用NLLW（NoLanguageLeftWaiting）翻译模型，支持200多种语言的实时翻译：

- 基于NLLB（No Language Left Behind）模型
- 支持从任何语言翻译到任何语言
- 与转录过程并行执行，实现实时翻译

### 5. 音频处理

WhisperLiveKit使用FFmpeg和Silero VAD进行音频处理：

- **FFmpeg**：处理各种音频格式，转换为模型可接受的格式
- **Silero VAD**：检测语音活动，减少非语音段的处理开销

---

## 程序运行流程

### 1. 服务器启动流程

```
1. 解析命令行参数 → parse_args()函数
2. 初始化TranscriptionEngine → 创建核心引擎实例，初始化各种模型
3. 启动FastAPI服务器 → 启动Web服务器，监听指定端口
4. 提供Web界面 → 通过根路径提供Web界面
```

### 2. 客户端连接流程

```
1. 客户端连接 → 浏览器连接到服务器的WebSocket端点
2. 创建AudioProcessor → 为每个客户端创建一个AudioProcessor实例
3. 配置传输 → 根据配置确定是否使用AudioWorklet
4. 创建处理任务 → 创建转录、说话人识别、翻译等处理任务
5. 开始接收音频 → 开始接收客户端发送的音频数据
```

### 3. 音频处理流程

```
1. 接收音频数据 → 通过WebSocket接收音频数据
2. 音频预处理 → 如果需要，使用FFmpeg处理音频数据
3. 语音活动检测 → 使用VAD检测语音活动
4. 分发音频数据 → 将语音数据分发给各个处理模块
5. 执行转录 → 使用选定的后端策略执行转录
6. 执行说话人识别 → 如果启用，执行说话人识别
7. 执行翻译 → 如果启用，执行翻译
8. 格式化结果 → 将处理结果格式化为前端可显示的格式
9. 发送结果 → 通过WebSocket将结果发送回客户端
```

### 4. 会话结束流程

```
1. 接收结束信号 → 接收客户端发送的结束信号
2. 清理资源 → 清理各种处理任务和资源
3. 发送完成信号 → 向客户端发送处理完成信号
4. 关闭连接 → 关闭WebSocket连接
```

---

## 各组件之间的交互关系

### 1. TranscriptionEngine与其他组件

| 组件 | 交互内容 |
|------|----------|
| **AudioProcessor** | 提供ASR模型、分词器、说话人识别模型等 |
| **后端策略** | 根据配置选择和初始化后端策略 |
| **辅助模型** | 初始化和管理翻译模型、VAD模型等 |

### 2. AudioProcessor与其他组件

| 组件 | 交互内容 |
|------|----------|
| **basic_server** | 接收音频数据，发送处理结果 |
| **TranscriptionEngine** | 获取模型和配置 |
| **后端策略** | 使用后端策略执行转录 |
| **辅助模型** | 使用VAD模型检测语音活动 |

### 3. basic_server与其他组件

| 组件 | 交互内容 |
|------|----------|
| **前端浏览器** | 提供Web界面，处理WebSocket连接 |
| **AudioProcessor** | 为每个客户端创建AudioProcessor实例 |
| **TranscriptionEngine** | 获取核心引擎实例 |

### 4. 后端策略与其他组件

| 组件 | 交互内容 |
|------|----------|
| **TranscriptionEngine** | 获取模型和配置 |
| **AudioProcessor** | 接收音频数据，返回处理结果 |

### 5. 辅助模型与其他组件

| 组件 | 交互内容 |
|------|----------|
| **AudioProcessor** | 使用VAD模型检测语音活动 |
| **TranscriptionEngine** | 初始化和管理辅助模型 |

---

## 技术栈与依赖

### 核心技术栈

| 技术 | 说明 |
|------|------|
| **Python** | 主要开发语言 |
| **FastAPI** | Web服务器框架 |
| **WebSocket** | 实时通信协议 |
| **FFmpeg** | 音频处理工具 |
| **NumPy** | 数值计算库 |
| **PyTorch** | 深度学习框架 |
| **Whisper** | 语音识别模型 |

### 主要依赖

| 依赖 | 说明 |
|------|------|
| **whisper** | OpenAI的语音识别模型 |
| **faster-whisper** | 优化的Whisper实现（可选） |
| **mlx-whisper** | 针对Apple Silicon优化的Whisper实现（可选） |
| **nllw** | 多语言翻译模型（可选） |
| **diart** | 说话人识别系统（可选） |
| **onnxruntime** | ONNX模型推理引擎（可选） |
| **pyannote.audio** | 说话人识别库（可选） |

---

## 部署与配置

### 本地部署

```bash
# 1. 安装
pip install whisperlivekit

# 2. 启动服务器
wlk --model base --language en

# 3. 访问
# 打开浏览器，导航到 http://localhost:8000
```

### Docker部署

```bash
# 1. 构建镜像
# GPU版本
docker build -t wlk .

# CPU版本
docker build -f Dockerfile.cpu -t wlk .

# 2. 运行容器
# GPU版本
docker run --gpus all -p 8000:8000 --name wlk wlk

# CPU版本
docker run -p 8000:8000 --name wlk wlk
```

### 生产部署

```bash
# 1. 安装生产依赖
pip install uvicorn gunicorn

# 2. 启动多 worker
gunicorn -k uvicorn.workers.UvicornWorker -w 4 your_app:app

# 3. 配置Nginx
# 设置反向代理，处理WebSocket连接

# 4. 启用HTTPS
# 配置SSL证书，使用wss://协议
```

### 主要配置参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model` | Whisper模型大小 | `small` |
| `--language` | 语言代码 | `auto` |
| `--target-language` | 目标翻译语言 | `None` |
| `--diarization` | 启用说话人识别 | `False` |
| `--backend-policy` | 流式处理策略 | `simulstreaming` |
| `--backend` | Whisper实现选择 | `auto` |
| `--host` | 服务器主机地址 | `localhost` |
| `--port` | 服务器端口 | `8000` |

---

## 典型用例

### 1. 会议实时转录

- **功能**：实时转录会议内容，区分不同说话人
- **配置**：
  ```bash
  wlk --model medium --language en --diarization
  ```
- **优势**：超低延迟，实时显示转录结果，自动区分说话人

### 2. 多语言翻译

- **功能**：实时转录并翻译多种语言
- **配置**：
  ```bash
  wlk --model large-v3 --language fr --target-language en
  ```
- **优势**：支持200多种语言，实时翻译，高质量结果

### 3. 视频字幕生成

- **功能**：为视频生成实时字幕
- **配置**：
  ```bash
  wlk --model medium --language auto
  ```
- **优势**：实时生成字幕，支持多种语言，准确捕捉对话内容

### 4. 辅助听力设备

- **功能**：帮助听力障碍人士实时了解对话内容
- **配置**：
  ```bash
  wlk --model small --language en
  ```
- **优势**：超低延迟，实时显示对话内容，易于集成到辅助设备

---

## 总结

WhisperLiveKit是一个功能强大、性能优异的实时语音转文本系统，具有以下核心优势：

1. **超低延迟**：基于最新的Simul-Whisper/Streaming技术，实现接近实时的转录
2. **多语言支持**：支持200多种语言的转录和翻译
3. **说话人识别**：自动区分不同说话人，提高转录可读性
4. **自托管**：完全控制数据隐私，无需依赖第三方服务
5. **灵活配置**：支持多种模型大小、后端策略和部署方式
6. **易于集成**：提供Python API和Web界面，易于集成到现有系统

WhisperLiveKit适用于多种场景，包括会议转录、多语言翻译、视频字幕生成、辅助听力设备等，为用户提供高质量、低延迟的语音转文本服务。

---

