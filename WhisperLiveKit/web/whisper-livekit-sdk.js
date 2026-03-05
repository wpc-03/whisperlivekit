/**
 * WhisperLiveKit JavaScript SDK
 * 供第三方系统快速集成实时语音转录功能
 *
 * 使用方法:
 * 1. 引入此 SDK: <script src="whisper-livekit-sdk.js"></script>
 * 2. 创建客户端: const client = new WhisperLiveKit({ serverUrl: 'ws://localhost:8000' });
 * 3. 开始录音: await client.start();
 * 4. 监听结果: client.onResult = (result) => console.log(result.text);
 */

class WhisperLiveKit {
  /**
   * 创建 WhisperLiveKit 客户端
   * @param {Object} options - 配置选项
   * @param {string} options.serverUrl - 服务器地址 (如: ws://localhost:8000 或 wss://your-server.com)
   * @param {string} options.language - 语言代码 (如: 'zh', 'en')，留空则自动检测
   * @param {boolean} options.diarization - 是否启用说话人分离，默认 true
   * @param {string} options.microphoneId - 指定麦克风设备 ID
   */
  constructor(options = {}) {
    this.serverUrl = options.serverUrl || this._detectServerUrl();
    this.language = options.language || null;
    this.diarization = options.diarization !== false;
    this.microphoneId = options.microphoneId || null;

    // 内部状态
    this.websocket = null;
    this.audioContext = null;
    this.analyser = null;
    this.microphone = null;
    this.workletNode = null;
    this.isRecording = false;
    this.isConnected = false;

    // 回调函数
    this.onResult = null;        // 收到转录结果时调用
    this.onPartial = null;       // 收到临时结果时调用
    this.onError = null;         // 发生错误时调用
    this.onConnect = null;       // 连接成功时调用
    this.onDisconnect = null;    // 断开连接时调用
    this.onReady = null;         // 服务器准备就绪时调用
    this.onReadyToStop = null;   // 服务器通知所有音频处理完成时调用

    // 配置信息
    this.serverUseAudioWorklet = null;
  }

  /**
   * 自动检测服务器地址
   */
  _detectServerUrl() {
    if (typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const host = window.location.hostname || 'localhost';
      const port = window.location.port ? `:${window.location.port}` : '';
      return `${protocol}://${host}${port}`;
    }
    return 'ws://localhost:8000';
  }

  /**
   * 开始录音和转录
   * @returns {Promise<void>}
   */
  async start() {
    if (this.isRecording) {
      throw new Error('Already recording');
    }

    try {
      // 1. 连接 WebSocket
      await this._connectWebSocket();

      // 2. 等待配置信息
      await this._waitForConfig();

      // 3. 初始化音频
      await this._initAudio();

      this.isRecording = true;
    } catch (error) {
      this._cleanup();
      if (this.onError) this.onError(error);
      throw error;
    }
  }

  /**
   * 停止录音
   */
  stop() {
    if (!this.isRecording) return;

    this.isRecording = false;
    this._cleanup();

    // 发送停止信号
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.close();
    }
  }

  /**
   * 获取可用的麦克风列表
   * @returns {Promise<Array<{deviceId: string, label: string}>>}
   */
  static async getMicrophones() {
    try {
      // 先请求权限
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());

      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter(device => device.kind === 'audioinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Microphone ${device.deviceId.slice(0, 8)}...`
        }));
    } catch (error) {
      throw new Error(`Failed to get microphones: ${error.message}`);
    }
  }

  /**
   * 连接 WebSocket
   */
  _connectWebSocket() {
    return new Promise((resolve, reject) => {
      const wsUrl = `${this.serverUrl}/asr`;

      try {
        this.websocket = new WebSocket(wsUrl);
      } catch (error) {
        reject(new Error(`Failed to create WebSocket: ${error.message}`));
        return;
      }

      this.websocket.onopen = () => {
        this.isConnected = true;
        if (this.onConnect) this.onConnect();
        resolve();
      };

      this.websocket.onclose = () => {
        this.isConnected = false;
        if (this.onDisconnect) this.onDisconnect();
        this._cleanup();
      };

      this.websocket.onerror = (error) => {
        reject(new Error('WebSocket connection failed'));
      };

      this.websocket.onmessage = (event) => {
        this._handleMessage(JSON.parse(event.data));
      };
    });
  }

  /**
   * 等待服务器配置
   */
  _waitForConfig() {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Timeout waiting for server config'));
      }, 5000);

      const checkConfig = setInterval(() => {
        if (this.serverUseAudioWorklet !== null) {
          clearTimeout(timeout);
          clearInterval(checkConfig);
          resolve();
        }
      }, 100);
    });
  }

  /**
   * 初始化音频采集
   */
  async _initAudio() {
    this.audioContext = new AudioContext({ sampleRate: 16000 });

    // 获取麦克风权限
    const constraints = {
      audio: {
        deviceId: this.microphoneId ? { exact: this.microphoneId } : undefined,
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    };

    this.microphone = await navigator.mediaDevices.getUserMedia(constraints);

    // 创建音频处理链
    const source = this.audioContext.createMediaStreamSource(this.microphone);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;

    if (this.serverUseAudioWorklet) {
      // 使用 AudioWorklet (PCM 模式)
      await this._initAudioWorklet(source);
    } else {
      // 使用 ScriptProcessor (兼容性模式)
      await this._initScriptProcessor(source);
    }

    source.connect(this.analyser);
  }

  /**
   * 初始化 AudioWorklet
   */
  async _initAudioWorklet(source) {
    const workletCode = `
      class PCMProcessor extends AudioWorkletProcessor {
        process(inputs, outputs, parameters) {
          const input = inputs[0];
          if (input && input[0]) {
            const floatData = input[0];
            const intData = new Int16Array(floatData.length);
            for (let i = 0; i < floatData.length; i++) {
              intData[i] = Math.max(-32768, Math.min(32767, floatData[i] * 32767));
            }
            this.port.postMessage(intData.buffer, [intData.buffer]);
          }
          return true;
        }
      }
      registerProcessor('pcm-processor', PCMProcessor);
    `;

    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);

    await this.audioContext.audioWorklet.addModule(url);

    this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm-processor');
    this.workletNode.port.onmessage = (event) => {
      if (this.isRecording && this.websocket?.readyState === WebSocket.OPEN) {
        this.websocket.send(event.data);
      }
    };

    source.connect(this.workletNode);
    this.workletNode.connect(this.audioContext.destination);
  }

  /**
   * 初始化 ScriptProcessor (兼容性模式)
   */
  _initScriptProcessor(source) {
    return new Promise((resolve, reject) => {
      const processor = this.audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (event) => {
        if (!this.isRecording) return;

        const floatData = event.inputBuffer.getChannelData(0);
        const intData = new Int16Array(floatData.length);
        for (let i = 0; i < floatData.length; i++) {
          intData[i] = Math.max(-32768, Math.min(32767, floatData[i] * 32767));
        }

        if (this.websocket?.readyState === WebSocket.OPEN) {
          this.websocket.send(intData.buffer);
        }
      };

      source.connect(processor);
      processor.connect(this.audioContext.destination);
      this.workletNode = processor;
      resolve();
    });
  }

  /**
   * 处理 WebSocket 消息
   */
  _handleMessage(data) {
    if (data.type === 'config') {
      this.serverUseAudioWorklet = data.useAudioWorklet;
      if (this.onReady) this.onReady(data);
      return;
    }

    if (data.type === 'ready_to_stop') {
      if (this.onReadyToStop) this.onReadyToStop();
      this.stop();
      return;
    }

    // 处理转录结果
    const result = {
      text: data.buffer_transcription || '',
      lines: data.lines || [],
      isFinal: !data.buffer_transcription,
      speaker: data.buffer_diarization || null,
      language: data.detected_language || null
    };

    if (data.buffer_transcription && this.onPartial) {
      this.onPartial(result);
    }

    if (this.onResult) {
      this.onResult(result);
    }
  }

  /**
   * 清理资源
   */
  _cleanup() {
    if (this.workletNode) {
      try {
        this.workletNode.disconnect();
      } catch (e) {}
      this.workletNode = null;
    }

    if (this.analyser) {
      try {
        this.analyser.disconnect();
      } catch (e) {}
      this.analyser = null;
    }

    if (this.microphone) {
      this.microphone.getTracks().forEach(track => track.stop());
      this.microphone = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.isRecording = false;
  }
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WhisperLiveKit;
}
