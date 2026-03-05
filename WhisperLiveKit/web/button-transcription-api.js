/**
 * Button Transcription API
 * 
 * 基于官方示例 live_transcription.js 封装的编程接口
 * 提供简单的按钮控制转录功能，与官方示例完全兼容
 * 
 * 使用示例：
 * 
 * // 创建客户端
 * const client = new ButtonTranscriptionAPI({
 *   serverUrl: 'ws://localhost:8000',
 *   language: 'zh',
 *   diarization: true,
 *   microphoneId: null // 使用默认麦克风
 * });
 * 
 * // 设置回调
 * client.onresult = (result) => {
 *   console.log('转录结果:', result.text);
 * };
 * 
 * client.onerror = (error) => {
 *   console.error('错误:', error.message);
 * };
 * 
 * client.onstatuschange = (status) => {
 *   console.log('状态变化:', status.state, status.message);
 * };
 * 
 * // 开始转录
 * await client.start();
 * 
 * // 停止转录
 * client.stop();
 */

class ButtonTranscriptionAPI {
  /**
   * 创建按钮转录客户端
   * @param {Object} options - 配置选项
   * @param {string} options.serverUrl - WebSocket服务器地址，默认 'ws://localhost:8000'
   * @param {string} options.language - 语言代码，如 'zh', 'en'，默认自动检测
   * @param {boolean} options.diarization - 是否启用说话人分离，默认 true
   * @param {string} options.microphoneId - 麦克风设备ID，默认使用系统默认麦克风
   * @param {boolean} options.autoStart - 连接后自动开始录音，默认 true
   */
  constructor(options = {}) {
    // 配置选项
    this.serverUrl = options.serverUrl || this._detectServerUrl();
    this.language = options.language || null;
    this.diarization = options.diarization !== false;
    this.microphoneId = options.microphoneId || null;
    this.autoStart = options.autoStart !== false;
    
    // 状态变量
    this.isRecording = false;
    this.isConnected = false;
    this.isProcessing = false;
    this.currentStatus = 'idle'; // idle, connecting, ready, recording, processing, error
    
    // WebSocket和音频资源
    this.websocket = null;
    this.audioContext = null;
    this.analyser = null;
    this.microphone = null;
    this.workletNode = null;
    this.recorder = null;
    this.recorderWorker = null;
    
    // 计时和UI相关
    this.startTime = null;
    this.timerInterval = null;
    this.animationFrame = null;
    this.lastReceivedData = null;
    this.lastSignature = null;
    
    // 配置相关
    this.serverUseAudioWorklet = null;
    this.configReadyResolve = null;
    this.configReady = new Promise((resolve) => {
      this.configReadyResolve = resolve;
    });
    
    // 回调函数
    this.onresult = null;          // 收到转录结果时调用
    this.onpartial = null;         // 收到临时结果时调用  
    this.onerror = null;           // 发生错误时调用
    this.onconnect = null;         // 连接成功时调用
    this.ondisconnect = null;      // 断开连接时调用
    this.onready = null;           // 服务器准备就绪时调用
    this.oncompletetion = null;    // 转录完成时调用
    this.onstatuschange = null;    // 状态变化时调用
    
    // 内部状态跟踪
    this._audioSource = null;
    this._shouldSendAudio = false;
    this._userClosing = false;
    this._waitingForStop = false;
    this._availableMicrophones = [];
    this._selectedMicrophoneId = null;
  }
  
  /**
   * 自动检测服务器地址
   * @private
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
   * 开始转录
   * @returns {Promise<void>}
   */
  async start() {
    if (this.isRecording) {
      throw new Error('已经在录音中');
    }
    
    if (this.currentStatus === 'error') {
      throw new Error('客户端处于错误状态，请重新初始化');
    }
    
    try {
      this._updateStatus('connecting', '正在连接服务器...');
      
      // 连接WebSocket
      await this._connectWebSocket();
      
      // 等待服务器配置
      await this.configReady;
      
      this._updateStatus('ready', '服务器准备就绪');
      
      // 开始录音
      await this._startRecording();
      
      this._updateStatus('recording', '正在录音...');
      this.isRecording = true;
      
    } catch (error) {
      this._updateStatus('error', `启动失败: ${error.message}`);
      this._cleanupResources();
      if (this.onerror) {
        this.onerror(error);
      }
      throw error;
    }
  }
  
  /**
   * 停止转录
   */
  stop() {
    if (!this.isRecording) {
      return;
    }
    
    this._updateStatus('processing', '正在处理音频...');
    this.isRecording = false;
    this._shouldSendAudio = false;
    this._userClosing = true;
    this._waitingForStop = true;
    
    // 发送空数据包表示结束
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      const emptyBlob = new Blob([], { type: 'audio/webm' });
      this.websocket.send(emptyBlob);
    }
    
    // 停止录音设备
    this._stopRecording();
    
    // 注意：不立即关闭WebSocket，等待服务器发送 ready_to_stop 消息
  }
  
  /**
   * 切换录音状态
   * @returns {Promise<void>}
   */
  async toggle() {
    if (this.isRecording) {
      this.stop();
    } else {
      await this.start();
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
      throw new Error(`获取麦克风失败: ${error.message}`);
    }
  }
  
  /**
   * 设置麦克风设备
   * @param {string} deviceId - 麦克风设备ID
   */
  setMicrophone(deviceId) {
    this.microphoneId = deviceId;
    this._selectedMicrophoneId = deviceId;
    
    // 如果正在录音，需要重新启动
    if (this.isRecording) {
      this._updateStatus('connecting', '正在切换麦克风...');
      this.stop();
      setTimeout(() => {
        this.start().catch(error => {
          if (this.onerror) this.onerror(error);
        });
      }, 1000);
    }
  }
  
  /**
   * 连接WebSocket服务器
   * @private
   */
  async _connectWebSocket() {
    return new Promise((resolve, reject) => {
      const wsUrl = `${this.serverUrl}/asr`;
      
      try {
        this.websocket = new WebSocket(wsUrl);
      } catch (error) {
        reject(new Error(`创建WebSocket失败: ${error.message}`));
        return;
      }
      
      this.websocket.onopen = () => {
        this.isConnected = true;
        this._updateStatus('connected', '已连接服务器');
        if (this.onconnect) this.onconnect();
        resolve();
      };
      
      this.websocket.onclose = () => {
        this.isConnected = false;
        this._userClosing = false;
        this._waitingForStop = false;
        this._cleanupResources();
        if (this.ondisconnect) this.ondisconnect();
        this._updateStatus('idle', '已断开连接');
      };
      
      this.websocket.onerror = () => {
        reject(new Error('WebSocket连接错误'));
      };
      
      this.websocket.onmessage = (event) => {
        this._handleWebSocketMessage(event);
      };
    });
  }
  
  /**
   * 处理WebSocket消息
   * @private
   */
  _handleWebSocketMessage(event) {
    try {
      const data = JSON.parse(event.data);
      
      // 处理配置消息
      if (data.type === 'config') {
        this.serverUseAudioWorklet = !!data.useAudioWorklet;
        if (this.configReadyResolve) {
          this.configReadyResolve();
          this.configReadyResolve = null;
        }
        if (this.onready) this.onready(data);
        return;
      }
      
      // 处理完成消息
      if (data.type === 'ready_to_stop') {
        this._waitingForStop = false;
        this._userClosing = false;
        this._cleanupResources();
        if (this.websocket) {
          this.websocket.close();
        }
        if (this.oncompletetion) this.oncompletetion(data);
        this._updateStatus('idle', '转录完成');
        return;
      }
      
      this.lastReceivedData = data;
      
      // 提取转录结果
      const result = this._extractTranscriptionResult(data);
      
      // 调用回调函数
      if (result.isPartial && this.onpartial) {
        this.onpartial(result);
      }
      
      if (this.onresult) {
        this.onresult(result);
      }
      
    } catch (error) {
      console.error('处理WebSocket消息失败:', error);
    }
  }
  
  /**
   * 从服务器数据中提取转录结果
   * @private
   */
  _extractTranscriptionResult(data) {
    const {
      lines = [],
      buffer_transcription = '',
      buffer_diarization = '',
      buffer_translation = '',
      remaining_time_transcription = 0,
      remaining_time_diarization = 0,
      status = 'active_transcription',
    } = data;
    
    let text = '';
    
    // 如果有缓冲区转录，使用缓冲区内容（中间结果）
    if (buffer_transcription) {
      text = buffer_transcription;
    }
    // 如果没有缓冲区转录但有lines，使用最后一条完成的句子（最终结果）
    else if (lines && lines.length > 0) {
      const lastLine = lines[lines.length - 1];
      if (lastLine && lastLine.text) {
        text = lastLine.text;
      }
    }
    
    // 处理lines中的数字字段，确保start和end是数字类型
    const processedLines = lines.map(line => {
      if (!line) return line;
      
      const processedLine = { ...line };
      
      // 转换start和end为数字
      if (line.start !== undefined) {
        processedLine.start = parseFloat(line.start);
        if (isNaN(processedLine.start)) {
          processedLine.start = line.start; // 如果转换失败，保留原始值
        }
      }
      
      if (line.end !== undefined) {
        processedLine.end = parseFloat(line.end);
        if (isNaN(processedLine.end)) {
          processedLine.end = line.end; // 如果转换失败，保留原始值
        }
      }
      
      // 确保speaker是数字（如果存在）
      if (line.speaker !== undefined) {
        const speakerNum = parseFloat(line.speaker);
        if (!isNaN(speakerNum)) {
          processedLine.speaker = speakerNum;
        }
      }
      
      return processedLine;
    });
    
    // 确保remainingTime是数字
    const remainingTime = parseFloat(remaining_time_transcription);
    
    return {
      text,
      lines: processedLines,
      isPartial: !!buffer_transcription, // 有缓冲区转录就是中间结果
      speaker: buffer_diarization || null,
      language: data.detected_language || null,
      status,
      remainingTime: isNaN(remainingTime) ? 0 : remainingTime
    };
  }
  
  /**
   * 开始录音
   * @private
   */
  async _startRecording() {
    try {
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
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // 创建音频上下文
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this._audioSource = this.audioContext.createMediaStreamSource(stream);
      this._audioSource.connect(this.analyser);
      
      // 根据服务器配置选择音频处理方式
      if (this.serverUseAudioWorklet) {
        await this._setupAudioWorklet();
      } else {
        this._setupMediaRecorder(stream);
      }
      
      this._shouldSendAudio = true;
      this.startTime = Date.now();
      
      // 开始计时
      this.timerInterval = setInterval(() => {
        // 可以在这里更新UI计时器
      }, 1000);
      
    } catch (error) {
      throw new Error(`启动录音失败: ${error.message}`);
    }
  }
  
  /**
   * 设置AudioWorklet（PCM模式）
   * @private
   */
  async _setupAudioWorklet() {
    if (!this.audioContext.audioWorklet) {
      throw new Error('浏览器不支持AudioWorklet');
    }
    
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
      if (this._shouldSendAudio && this.websocket?.readyState === WebSocket.OPEN) {
        this.websocket.send(event.data);
      }
    };
    
    this._audioSource.connect(this.workletNode);
    this.workletNode.connect(this.audioContext.destination);
  }
  
  /**
   * 设置MediaRecorder（WebM模式）
   * @private
   */
  _setupMediaRecorder(stream) {
    try {
      this.recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    } catch (e) {
      this.recorder = new MediaRecorder(stream);
    }
    
    this.recorder.ondataavailable = (event) => {
      if (this._shouldSendAudio && this.websocket?.readyState === WebSocket.OPEN) {
        if (event.data && event.data.size > 0) {
          this.websocket.send(event.data);
        }
      }
    };
    
    this.recorder.start(100); // 100ms的块大小
  }
  
  /**
   * 停止录音
   * @private
   */
  _stopRecording() {
    if (this.recorder) {
      try {
        this.recorder.stop();
      } catch (e) {
        // 忽略错误
      }
      this.recorder = null;
    }
    
    if (this.recorderWorker) {
      this.recorderWorker.terminate();
      this.recorderWorker = null;
    }
    
    if (this.workletNode) {
      try {
        this.workletNode.port.onmessage = null;
        this.workletNode.disconnect();
      } catch (e) {
        // 忽略错误
      }
      this.workletNode = null;
    }
    
    if (this._audioSource) {
      try {
        this._audioSource.disconnect();
      } catch (e) {
        // 忽略错误
      }
      this._audioSource = null;
    }
    
    if (this.analyser) {
      this.analyser = null;
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    
    this.startTime = null;
  }
  
  /**
   * 清理所有资源
   * @private
   */
  _cleanupResources() {
    this._stopRecording();
    
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    this.isRecording = false;
    this.isConnected = false;
    this._shouldSendAudio = false;
    this._userClosing = false;
    this._waitingForStop = false;
  }
  
  /**
   * 更新状态
   * @private
   */
  _updateStatus(state, message) {
    const previousState = this.currentStatus;
    this.currentStatus = state;
    
    if (this.onstatuschange && previousState !== state) {
      this.onstatuschange({
        previous: previousState,
        current: state,
        message: message
      });
    }
  }
  
  /**
   * 销毁客户端，释放所有资源
   */
  destroy() {
    this._cleanupResources();
    this._updateStatus('idle', '客户端已销毁');
  }
}

// 全局导出
if (typeof window !== 'undefined') {
  window.ButtonTranscriptionAPI = ButtonTranscriptionAPI;
}