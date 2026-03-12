/**
 * Hotword Detection API
 * 
 * 唤醒词检测前端API，与服务器端/hotword WebSocket端点通信
 * 检测到唤醒词后自动触发回调，可以自动启动转录
 * 
 * 使用示例：
 * 
 * // 创建唤醒词检测客户端
 * const hotwordClient = new HotwordDetectionAPI({
 *   serverUrl: 'ws://localhost:8000',
 *   onWakewordDetected: (wakeword) => {
 *     console.log('检测到唤醒词:', wakeword);
 *     // 在这里启动转录
 *     startTranscription();
 *   }
 * });
 * 
 * // 开始监听唤醒词
 * await hotwordClient.start();
 * 
 * // 停止监听
 * hotwordClient.stop();
 */

class HotwordDetectionAPI {
  /**
   * 创建唤醒词检测客户端
   * @param {Object} options - 配置选项
   * @param {string} options.serverUrl - WebSocket服务器地址，默认 'ws://localhost:8000'
   * @param {string} options.microphoneId - 麦克风设备ID，默认使用系统默认麦克风
   * @param {Function} options.onWakewordDetected - 检测到唤醒词时的回调函数
   * @param {Function} options.onError - 发生错误时的回调函数
   * @param {Function} options.onStatusChange - 状态变化时的回调函数
   * @param {Function} options.onConnected - 连接成功时的回调函数
   * @param {Function} options.onDisconnected - 断开连接时的回调函数
   */
  constructor(options = {}) {
    // 配置选项
    this.serverUrl = options.serverUrl || this._detectServerUrl();
    this.microphoneId = options.microphoneId || null;
    
    // 状态变量
    this.isListening = false;
    this.isConnected = false;
    this.currentStatus = 'idle'; // idle, connecting, connected, listening, error
    this.connectionId = null;
    
    // WebSocket和音频资源
    this.websocket = null;
    this.audioContext = null;
    this.microphone = null;
    this.workletNode = null;
    this.recorder = null;
    this.recorderWorker = null;
    
    // 音频处理相关
    this.sampleRate = 16000;
    this.chunkDuration = 0.1; // 100ms
    this.chunkSize = this.sampleRate * this.chunkDuration;
    
    // 回调函数
    this.onWakewordDetected = options.onWakewordDetected || null;
    this.onError = options.onError || null;
    this.onStatusChange = options.onStatusChange || null;
    this.onConnected = options.onConnected || null;
    this.onDisconnected = options.onDisconnected || null;
    
    // 内部状态跟踪
    this._audioSource = null;
    this._shouldSendAudio = false;
    this._userClosing = false;
    this._availableMicrophones = [];
    this._selectedMicrophoneId = null;
    
    logger.info(`唤醒词检测API初始化，服务器: ${this.serverUrl}`);
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
   * 更新状态并触发回调
   * @private
   */
  _updateStatus(newStatus, message = '') {
    const oldStatus = this.currentStatus;
    this.currentStatus = newStatus;
    
    if (oldStatus !== newStatus && this.onStatusChange) {
      this.onStatusChange({
        oldStatus,
        newStatus,
        message
      });
    }
  }
  
  /**
   * 连接WebSocket服务器
   * @private
   */
  async _connectWebSocket() {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = `${this.serverUrl.replace(/^http/, 'ws')}/hotword`;
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
          this.isConnected = true;
          this._updateStatus('connected', '已连接到唤醒词检测服务器');
          if (this.onConnected) {
            this.onConnected();
          }
          resolve();
        };
        
        this.websocket.onclose = () => {
          this.isConnected = false;
          this._updateStatus('idle', '已断开连接');
          if (this.onDisconnected) {
            this.onDisconnected();
          }
          if (!this._userClosing) {
            this._handleError(new Error('WebSocket连接意外关闭'));
          }
        };
        
        this.websocket.onerror = () => {
          reject(new Error('WebSocket连接错误'));
        };
        
        this.websocket.onmessage = (event) => {
          this._handleWebSocketMessage(event);
        };
        
      } catch (error) {
        reject(new Error(`创建WebSocket失败: ${error.message}`));
      }
    });
  }
  
  /**
   * 处理WebSocket消息
   * @private
   */
  _handleWebSocketMessage(event) {
    try {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'config':
          // 收到服务器配置
          this.connectionId = data.connection_id;
          this.sampleRate = data.sample_rate || this.sampleRate;
          this.chunkSize = this.sampleRate * this.chunkDuration;
          logger.info(`唤醒词检测配置: 连接ID=${this.connectionId}, 采样率=${this.sampleRate}Hz`);
          break;
          
        case 'wakeword_detected':
          // 收到唤醒词检测通知
          const wakeword = data.wakeword;
          const timestamp = data.timestamp;
          
          logger.info(`🎯 检测到唤醒词: ${wakeword}`);
          
          if (this.onWakewordDetected) {
            this.onWakewordDetected(wakeword, timestamp);
          }
          break;
          
        default:
          logger.warn(`未知的WebSocket消息类型: ${data.type}`);
      }
    } catch (error) {
      logger.error(`处理WebSocket消息失败: ${error}`);
    }
  }
  
  /**
   * 开始录音和发送音频
   * @private
   */
  async _startRecording() {
    try {
      // 获取用户媒体权限
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: this.microphoneId ? { exact: this.microphoneId } : undefined,
          sampleRate: { ideal: this.sampleRate },
          channelCount: 1
        }
      });
      
      // 创建音频上下文
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate
      });
      
      // 创建媒体流源
      this._audioSource = this.audioContext.createMediaStreamSource(stream);
      
      // 检查是否支持AudioWorklet
      if (this.audioContext.audioWorklet && this._shouldUseAudioWorklet()) {
        await this._setupAudioWorklet();
      } else {
        await this._setupScriptProcessor();
      }
      
      this._shouldSendAudio = true;
      this._updateStatus('listening', '正在监听唤醒词...');
      
    } catch (error) {
      throw new Error(`启动录音失败: ${error.message}`);
    }
  }
  
  /**
   * 判断是否使用AudioWorklet
   * @private
   */
  _shouldUseAudioWorklet() {
    // 可以根据需要调整条件
    return typeof window !== 'undefined' && 
           this.audioContext.audioWorklet &&
           this.sampleRate === 16000; // AudioWorklet更适合固定采样率
  }
  
  /**
   * 设置AudioWorklet处理
   * @private
   */
  async _setupAudioWorklet() {
    try {
      // 加载AudioWorklet处理器
      await this.audioContext.audioWorklet.addModule('pcm_worklet.js');
      
      // 创建AudioWorkletNode
      this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm-forwarder', {
        processorOptions: {
          sampleRate: this.sampleRate,
          chunkSize: this.chunkSize
        }
      });
      
      // 连接音频处理链
      this._audioSource.connect(this.workletNode);
      this.workletNode.connect(this.audioContext.destination);
      
      // 处理音频数据
      this.workletNode.port.onmessage = (event) => {
        if (this._shouldSendAudio && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
          const pcmData = event.data;
          // 直接发送 Float32 格式数据
          this.websocket.send(pcmData.buffer);
        }
      };
      
    } catch (error) {
      logger.warn(`AudioWorklet设置失败，降级到ScriptProcessor: ${error.message}`);
      await this._setupScriptProcessor();
    }
  }
  
  /**
   * 设置ScriptProcessor处理（降级方案）
   * @private
   */
  async _setupScriptProcessor() {
    // 创建ScriptProcessorNode
    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    
    let audioBuffer = [];
    
    this.processorNode.onaudioprocess = (event) => {
      if (!this._shouldSendAudio) return;
      
      // 获取音频数据
      const inputData = event.inputBuffer.getChannelData(0);
      audioBuffer.push(...inputData);
      
      // 当累积足够数据时发送
      while (audioBuffer.length >= this.chunkSize) {
        const chunk = audioBuffer.slice(0, this.chunkSize);
        audioBuffer = audioBuffer.slice(this.chunkSize);
        
        // 转换为Int16 PCM
        const pcmData = this._float32ToInt16(chunk);
        
        // 发送到WebSocket
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
          this.websocket.send(pcmData);
        }
      }
    };
    
    // 连接音频处理链
    this._audioSource.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }
  
  /**
   * 将Float32音频数据转换为Int16 PCM
   * @private
   */
  _float32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      let val = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = val < 0 ? val * 0x8000 : val * 0x7FFF;
    }
    return int16Array.buffer;
  }
  
  /**
   * 停止录音
   * @private
   */
  _stopRecording() {
    this._shouldSendAudio = false;
    
    // 断开音频节点
    if (this._audioSource) {
      this._audioSource.disconnect();
      this._audioSource = null;
    }
    
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    
    // 关闭音频上下文
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
  
  /**
   * 处理错误
   * @private
   */
  _handleError(error) {
    logger.error(`唤醒词检测错误: ${error.message}`);
    this._updateStatus('error', error.message);
    
    if (this.onError) {
      this.onError(error);
    }
    
    this.stop();
  }
  
  /**
   * 清理资源
   * @private
   */
  _cleanupResources() {
    this._stopRecording();
    
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    this.isConnected = false;
    this.isListening = false;
    this.connectionId = null;
  }
  
  /**
   * 开始监听唤醒词
   * @returns {Promise<void>}
   */
  async start() {
    if (this.isListening) {
      throw new Error('已经在监听中');
    }
    
    if (this.currentStatus === 'error') {
      throw new Error('客户端处于错误状态，请重新初始化');
    }
    
    try {
      this._updateStatus('connecting', '正在连接唤醒词检测服务器...');
      
      // 连接WebSocket
      await this._connectWebSocket();
      
      this._updateStatus('connected', '已连接到唤醒词检测服务器');
      
      // 开始录音
      await this._startRecording();
      
      this._updateStatus('listening', '正在监听唤醒词...');
      this.isListening = true;
      
    } catch (error) {
      this._handleError(error);
      throw error;
    }
  }
  
  /**
   * 停止监听唤醒词
   */
  stop() {
    if (!this.isListening) {
      return;
    }
    
    this._userClosing = true;
    this.isListening = false;
    this._shouldSendAudio = false;
    
    this._updateStatus('disconnecting', '正在停止监听...');
    
    // 清理资源
    this._cleanupResources();
    
    this._updateStatus('idle', '已停止监听');
    this._userClosing = false;
  }
  
  /**
   * 获取可用的麦克风列表
   * @returns {Promise<Array<{deviceId: string, label: string}>>}
   */
  static async getMicrophones() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      throw new Error('浏览器不支持获取设备列表');
    }
    
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      const devices = await navigator.mediaDevices.enumerateDevices();
      
      return devices
        .filter(device => device.kind === 'audioinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `麦克风 ${device.deviceId.slice(0, 8)}...`
        }));
    } catch (error) {
      throw new Error(`获取麦克风列表失败: ${error.message}`);
    }
  }
  
  /**
   * 销毁客户端，清理所有资源
   */
  destroy() {
    this.stop();
    
    // 清理所有引用
    this.onWakewordDetected = null;
    this.onError = null;
    this.onStatusChange = null;
    this.onConnected = null;
    this.onDisconnected = null;
    
    logger.info('唤醒词检测客户端已销毁');
  }
}

// 导出全局日志函数
const logger = {
  info: (message) => console.log(`[HotwordDetectionAPI] ${message}`),
  warn: (message) => console.warn(`[HotwordDetectionAPI] ${message}`),
  error: (message) => console.error(`[HotwordDetectionAPI] ${message}`),
  debug: (message) => console.debug(`[HotwordDetectionAPI] ${message}`)
};