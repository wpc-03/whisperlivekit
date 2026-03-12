/**
 * WhisperHotword - 唤醒词模式语音转录客户端
 * 
 * @description 用于唤醒词模式的语音转录功能，支持唤醒词检测和自动转录
 * 
 * @example
 * const hotword = new WhisperHotword({
 *   serverUrl: 'ws://localhost:8000'
 * });
 * 
 * hotword.onhotword = (keyword) => {
 *   console.log('检测到唤醒词:', keyword);
 * };
 * 
 * hotword.onresult = (result) => {
 *   console.log('转录结果:', result.text);
 * };
 * 
 * await hotword.start();
 */

class WhisperHotword {
  /**
   * 创建唤醒词转录客户端
   * @param {Object} options - 配置选项
   * @param {string} [options.serverUrl='ws://localhost:8000'] - WebSocket服务器地址
   * @param {string} [options.microphoneId=null] - 麦克风设备ID
   * @param {number} [options.autoStopTimeout=6000] - 用户停止说话后自动结束转录的等待时间(毫秒)
   * @param {boolean} [options.autoStartTranscription=true] - 检测到唤醒词后自动开始转录
   * @param {string} [options.logLevel='info'] - 日志级别: debug, info, warn, error
   * @param {Function} [options.logHandler=null] - 自定义日志处理器
   * @description 转录结束后，系统会固定等待500ms再重新开始监听唤醒词
   */
  constructor(options = {}) {
    // 日志系统
    this.logger = new Logger({
      level: options.logLevel || 'info',
      handler: options.logHandler,
      prefix: 'WhisperHotword'
    });

    // 配置选项
    this.serverUrl = options.serverUrl || this._detectServerUrl();
    this.microphoneId = options.microphoneId || null;
    this.silenceTimeout = 500; // 固定值，转录结束后等待500ms再重新开始监听
    this.autoStopTimeout = options.autoStopTimeout || 6000; // 用户停止说话后自动结束转录的等待时间
    this.autoStartTranscription = options.autoStartTranscription !== false;

    // 状态变量
    this.isListening = false;
    this.isConnected = false;
    this.isRecording = false;
    this.currentStatus = 'idle';

    // WebSocket和音频资源
    this.websocket = null;
    this.audioContext = null;
    this.analyser = null;
    this.microphone = null;
    this.workletNode = null;
    this.recorder = null;
    this.recorderWorker = null;
    this._autoStopTimer = null;

    // 唤醒词检测相关
    this.sampleRate = 16000;
    this.chunkDuration = 0.1;
    this.chunkSize = this.sampleRate * this.chunkDuration;
    this.connectionId = null;

    // 转录相关
    this.transcriptionClient = null;

    // 回调函数
    this.onhotword = null;
    this.onresult = null;
    this.onpartial = null;
    this.onerror = null;
    this.onconnect = null;
    this.ondisconnect = null;
    this.onstatuschange = null;
    this.onready = null;

    // 内部状态
    this._audioSource = null;
    this._shouldSendAudio = false;
    this._sendFloat32 = true;
    this._userClosing = false;
    this._isTranscribing = false;

    // 配置就绪 - 每次连接都需要重新创建
    this._resetConfigReady();

    // 服务器配置，由服务器在config消息中指定
    this.serverUseAudioWorklet = null;

    this.logger.info('WhisperHotword 初始化完成', { serverUrl: this.serverUrl });
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
   * 重置configReady promise，用于每次重新连接
   * @private
   */
  _resetConfigReady() {
    this.configReadyResolve = null;
    this.configReady = new Promise((resolve) => {
      this.configReadyResolve = resolve;
    });
  }

  /**
   * 更新状态
   * @private
   */
  _updateStatus(state, message) {
    const previousState = this.currentStatus;
    this.currentStatus = state;
    
    this.logger.info(`状态变化: ${previousState} -> ${state}`, { message });

    if (this.onstatuschange && previousState !== state) {
      this.onstatuschange({
        previous: previousState,
        current: state,
        message: message
      });
    }
  }

  /**
   * 开始监听唤醒词
   * @returns {Promise<void>}
   */
  async start() {
    if (this.isListening) {
      const error = new WhisperError('已经在监听中', ErrorCodes.ALREADY_RECORDING);
      this.logger.warn(error.message);
      return;
    }

    try {
      this.logger.info('正在启动唤醒词检测...');
      this._updateStatus('connecting', '正在连接服务器...');

      await this._connectWebSocket();

      // 等待服务器配置，确保 serverUseAudioWorklet 已设置
      await this.configReady;
      this.logger.debug('已收到服务器配置', { useAudioWorklet: this.serverUseAudioWorklet });

      // 开始音频捕获
      await this._startAudioCapture();

      this._updateStatus('listening', '正在监听唤醒词...');
      this.isListening = true;

      this.logger.info('唤醒词检测已启动');

    } catch (error) {
      this._updateStatus('error', `启动失败: ${error.message}`);
      this._cleanupResources();
      
      this.logger.error('启动失败', error);
      
      if (this.onerror) {
        this.onerror(error);
      }
      throw error;
    }
  }

  /**
   * 停止监听唤醒词
   * @returns {void}
   */
  stop() {
    if (!this.isListening) {
      return;
    }

    this.logger.info('正在停止唤醒词检测...');
    this.isListening = false;
    this._shouldSendAudio = false;
    this._userClosing = true;

    this._stopAudioCapture();

    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.close();
    }

    this._updateStatus('idle', '已停止监听');
    this.logger.info('唤醒词检测已停止');
  }

  /**
   * 销毁客户端
   */
  destroy() {
    this.logger.info('正在销毁客户端');
    this.stop();
    this._cleanupResources();
    this._updateStatus('idle', '客户端已销毁');
  }

  /**
   * 获取可用的麦克风列表
   * @returns {Promise<Array<{deviceId: string, label: string}>>}
   */
  static async getMicrophones() {
    try {
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
      const err = new WhisperError(`获取麦克风失败: ${error.message}`, ErrorCodes.MICROPHONE_ACCESS_DENIED);
      this.logger.error(err.message, error);
      throw err;
    }
  }

  /**
   * 获取音频分析器
   * @returns {AnalyserNode|null}
   */
  getAnalyser() {
    return this.analyser;
  }

  /**
   * 连接WebSocket服务器
   * @private
   */
  async _connectWebSocket() {
    return new Promise((resolve, reject) => {
      // 先关闭旧的连接
      if (this.websocket) {
        try {
          this.websocket.close();
        } catch (e) {}
        this.websocket = null;
      }

      // 重置configReady
      this._resetConfigReady();

      // 确保使用 ws 协议
      const wsBaseUrl = this.serverUrl.replace(/^http/, 'ws');
      const wsUrl = `${wsBaseUrl}/hotword`;
      this.logger.debug(`连接WebSocket: ${wsUrl}`);

      try {
        this.websocket = new WebSocket(wsUrl);
      } catch (error) {
        const err = new WhisperError(`创建WebSocket失败: ${error.message}`, ErrorCodes.WEBSOCKET_ERROR);
        this.logger.error(err.message, error);
        reject(err);
        return;
      }

      this.websocket.onopen = () => {
        this.isConnected = true;
        this._updateStatus('connected', '已连接服务器');
        this.logger.info('WebSocket已连接');
        
        if (this.onconnect) this.onconnect();
        resolve();
      };

      this.websocket.onclose = () => {
        this.isConnected = false;
        // 只有当不是在转录时才重置 isListening
        if (!this._isTranscribing) {
          this.isListening = false;
        }
        this._userClosing = false;
        // 只有当不是在转录时才清理资源
        if (!this._isTranscribing) {
          this._cleanupResources();
        }
        this.logger.info('WebSocket已断开');
        
        if (this.ondisconnect) this.ondisconnect();
        // 只有当不是在转录时才更新状态
        if (!this._isTranscribing) {
          this._updateStatus('idle', '已断开连接');
        }
      };

      this.websocket.onerror = () => {
        const err = new WhisperError('WebSocket连接错误', ErrorCodes.WEBSOCKET_ERROR);
        this.logger.error(err.message);
        reject(err);
      };

      this.websocket.onmessage = async (event) => {
        await this._handleWebSocketMessage(event);
      };
    });
  }

  /**
   * 处理WebSocket消息
   * @private
   */
  async _handleWebSocketMessage(event) {
    try {
      const data = JSON.parse(event.data);

      if (data.type === 'config') {
        // 保存服务器配置，用于决定使用哪种音频传输方式
        this.serverUseAudioWorklet = !!data.useAudioWorklet;
        
        if (this.configReadyResolve) {
          this.configReadyResolve();
          this.configReadyResolve = null;
        }
        this.logger.debug('收到服务器配置', data);
        if (this.onready) this.onready(data);
        return;
      }

      if (data.type === 'hotword' || data.type === 'wakeword_detected') {
        const keyword = data.hotword || data.wakeword || data.keyword || 'unknown';
        this.logger.info('检测到唤醒词:', keyword);
        
        if (this.onhotword) {
          this.onhotword(keyword, data);
        }

        if (this.autoStartTranscription) {
          this.logger.info('正在启动转录...');
          await this._startTranscription();
        }
        return;
      }

      if (data.type === 'transcription') {
        const result = this._extractTranscriptionResult(data);
        if (this.onresult) {
          this.onresult(result);
        }
        return;
      }

      if (data.type === 'partial_transcription') {
        const result = this._extractTranscriptionResult(data);
        result.isPartial = true;
        if (this.onpartial) {
          this.onpartial(result);
        }
        return;
      }

    } catch (error) {
      this.logger.error('处理WebSocket消息失败', error);
    }
  }

  /**
   * 提取转录结果
   * @private
   */
  _extractTranscriptionResult(data) {
    const result = {
      text: '',
      lines: [],
      isPartial: false,
      buffer_transcription: data.buffer_transcription || ''
    };

    // 处理 buffer_transcription（中间结果）
    if (data.buffer_transcription) {
      result.buffer_transcription = data.buffer_transcription;
      result.isPartial = true;
    }

    // 处理 lines 数组
    if (data.lines && Array.isArray(data.lines)) {
      result.lines = data.lines.map(line => ({
        text: line.text || '',
        start: parseFloat(line.start) || 0,
        end: parseFloat(line.end) || 0,
        speaker: parseFloat(line.speaker) || null,
        detected_language: line.detected_language || null
      }));
      
      // 如果 text 为空但有 lines，从最后一条提取文本
      if (!result.text && result.lines.length > 0) {
        const lastLine = result.lines[result.lines.length - 1];
        if (lastLine && lastLine.text) {
          result.text = lastLine.text;
          result.isPartial = false;
        }
      }
    }

    return result;
  }

  /**
   * 开始音频捕获
   * @private
   */
  async _startAudioCapture() {
    // 清理旧的音频资源
    if (this.audioContext && this.audioContext.state !== 'closed') {
      try {
        await this.audioContext.close();
      } catch (e) {}
      this.audioContext = null;
    }
    this.workletNode = null;
    if (this.recorderWorker) {
      this.recorderWorker.terminate();
      this.recorderWorker = null;
    }
    if (this.recorder) {
      try { this.recorder.stop(); } catch (e) {}
      this.recorder = null;
    }

    try {
      const constraints = {
        audio: {
          deviceId: this.microphoneId ? { exact: this.microphoneId } : undefined,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      };

      this.logger.debug('请求麦克风权限');
      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      // 使用16000采样率，与服务器期望一致
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      
      // 等待 AudioContext 真正准备好
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }
      // 给一点时间让 AudioContext 初始化
      await new Promise(resolve => setTimeout(resolve, 100));
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this._audioSource = this.audioContext.createMediaStreamSource(stream);
      this._audioSource.connect(this.analyser);

      // 唤醒词检测使用 AudioWorklet + recorder_worker
      try {
        await this._setupAudioWorklet(stream);
      } catch (error) {
        this.logger.warn(`AudioWorklet设置失败: ${error.message}`);
        throw error;
      }

      this._shouldSendAudio = true;
      this.logger.info('音频捕获已启动');

    } catch (error) {
      const err = new WhisperError(`启动音频捕获失败: ${error.message}`, ErrorCodes.AUDIO_PROCESSING_ERROR);
      this.logger.error(err.message, error);
      throw err;
    }
  }

  /**
   * 设置AudioWorklet
   * @private
   */
  async _setupAudioWorklet(stream) {
    if (!this.audioContext.audioWorklet) {
      throw new WhisperError('浏览器不支持AudioWorklet', ErrorCodes.AUDIO_PROCESSING_ERROR);
    }

    // 加载 pcm_worklet.js
    await this.audioContext.audioWorklet.addModule('pcm_worklet.js');

    // 创建 AudioWorkletNode
    this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm-forwarder');

    // 创建 recorder_worker.js 用于处理音频（重采样和转换）
    this.recorderWorker = new Worker('recorder_worker.js');
    this.recorderWorker.postMessage({
      command: 'init',
      config: {
        sampleRate: this.audioContext.sampleRate,
        targetSampleRate: 16000
      }
    });

    // worker 处理完成的音频数据，发送给服务器（仅转录期间）
    this.recorderWorker.onmessage = (e) => {
      if (!this._sendFloat32 && this._shouldSendAudio && this.websocket?.readyState === WebSocket.OPEN) {
        this.websocket.send(e.data.buffer);
      }
    };

    this._float32Processor = null;

    // AudioWorklet 收到音频后进行处理
    this.workletNode.port.onmessage = (e) => {
      const data = e.data;
      const ab = data instanceof ArrayBuffer ? data : data.buffer;

      // 唤醒词期间直接发送 float32 数据给服务器
      if (this._shouldSendAudio && this._sendFloat32 && this.websocket?.readyState === WebSocket.OPEN) {
        this.websocket.send(ab);
      } else if (this._shouldSendAudio && !this._sendFloat32) {
        // 转录期间发送给 worker 处理
        this.recorderWorker.postMessage({
          command: 'record',
          buffer: ab
        }, [ab]);
      } else {
        this.logger.debug('未发送，_shouldSendAudio:', this._shouldSendAudio, '_sendFloat32:', this._sendFloat32);
      }
    };

    this._audioSource.connect(this.workletNode);
    this.workletNode.connect(this.audioContext.destination);
    
    this.logger.debug('AudioWorklet已设置');
  }

  _processAndSendFloat32(inputBuffer) {
    this.websocket.send(inputBuffer);
  }

  /**
   * 检测浏览器是否支持AudioWorklet
   * @private
   */
  _detectAudioWorkletSupport() {
    try {
      const testCtx = new (window.AudioContext || window.webkitAudioContext)();
      const supported = !!testCtx.audioWorklet;
      testCtx.close();
      return supported;
    } catch (e) {
      return false;
    }
  }

  /**
   * 启动自动停止计时器
   * @private
   */
  _startAutoStopTimer() {
    this._clearAutoStopTimer();
    this.logger.info(`启动自动停止计时器，超时时间: ${this.autoStopTimeout}ms`);
    console.log(`WhisperHotword: 启动自动停止计时器，超时时间: ${this.autoStopTimeout}ms`);
    this._autoStopTimer = setTimeout(() => {
      this.logger.info('自动停止计时器触发，检查是否需要停止转录...');
      console.log('WhisperHotword: 自动停止计时器触发');
      if (this._isTranscribing && this.transcriptionClient) {
        this.logger.info('自动停止计时器触发，正在停止转录...');
        console.log('WhisperHotword: 调用 transcriptionClient.stop()');
        this.transcriptionClient.stop();
      } else {
        this.logger.info('自动停止计时器触发，但转录已停止或 transcriptionClient 不存在');
      }
    }, this.autoStopTimeout);
  }

  /**
   * 重置自动停止计时器
   * @private
   */
  _resetAutoStopTimer() {
    this._startAutoStopTimer();
  }

  /**
   * 清除自动停止计时器
   * @private
   */
  _clearAutoStopTimer() {
    if (this._autoStopTimer) {
      clearTimeout(this._autoStopTimer);
      this._autoStopTimer = null;
    }
  }

  /**
   * 设置MediaRecorder
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

    this.recorder.start(100);
    this.logger.debug('MediaRecorder已设置');
  }

  /**
   * 停止音频捕获
   * @private
   */
  _stopAudioCapture() {
    if (this.recorder) {
      try { this.recorder.stop(); } catch (e) {}
      this.recorder = null;
    }

    if (this.recorderWorker) {
      try { this.recorderWorker.terminate(); } catch (e) {}
      this.recorderWorker = null;
    }

    if (this.workletNode) {
      try {
        this.workletNode.port.onmessage = null;
        this.workletNode.disconnect();
      } catch (e) {}
      this.workletNode = null;
    }

    if (this._audioSource) {
      try { this._audioSource.disconnect(); } catch (e) {}
      this._audioSource = null;
    }

    if (this.analyser) {
      this.analyser = null;
    }

    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
  }

  /**
   * 开始转录（检测到唤醒词后）
   * @private
   */
  async _startTranscription() {
    if (this._isTranscribing) {
      this.logger.warn('转录已在进行中');
      return;
    }

    this._isTranscribing = true;
    this._sendFloat32 = false;
    this._updateStatus('transcribing', '正在转录...');
    
    // 启动自动停止计时器
    this._startAutoStopTimer();

    // 停止发送音频到唤醒词 WebSocket，但不立即关闭
    this._shouldSendAudio = false;

    if (this.transcriptionClient) {
      try {
        this.transcriptionClient.destroy();
      } catch (e) {}
    }

    this.transcriptionClient = new WhisperButton({
      serverUrl: this.serverUrl,
      logLevel: 'info'
    });

    this.transcriptionClient.onresult = (result) => {
      if (this.onresult) {
        this.onresult(result);
      }
    };

    this.transcriptionClient.onpartial = (result) => {
      if (this.onpartial) {
        this.onpartial(result);
      }
      
      // 只有当有实际转录内容时才重置自动停止计时器
      // 检查 result.text 是否为非空字符串
      const hasText = result.text && result.text.trim().length > 0;
      // 检查 result.lines 是否包含非空文本
      let hasLines = false;
      if (result.lines && result.lines.length > 0) {
        hasLines = result.lines.some(line => line.text && line.text.trim().length > 0);
      }
      
      if (hasText || hasLines) {
        this.logger.debug('检测到有效转录内容，重置自动停止计时器');
        this._resetAutoStopTimer();
      }
    };

    this.transcriptionClient.oncompletetion = () => {
      this.logger.info('WhisperHotword: 收到 oncompletetion 回调，转录完成');
      console.log('WhisperHotword: transcriptionClient.oncompletetion 被调用');
      this.logger.info('收到 ready_to_stop 信号，转录完成，等待重新监听唤醒词...');
      this._isTranscribing = false;
      this._sendFloat32 = true;
      this._updateStatus('resuming', '检测到语音结束，正在恢复唤醒词监听...');
      
      this._clearAutoStopTimer();
      
      setTimeout(async () => {
        this.logger.info('开始重新监听唤醒词');
        
        if (this.transcriptionClient) {
          try {
            this.transcriptionClient.destroy();
          } catch (e) {
            this.logger.warn('销毁transcriptionClient失败', e);
          }
          this.transcriptionClient = null;
        }
        
        if (this.isListening) {
          try {
            // 重新连接 WebSocket
            this.logger.info('正在重新连接 WebSocket...');
            await this._connectWebSocket();
            await this.configReady;
            this.logger.debug('已收到服务器配置', { useAudioWorklet: this.serverUseAudioWorklet });
            
            // 重置音频捕获
            await this._startAudioCapture();
            // 确保 isListening 仍然是 true
            this.isListening = true;
            this._shouldSendAudio = true;
            this._updateStatus('listening', '正在监听唤醒词...');
            this.logger.info('唤醒词监听已恢复');
          } catch (error) {
            this.logger.error('重新启动唤醒词监听失败', error);
            this._updateStatus('error', `重启监听失败: ${error.message}`);
            if (this.onerror) {
              this.onerror(error);
            }
          }
        } else {
          this.logger.info('用户已停止监听，跳过重新启动');
        }
      }, this.silenceTimeout);
    };

    this.transcriptionClient.onerror = (error) => {
      this._isTranscribing = false;
      this._sendFloat32 = true;
      this._clearAutoStopTimer();
      this.logger.error('转录出错，正在重新启动唤醒词监听...', error);
      this._updateStatus('resuming', '转录出错，正在恢复唤醒词监听...');
      
      setTimeout(async () => {
        if (!this.isListening) {
          this.logger.info('用户已停止监听，跳过重新启动');
          if (this.onerror) {
            this.onerror(error);
          }
          return;
        }
        
        if (this.transcriptionClient) {
          try {
            this.transcriptionClient.destroy();
          } catch (e) {
            this.logger.warn('销毁transcriptionClient失败', e);
          }
          this.transcriptionClient = null;
        }
        
        try {
          // 重新连接 WebSocket
          this.logger.info('正在重新连接 WebSocket...');
          await this._connectWebSocket();
          await this.configReady;
          this.logger.debug('已收到服务器配置', { useAudioWorklet: this.serverUseAudioWorklet });
          
          // 重置音频捕获
          await this._startAudioCapture();
          // 确保 isListening 仍然是 true
          this.isListening = true;
          this._shouldSendAudio = true;
          this._updateStatus('listening', '正在监听唤醒词...');
          this.logger.info('唤醒词监听已恢复（错误恢复）');
        } catch (startError) {
          this.logger.error('重新启动唤醒词监听失败', startError);
          this._updateStatus('error', `重启监听失败: ${startError.message}`);
        }
        
        if (this.onerror) {
          this.onerror(error);
        }
      }, this.silenceTimeout);
    };

    try {
      await this.transcriptionClient.start();
      this.analyser = this.transcriptionClient.getAnalyser();
    } catch (error) {
      this._isTranscribing = false;
      this._sendFloat32 = true;
      this._updateStatus('listening', '正在监听唤醒词...');
      this.logger.error('转录启动失败', error);
    }
  }

  /**
   * 清理资源
   * @private
   */
  _cleanupResources() {
    this._stopAudioCapture();

    if (this.transcriptionClient) {
      try {
        this.transcriptionClient.destroy();
      } catch (e) {}
      this.transcriptionClient = null;
    }

    this._isTranscribing = false;
  }
}

// 全局导出
if (typeof window !== 'undefined') {
  window.WhisperHotword = WhisperHotword;
}

// 模块导出
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WhisperHotword;
}
