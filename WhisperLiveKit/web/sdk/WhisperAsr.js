/**
 * TranscriptRenderer - 转录结果实时渲染器
 * 
 * @description 用于实时渲染转录结果，支持已确认文本和实时中间结果的动态显示
 * 
 * @example
 * const renderer = new TranscriptRenderer({
 *   containerId: 'transcript',
 *   className: 'transcript-line',
 *   confirmedClass: 'confirmed',
 *   unconfirmedClass: 'unconfirmed'
 * });
 * 
 * client.onresult = (result) => {
 *   renderer.render(result);
 * };
 */

class TranscriptRenderer {
  /**
   * 创建转录渲染器
   * @param {Object} options - 配置选项
   * @param {string} options.containerId - 容器元素ID
   * @param {string} [options.className='transcript-line'] - 转录行容器类名
   * @param {string} [options.confirmedClass='confirmed'] - 已确认文本类名
   * @param {string} [options.unconfirmedClass='buffer_transcription'] - 未确认文本类名
   * @param {string} [options.waitingText='转录结果将显示在这里...'] - 等待时显示的文本
   * @param {boolean} [options.autoScroll=true] - 是否自动滚动到底部
   */
  constructor(options = {}) {
    this.container = document.getElementById(options.containerId);
    if (!this.container) {
      throw new Error(`TranscriptRenderer: 容器元素 #${options.containerId} 不存在`);
    }

    this.className = options.className || 'transcript-line';
    this.confirmedClass = options.confirmedClass || 'confirmed';
    this.unconfirmedClass = options.unconfirmedClass || 'buffer_transcription';
    this.waitingText = options.waitingText || '转录结果将显示在这里...';
    this.autoScroll = options.autoScroll !== false;

    this.lastSignature = null;
  }

  /**
   * HTML转义特殊字符
   * @private
   */
  _escapeHtml(text) {
    if (!text) return '';
    const escapeMap = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    };
    return text.replace(/[&<>"']/g, (m) => escapeMap[m]);
  }

  /**
   * 创建签名，用于避免不必要的重绘
   * @private
   */
  _createSignature(result) {
    return JSON.stringify({
      text: result.text || '',
      lines: result.lines || [],
      buffer: result.buffer_transcription || '',
      status: result.status || 'active_transcription'
    });
  }

  /**
   * 渲染转录结果
   * @param {Object} result - 转录结果对象
   */
  render(result) {
    if (!result) {
      this._showWaiting();
      return;
    }

    const signature = this._createSignature(result);
    if (signature === this.lastSignature) {
      return;
    }
    this.lastSignature = signature;

    const lines = result.lines || [];
    const bufferText = result.buffer_transcription || result.text || '';

    if ((!lines || lines.length === 0) && !bufferText) {
      this._showWaiting();
      return;
    }

    let html = `<div class="${this.className}">`;

    let hasConfirmedText = false;
    let lastConfirmedIndex = -1;
    lines.forEach((line, index) => {
      if (line.text && line.text.trim()) {
        hasConfirmedText = true;
        lastConfirmedIndex = index;
      }
    });

    lines.forEach((line, index) => {
      if (line.text && line.text.trim()) {
        html += `<span class="${this.confirmedClass}">`;
        html += this._escapeHtml(line.text);
        html += `</span>`;

        if (index < lastConfirmedIndex) {
          html += `<br>`;
        }
      }
    });

    if (bufferText) {
      html += `<span class="${this.unconfirmedClass}">${this._escapeHtml(bufferText)}</span>`;
    }

    html += `</div>`;

    this.container.innerHTML = html;

    if (this.autoScroll) {
      this.container.scrollTop = this.container.scrollHeight;
    }
  }

  /**
   * 显示等待提示
   * @private
   */
  _showWaiting() {
    this.container.innerHTML = `<p style="color: #999; text-align: center;">${this.waitingText}</p>`;
    this.lastSignature = null;
  }

  /**
   * 清空显示
   */
  clear() {
    this.lastSignature = null;
    this._showWaiting();
  }
}

/**
 * WhisperAsr - 按钮模式语音转录客户端
 * 
 * @description 用于按钮模式的语音转录功能，支持实时语音识别和转录
 * 
 * @example
 * const client = new WhisperAsr({
 *   serverUrl: 'ws://localhost:8000',
 *   language: 'zh'
 * });
 * 
 * client.onresult = (result) => {
 *   console.log('转录结果:', result.text);
 * };
 * 
 * await client.start();
 */

class WhisperAsr {
  /**
   * 创建按钮转录客户端
   * @param {Object} options - 配置选项
   * @param {string} [options.serverUrl='ws://localhost:8000'] - WebSocket服务器地址
   * @param {string} [options.language=null] - 语言代码，如 'zh', 'en'，默认自动检测
   * @param {string} [options.microphoneId=null] - 麦克风设备ID
   * @param {boolean} [options.autoStart=true] - 连接后自动开始录音
   * @param {string} [options.logLevel='info'] - 日志级别: debug, info, warn, error
   * @param {Function} [options.logHandler=null] - 自定义日志处理器
   */
  constructor(options = {}) {
    // 日志系统
    this.logger = new Logger({
      level: options.logLevel || 'info',
      handler: options.logHandler,
      prefix: 'WhisperAsr'
    });

    // 配置选项
    this.serverUrl = options.serverUrl || this._detectServerUrl();
    this.language = options.language || null;
    this.microphoneId = options.microphoneId || null;
    this.autoStart = options.autoStart !== false;

    // 状态变量
    this.isRecording = false;
    this.isConnected = false;
    this.isProcessing = false;
    this.currentStatus = 'idle';

    // WebSocket和音频资源
    this.websocket = null;
    this.audioContext = null;
    this.analyser = null;
    this.microphone = null;
    this.workletNode = null;
    this.recorder = null;
    this.recorderWorker = null;

    // 计时相关
    this.startTime = null;
    this.timerInterval = null;
    this.lastReceivedData = null;

    // 配置相关
    this.configReadyResolve = null;
    this.configReady = new Promise((resolve) => {
      this.configReadyResolve = resolve;
    });

    // 服务器配置，由服务器在config消息中指定
    this.serverUseAudioWorklet = null;

    // 回调函数
    this.onresult = null;
    this.onpartial = null;
    this.onerror = null;
    this.onconnect = null;
    this.ondisconnect = null;
    this.onready = null;
    this.oncompletetion = null;
    this.onstatuschange = null;

    // 内部状态
    this._audioSource = null;
    this._shouldSendAudio = false;
    this._userClosing = false;
    this._waitingForStop = false;

    this.logger.info('WhisperAsr 初始化完成', { serverUrl: this.serverUrl });
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
   * 开始转录
   * @returns {Promise<void>}
   * @throws {WhisperError} 如果已经在录音中或处于错误状态
   */
  async start() {
    if (this.isRecording) {
      const error = new WhisperError('已经在录音中', ErrorCodes.ALREADY_RECORDING);
      this.logger.error(error.message);
      throw error;
    }

    if (this.currentStatus === 'error') {
      const error = new WhisperError('客户端处于错误状态，请重新初始化', ErrorCodes.INVALID_CONFIG);
      this.logger.error(error.message);
      throw error;
    }

    try {
      this.logger.info('正在启动转录...');
      this._updateStatus('connecting', '正在连接服务器...');

      await this._connectWebSocket();
      await this.configReady;

      this._updateStatus('ready', '服务器准备就绪');
      await this._startRecording();

      this._updateStatus('recording', '正在录音...');
      this.isRecording = true;
      
      this.logger.info('转录已启动');

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
   * 停止转录
   * @returns {void}
   */
  stop() {
    if (!this.isRecording) {
      return;
    }

    this.logger.info('正在停止转录...');
    this._updateStatus('processing', '正在处理音频...');
    this.isRecording = false;
    this._shouldSendAudio = false;
    this._userClosing = true;
    this._waitingForStop = true;

    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      const emptyBlob = new Blob([], { type: 'audio/webm' });
      this.websocket.send(emptyBlob);
    }

    this._stopRecording();
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
   * 获取音频分析器（用于波形可视化）
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
      // 确保使用 ws 协议
      const wsBaseUrl = this.serverUrl.replace(/^http/, 'ws');
      const wsUrl = `${wsBaseUrl}/asr`;
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
        this._userClosing = false;
        this._waitingForStop = false;
        this._cleanupResources();
        this.logger.info('WebSocket已断开');
        if (this.ondisconnect) this.ondisconnect();
        this._updateStatus('idle', '已断开连接');
      };

      this.websocket.onerror = () => {
        const err = new WhisperError('WebSocket连接错误', ErrorCodes.WEBSOCKET_ERROR);
        this.logger.error(err.message);
        reject(err);
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

      if (data.type === 'ready_to_stop') {
        console.log('WhisperAsr 收到 ready_to_stop！', data);
        this._waitingForStop = false;
        this._userClosing = false;
        this._cleanupResources();
        if (this.websocket) {
          this.websocket.close();
        }
        this.logger.info('转录完成');
        console.log('WhisperAsr oncompletetion=', this.oncompletetion, 'this=', this);
        if (this.oncompletetion) {
          console.log('调用 oncompletetion...');
          this.oncompletetion(data);
        } else {
          console.warn('WhisperAsr.oncompletetion 回调未设置！');
        }
        this._updateStatus('idle', '转录完成');
        return;
      }

      this.logger.debug('收到原始转录数据:', data);
      this.lastReceivedData = data;
      const result = this._extractTranscriptionResult(data);

      if (result.text || (result.lines && result.lines.length > 0)) {
        this.logger.debug('转录结果:', {
          text: result.text,
          lines: result.lines,
          isPartial: result.isPartial,
          buffer_transcription: result.buffer_transcription
        });
      }

      if (result.isPartial && this.onpartial) {
        this.onpartial(result);
      }

      if (this.onresult) {
        this.onresult(result);
      }

    } catch (error) {
      this.logger.error('处理WebSocket消息失败', error);
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
      status = 'active_transcription',
      detected_language = null
    } = data || {};
    
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
    
    // 处理lines数组
    const processedLines = lines.map(line => this._processLine(line));
    
    return {
      text,
      lines: processedLines,
      buffer_transcription,
      isPartial: !!buffer_transcription, // 有缓冲区转录就是中间结果
      speaker: buffer_diarization || null,
      language: detected_language,
      status,
      remainingTime: 0 // 为了兼容性
    };
  }

  /**
   * 处理单行转录数据
   * @private
   */
  _processLine(line) {
    const processedLine = {};
    
    if (line.text !== undefined) {
      processedLine.text = line.text;
    }
    
    if (line.start !== undefined) {
      processedLine.start = parseFloat(line.start);
      if (isNaN(processedLine.start)) {
        processedLine.start = line.start;
      }
    }
    
    if (line.end !== undefined) {
      processedLine.end = parseFloat(line.end);
      if (isNaN(processedLine.end)) {
        processedLine.end = line.end;
      }
    }
    
    if (line.speaker !== undefined) {
      const speakerNum = parseFloat(line.speaker);
      if (!isNaN(speakerNum)) {
        processedLine.speaker = speakerNum;
      }
    }
    
    if (line.detected_language !== undefined) {
      processedLine.detected_language = line.detected_language;
    }
    
    return processedLine;
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
   * 开始录音
   * @private
   */
  async _startRecording() {
    try {
      const constraints = {
        audio: {
          deviceId: this.microphoneId ? { exact: this.microphoneId } : undefined,
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      };

      this.logger.debug('请求麦克风权限', constraints);
      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this._audioSource = this.audioContext.createMediaStreamSource(stream);
      this._audioSource.connect(this.analyser);

      // 根据服务器配置决定使用AudioWorklet还是MediaRecorder
      // 如果服务器需要PCM输入，使用AudioWorklet；否则使用MediaRecorder
      // 如果服务器未指定配置，检测浏览器是否支持AudioWorklet作为后备
      const useWorklet = this.serverUseAudioWorklet ?? this._detectAudioWorkletSupport();
      
      if (useWorklet) {
        try {
          await this._setupAudioWorklet();
        } catch (error) {
          this.logger.warn(`AudioWorklet设置失败，降级到MediaRecorder: ${error.message}`);
          this._setupMediaRecorder(stream);
        }
      } else {
        this._setupMediaRecorder(stream);
      }

      this._shouldSendAudio = true;
      this.startTime = Date.now();

      this.logger.info('录音已启动');

    } catch (error) {
      const err = new WhisperError(`启动录音失败: ${error.message}`, ErrorCodes.AUDIO_PROCESSING_ERROR);
      this.logger.error(err.message, error);
      throw err;
    }
  }

  /**
   * 设置AudioWorklet
   * @private
   */
  async _setupAudioWorklet() {
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

    // worker 处理完成的音频数据，发送给服务器
    this.recorderWorker.onmessage = (e) => {
      if (this._shouldSendAudio && this.websocket?.readyState === WebSocket.OPEN) {
        this.websocket.send(e.data.buffer);
      }
    };

    // AudioWorklet 收到音频后发送给 worker 处理
    this.workletNode.port.onmessage = (e) => {
      const data = e.data;
      const ab = data instanceof ArrayBuffer ? data : data.buffer;
      this.recorderWorker.postMessage({
        command: 'record',
        buffer: ab
      }, [ab]);
    };

    this._audioSource.connect(this.workletNode);
    
    this.logger.debug('AudioWorklet已设置');
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
   * 停止录音
   * @private
   */
  _stopRecording() {
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

    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }

    this.logger.info('录音已停止');
  }

  /**
   * 清理资源
   * @private
   */
  _cleanupResources() {
    this._stopRecording();
  }

  /**
   * 销毁客户端
   */
  destroy() {
    this.logger.info('正在销毁客户端');
    this._cleanupResources();
    this._updateStatus('idle', '客户端已销毁');
  }
}

// 全局导出
if (typeof window !== 'undefined') {
  window.WhisperAsr = WhisperAsr;
  window.WhisperButton = WhisperAsr; // 兼容旧版本
  window.TranscriptRenderer = TranscriptRenderer;
}

// 模块导出
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WhisperAsr;
}
