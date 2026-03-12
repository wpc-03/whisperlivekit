/**
 * WhisperLiveKit - Logger Module
 * 日志系统模块
 */

const LogLevel = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3
};

const LogLevelNames = ['DEBUG', 'INFO', 'WARN', 'ERROR'];

/**
 * 日志工具类
 */
class Logger {
    /**
     * @param {Object} config - 配置
     * @param {string} [config.level='info'] - 日志级别: debug, info, warn, error
     * @param {Function} [config.handler=null] - 自定义日志处理器
     * @param {string} [config.prefix=''] - 日志前缀
     */
    constructor(config = {}) {
        this.level = this._parseLevel(config.level || 'info');
        this.handler = config.handler || this._defaultHandler;
        this.prefix = config.prefix || '';
    }

    _parseLevel(level) {
        const l = level.toLowerCase();
        if (l === 'debug') return LogLevel.DEBUG;
        if (l === 'info') return LogLevel.INFO;
        if (l === 'warn') return LogLevel.WARN;
        if (l === 'error') return LogLevel.ERROR;
        return LogLevel.INFO;
    }

    _defaultHandler(level, message, data) {
        const timestamp = new Date().toISOString();
        const levelName = LogLevelNames[level];
        const prefix = this.prefix ? `[${this.prefix}] ` : '';
        
        const logMessage = `${timestamp} [${levelName}] ${prefix}${message}`;
        
        if (data) {
            console[level === LogLevel.ERROR ? 'error' : level === LogLevel.WARN ? 'warn' : 'log'](logMessage, data);
        } else {
            console[level === LogLevel.ERROR ? 'error' : level === LogLevel.WARN ? 'warn' : 'log'](logMessage);
        }
    }

    _log(level, message, data) {
        if (level >= this.level) {
            this.handler(level, message, data);
        }
    }

    debug(message, data) {
        this._log(LogLevel.DEBUG, message, data);
    }

    info(message, data) {
        this._log(LogLevel.INFO, message, data);
    }

    warn(message, data) {
        this._log(LogLevel.WARN, message, data);
    }

    error(message, data) {
        this._log(LogLevel.ERROR, message, data);
    }
}

/**
 * Whisper错误类
 */
class WhisperError extends Error {
    /**
     * @param {string} message - 错误消息
     * @param {string} code - 错误代码
     * @param {Object} [details=null] - 详细信息
     */
    constructor(message, code, details = null) {
        super(message);
        this.name = 'WhisperError';
        this.code = code;
        this.details = details;
    }
}

const ErrorCodes = {
    SERVER_UNAVAILABLE: 'E001',
    MICROPHONE_ACCESS_DENIED: 'E002',
    WEBSOCKET_ERROR: 'E003',
    AUDIO_PROCESSING_ERROR: 'E004',
    INVALID_CONFIG: 'E005',
    ALREADY_RECORDING: 'E006',
    NOT_CONNECTED: 'E007',
    UNKNOWN: 'E999'
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Logger, LogLevel, WhisperError, ErrorCodes };
}
