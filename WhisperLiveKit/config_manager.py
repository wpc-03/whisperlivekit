import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 配置文件路径
# 默认使用当前工作目录下的config.json
# 在Docker容器中，工作目录为/app，因此配置文件路径为/app/config.json
# 可以通过环境变量CONFIG_FILE_PATH指定自定义路径
config_file_path = os.environ.get("CONFIG_FILE_PATH")
if config_file_path and config_file_path.strip():
    CONFIG_FILE = Path(config_file_path.strip())
else:
    CONFIG_FILE = Path("config.json")

class ConfigManager:
    """
    配置文件管理类
    用于读取和保存服务配置参数
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认路径
        """
        self.config_file = config_file or CONFIG_FILE
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"加载配置文件: {self.config_file}")
                return config
            else:
                # 返回默认配置
                default_config = self._get_default_config()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            
        Returns:
            True如果保存成功，否则False
        """
        try:
            os.makedirs(self.config_file.parent, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"保存配置文件: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        基于docker-compose.yml中的参数设置
        
        Returns:
            默认配置字典
        """
        return {
            # 服务设置
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "INFO",
            
            # 模型设置
            "model": "tiny",
            "model_path": "/app/models/models--Systran--faster-whisper-tiny",
            "language": "zh",
            "backend_policy": "localagreement",
            "backend": "faster-whisper",
            
            # 音频设置
            "min_chunk_size": 0.15,
            "buffer_trimming_sec": 10,
            "pcm_input": True,
            
            # 识别设置
            "confidence_validation": True,
            "beam_size": 5,
            "keywords_file": "/app/keywords.txt",
            "warmup_file": "/app/samples_zh.wav",
            
            # SSL设置
            "ssl_certfile": "",
            "ssl_keyfile": "",
            
            # 说话人识别设置
            "diarization": False,
            "diarization_model": "",
            "punctuation_split": False,
            
            # 唤醒词设置
            "hotword_model_dir": "",
            "hotword_threshold": 0.6,
            "hotword_sample_rate": 16000,
            "hotword_threads": 4,
            
            # 其他设置
            "forwarded_allow_ips": ""
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        Returns:
            配置字典
        """
        return self.config.copy()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        获取单个配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        return self.config.get(key, default)
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新配置
        
        Args:
            new_config: 新配置字典
            
        Returns:
            True如果更新成功，否则False
        """
        try:
            # 验证配置
            validated_config = self._validate_config(new_config)
            
            # 更新配置
            self.config.update(validated_config)
            
            # 保存到文件
            return self._save_config(self.config)
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证配置
        
        Args:
            config: 要验证的配置
            
        Returns:
            验证后的配置
        """
        validated = {}
        
        # 验证主机地址
        if "host" in config:
            validated["host"] = str(config["host"])
        
        # 验证端口
        if "port" in config:
            port = int(config["port"])
            if 1024 <= port <= 65535:
                validated["port"] = port
        
        # 验证日志级别
        if "log_level" in config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if config["log_level"].upper() in valid_levels:
                validated["log_level"] = config["log_level"].upper()
        
        # 验证模型类型
        if "model" in config:
            valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
            if config["model"] in valid_models:
                validated["model"] = config["model"]
        
        # 验证模型路径
        if "model_path" in config:
            validated["model_path"] = str(config["model_path"])
        
        # 验证语言
        if "language" in config:
            validated["language"] = str(config["language"])
        
        # 验证后端策略
        if "backend_policy" in config:
            validated["backend_policy"] = str(config["backend_policy"])
        
        # 验证后端
        if "backend" in config:
            valid_backends = ["auto", "mlx-whisper", "faster-whisper", "whisper", "openai-api"]
            if config["backend"] in valid_backends:
                validated["backend"] = config["backend"]
        
        # 验证最小音频块大小
        if "min_chunk_size" in config:
            min_chunk = float(config["min_chunk_size"])
            if min_chunk > 0:
                validated["min_chunk_size"] = min_chunk
        
        # 验证缓冲区修剪阈值
        if "buffer_trimming_sec" in config:
            buffer_sec = float(config["buffer_trimming_sec"])
            if buffer_sec > 0:
                validated["buffer_trimming_sec"] = buffer_sec
        
        # 验证beam size
        if "beam_size" in config:
            beam = int(config["beam_size"])
            if beam > 0:
                validated["beam_size"] = beam
        
        # 验证关键词文件
        if "keywords_file" in config:
            validated["keywords_file"] = str(config["keywords_file"])
        
        # 验证预热文件
        if "warmup_file" in config:
            validated["warmup_file"] = str(config["warmup_file"])
        
        # 验证SSL文件路径
        if "ssl_certfile" in config:
            validated["ssl_certfile"] = str(config["ssl_certfile"])
        
        if "ssl_keyfile" in config:
            validated["ssl_keyfile"] = str(config["ssl_keyfile"])
        
        # 验证说话人识别模型
        if "diarization_model" in config:
            validated["diarization_model"] = str(config["diarization_model"])
        
        # 验证唤醒词模型目录
        if "hotword_model_dir" in config:
            validated["hotword_model_dir"] = str(config["hotword_model_dir"])
        
        # 验证唤醒词阈值
        if "hotword_threshold" in config:
            hotword_thresh = float(config["hotword_threshold"])
            if 0.0 <= hotword_thresh <= 1.0:
                validated["hotword_threshold"] = hotword_thresh
        
        # 验证唤醒词采样率
        if "hotword_sample_rate" in config:
            rate = int(config["hotword_sample_rate"])
            if rate > 0:
                validated["hotword_sample_rate"] = rate
        
        # 验证唤醒词线程数
        if "hotword_threads" in config:
            threads = int(config["hotword_threads"])
            if threads > 0:
                validated["hotword_threads"] = threads
        
        # 验证转发允许的IP
        if "forwarded_allow_ips" in config:
            validated["forwarded_allow_ips"] = str(config["forwarded_allow_ips"])
        
        # 验证布尔值设置
        bool_settings = ["pcm_input", "confidence_validation", "diarization", "punctuation_split"]
        for setting in bool_settings:
            if setting in config:
                validated[setting] = bool(config[setting])
        
        return validated
    
    def reset_config(self) -> bool:
        """
        重置配置为默认值
        
        Returns:
            True如果重置成功，否则False
        """
        try:
            default_config = self._get_default_config()
            self.config = default_config
            return self._save_config(self.config)
        except Exception as e:
            logger.error(f"重置配置失败: {e}")
            return False
    
    def get_default_config_dict(self) -> Dict[str, Any]:
        """
        获取默认配置字典
        
        Returns:
            默认配置字典
        """
        return self._get_default_config()

# 创建全局配置管理器实例
config_manager = ConfigManager()