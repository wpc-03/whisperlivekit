import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 设置文件路径
SETTINGS_FILE = Path(__file__).parent / "data" / "settings.json"


class SettingsManager:
    """
    参数设置管理类
    """
    
    def __init__(self):
        """
        初始化设置管理器
        """
        self.settings_file = SETTINGS_FILE
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """
        加载设置文件
        
        Returns:
            设置字典
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                logger.info(f"Loaded settings from {self.settings_file}")
                return settings
            else:
                # 返回默认设置
                default_settings = self._get_default_settings()
                self._save_settings(default_settings)
                return default_settings
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return self._get_default_settings()
    
    def _save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        保存设置到文件
        
        Args:
            settings: 设置字典
            
        Returns:
            True如果保存成功，否则False
        """
        try:
            os.makedirs(self.settings_file.parent, exist_ok=True)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved settings to {self.settings_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """
        获取默认设置
        
        Returns:
            默认设置字典
        """
        return {
            "backend_policy": "localagreement",  # 后端策略（固定值，不允许修改）
            "model": "small",  # Whisper模型类型
            "model_path": r"D:\python\FastWhisperTranscriber\model\models--Systran--faster-whisper-small",  # 模型路径
            "language": "zh",  # 语言设置
            "backend": "faster-whisper",  # 后端
            "min_chunk_size": 0.15,  # 最小音频块大小
            "buffer_trimming_sec": 15,  # 缓冲区修剪阈值
            "confidence_validation": True,  # 置信度验证
            "beam_size": 3,  # beam size
            "keywords_file": r"d:\python\whisperlivekit\keywords_example.txt",  # 关键词文件
            "warmup_file": r"D:\python\whisperlivekit\samples_zh.wav",  # 预热文件
            "pcm_input": True,  # PCM输入
            "diarization": False,  # 说话人识别
            "diarization_model": "",  # 说话人识别模型
            "punctuation_split": False,  # 标点分割
            "hotword_model_dir": "",  # 唤醒词模型目录
            "hotword_threshold": 0.6,  # 唤醒词阈值
            "hotword_sample_rate": 16000,  # 唤醒词采样率
            "hotword_threads": 4  # 唤醒词线程数
            }
    
    def get_settings(self) -> Dict[str, Any]:
        """
        获取所有设置
        
        Returns:
            设置字典
        """
        return self.settings.copy()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        获取单个设置
        
        Args:
            key: 设置键
            default: 默认值
            
        Returns:
            设置值或默认值
        """
        return self.settings.get(key, default)
    
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        更新设置
        
        Args:
            new_settings: 新设置字典
            
        Returns:
            True如果更新成功，否则False
        """
        try:
            # 验证设置
            validated_settings = self._validate_settings(new_settings)
            
            # 更新设置
            self.settings.update(validated_settings)
            
            # 保存到文件
            return self._save_settings(self.settings)
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            return False
    
    def _validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证设置
        
        Args:
            settings: 要验证的设置
            
        Returns:
            验证后的设置
        """
        validated = {}
        
        # 后端策略固定为localagreement，不允许修改
        validated["backend_policy"] = "localagreement"
        
        # 验证模型类型
        if "model" in settings:
            valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
            if settings["model"] in valid_models:
                validated["model"] = settings["model"]
        
        # 验证模型路径
        if "model_path" in settings:
            validated["model_path"] = str(settings["model_path"])
        
        # 验证语言
        if "language" in settings:
            # 允许auto或具体语言代码
            validated["language"] = settings["language"]
        
        # 验证后端
        if "backend" in settings:
            valid_backends = ["auto", "mlx-whisper", "faster-whisper", "whisper", "openai-api"]
            if settings["backend"] in valid_backends:
                validated["backend"] = settings["backend"]
        
        # 验证最小音频块大小
        if "min_chunk_size" in settings:
            min_chunk = float(settings["min_chunk_size"])
            if min_chunk > 0:
                validated["min_chunk_size"] = min_chunk
        
        # 验证缓冲区修剪阈值
        if "buffer_trimming_sec" in settings:
            buffer_sec = float(settings["buffer_trimming_sec"])
            if buffer_sec > 0:
                validated["buffer_trimming_sec"] = buffer_sec
        
        # 验证beam size
        if "beam_size" in settings:
            beam = int(settings["beam_size"])
            if beam > 0:
                validated["beam_size"] = beam
        
        # 验证关键词文件
        if "keywords_file" in settings:
            validated["keywords_file"] = str(settings["keywords_file"])
        
        # 验证预热文件
        if "warmup_file" in settings:
            validated["warmup_file"] = str(settings["warmup_file"])
        
        # 验证说话人识别模型
        if "diarization_model" in settings:
            validated["diarization_model"] = str(settings["diarization_model"])
        
        # 验证唤醒词模型目录
        if "hotword_model_dir" in settings:
            validated["hotword_model_dir"] = str(settings["hotword_model_dir"])
        
        # 验证唤醒词阈值
        if "hotword_threshold" in settings:
            hotword_thresh = float(settings["hotword_threshold"])
            if 0.0 <= hotword_thresh <= 1.0:
                validated["hotword_threshold"] = hotword_thresh
        
        # 验证唤醒词采样率
        if "hotword_sample_rate" in settings:
            rate = int(settings["hotword_sample_rate"])
            if rate > 0:
                validated["hotword_sample_rate"] = rate
        
        # 验证唤醒词线程数
        if "hotword_threads" in settings:
            threads = int(settings["hotword_threads"])
            if threads > 0:
                validated["hotword_threads"] = threads
        
        # 验证布尔值设置
        bool_settings = ["confidence_validation", "pcm_input", "diarization", "punctuation_split"]
        for setting in bool_settings:
            if setting in settings:
                validated[setting] = bool(settings[setting])
        
        return validated
    
    def reset_settings(self) -> bool:
        """
        重置设置为默认值
        
        Returns:
            True如果重置成功，否则False
        """
        try:
            default_settings = self._get_default_settings()
            self.settings = default_settings
            return self._save_settings(self.settings)
        except Exception as e:
            logger.error(f"Failed to reset settings: {e}")
            return False
    
    def get_default_settings(self) -> Dict[str, Any]:
        """
        获取默认设置
        
        Returns:
            默认设置字典
        """
        return self._get_default_settings()


# 创建全局设置管理器实例
settings_manager = SettingsManager()