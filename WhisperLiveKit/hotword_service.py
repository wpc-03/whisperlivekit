"""
Hotword detection service for WebSocket endpoints
专门用于WebSocket唤醒词检测的服务类
"""

import logging
import os
import time
import threading
from typing import Optional, Dict, Any
import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)

class HotwordService:
    """
    唤醒词检测服务
    处理WebSocket连接中的音频数据，检测唤醒词
    """
    
    def __init__(self, 
                 model_dir: str,
                 keywords_file: Optional[str] = None,
                 threshold: float = 0.6,
                 sample_rate: int = 16000,
                 num_threads: int = 4):
        """
        初始化唤醒词检测服务
        
        Args:
            model_dir: 模型目录路径
            keywords_file: 关键词文件路径，如果为None则使用model_dir/keywords.txt
            threshold: 检测阈值
            sample_rate: 音频采样率
            num_threads: 推理线程数
        """
        self.model_dir = model_dir
        self.keywords_file = keywords_file or os.path.join(model_dir, "keywords.txt")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        
        # 检查必要文件是否存在
        required_files = [
            os.path.join(model_dir, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            os.path.join(model_dir, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            os.path.join(model_dir, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
            os.path.join(model_dir, "tokens.txt"),
            self.keywords_file
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
        
        # KWS模型和流
        self.kws: Optional[sherpa_onnx.KeywordSpotter] = None
        self.kws_streams: Dict[str, Any] = {}  # 每个WebSocket连接一个流
        
        # 文件监控相关
        self.keywords_file_mtime = 0
        self.file_monitor_thread: Optional[threading.Thread] = None
        self.is_running = False
        self._update_file_mtime()
        
        # 初始化模型
        self._initialize_model()
        
        # 启动文件监控线程
        self.is_running = True
        self.file_monitor_thread = threading.Thread(target=self._monitor_keywords_file, daemon=True)
        self.file_monitor_thread.start()
    
    def _initialize_model(self) -> bool:
        """初始化KWS模型"""
        try:
            logger.info(f"初始化KWS模型，目录: {self.model_dir}")
            logger.info(f"关键词文件: {self.keywords_file}")
            
            # 直接创建KeywordSpotter实例（sherpa-onnx 1.12.28 API）
            self.kws = sherpa_onnx.KeywordSpotter(
                tokens=os.path.join(self.model_dir, "tokens.txt"),
                encoder=os.path.join(
                    self.model_dir, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                ),
                decoder=os.path.join(
                    self.model_dir, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                ),
                joiner=os.path.join(
                    self.model_dir, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
                ),
                num_threads=self.num_threads,
                keywords_file=self.keywords_file,
                provider="cpu",
            )
            
            logger.info(f"KWS模型初始化成功，阈值: {self.threshold}, 采样率: {self.sample_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"KWS模型初始化失败: {e}")
            self.kws = None
            return False
    
    def create_stream(self, connection_id: str) -> bool:
        """
        为WebSocket连接创建新的KWS流
        
        Args:
            connection_id: 连接标识符
            
        Returns:
            是否成功创建
        """
        if not self.kws:
            logger.error("KWS模型未初始化，无法创建流")
            return False
        
        try:
            self.kws_streams[connection_id] = self.kws.create_stream()
            logger.debug(f"为连接 {connection_id} 创建KWS流")
            return True
        except Exception as e:
            logger.error(f"创建KWS流失败: {e}")
            return False
    
    def delete_stream(self, connection_id: str) -> None:
        """删除连接的KWS流"""
        if connection_id in self.kws_streams:
            del self.kws_streams[connection_id]
            logger.debug(f"删除连接 {connection_id} 的KWS流")
    
    def process_audio(self, connection_id: str, audio_data: bytes) -> Optional[str]:
        """
        处理音频数据，检测唤醒词
        
        Args:
            connection_id: 连接标识符
            audio_data: PCM音频数据 (float32格式)
            
        Returns:
            检测到的唤醒词文本，如果没有检测到则返回None
        """
        if not self.kws or connection_id not in self.kws_streams:
            logger.error(f"连接 {connection_id} 的KWS流不存在")
            return None
        
        try:
            data_len = len(audio_data)
            logger.info(f"处理音频: {data_len} bytes")
            
            if connection_id not in self.kws_streams:
                logger.warning(f"KWS流不存在，重新创建")
                if not self.create_stream(connection_id):
                    return None
            
            pcm_float32 = np.frombuffer(audio_data, dtype=np.float32)
            
            kws_stream = self.kws_streams[connection_id]
            
            # 处理音频数据
            kws_stream.accept_waveform(self.sample_rate, pcm_float32)
            
            # 检查是否有检测结果
            while self.kws.is_ready(kws_stream):
                self.kws.decode_stream(kws_stream)
                result = self.kws.get_result(kws_stream)
                
                if result:
                    wakeword = str(result).strip()
                    logger.info(f"🔊 检测到唤醒词: {wakeword} (连接: {connection_id})")
                    
                    # 重置流，准备下一次检测
                    self.kws.reset_stream(kws_stream)
                    
                    return wakeword
            
            return None
            
        except Exception as e:
            logger.error(f"处理音频数据失败: {e}")
            return None
    
    def _update_file_mtime(self) -> None:
        """更新关键词文件的最后修改时间"""
        try:
            if os.path.exists(self.keywords_file):
                self.keywords_file_mtime = os.path.getmtime(self.keywords_file)
                logger.info(f"更新关键词文件修改时间: {self.keywords_file_mtime}")
        except Exception as e:
            logger.error(f"更新文件修改时间失败: {e}")
    
    def _monitor_keywords_file(self) -> None:
        """监控关键词文件的变化"""
        logger.info("启动关键词文件监控线程")
        
        while self.is_running:
            try:
                if os.path.exists(self.keywords_file):
                    current_mtime = os.path.getmtime(self.keywords_file)
                    if current_mtime > self.keywords_file_mtime:
                        logger.info(f"检测到关键词文件变化，重新加载KWS模型")
                        self._reload_model()
                        self._update_file_mtime()
                time.sleep(2)  # 每2秒检查一次
            except Exception as e:
                logger.error(f"文件监控异常: {e}")
                time.sleep(2)
        
        logger.info("关键词文件监控线程结束")
    
    def _reload_model(self) -> bool:
        """重新加载KWS模型"""
        try:
            logger.info("正在重新加载KWS模型...")
            
            # 停止当前的KWS实例
            if self.kws:
                del self.kws
                self.kws = None
                # 清空所有流
                self.kws_streams.clear()
                logger.info("已停止当前KWS实例和所有流")
            
            # 重新初始化KWS模型
            success = self._initialize_model()
            
            if success:
                logger.info("KWS模型重新加载成功")
            else:
                logger.error("KWS模型重新加载失败")
            
            return success
        except Exception as e:
            logger.error(f"重新加载KWS模型失败: {e}")
            return False
    
    def stop(self) -> None:
        """停止服务"""
        logger.info("正在停止唤醒词检测服务...")
        
        self.is_running = False
        
        # 清空所有流
        self.kws_streams.clear()
        
        # 停止当前的KWS实例
        if self.kws:
            del self.kws
            self.kws = None
        
        logger.info("唤醒词检测服务已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "model_dir": self.model_dir,
            "keywords_file": self.keywords_file,
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "num_threads": self.num_threads,
            "active_streams": len(self.kws_streams),
            "is_model_loaded": self.kws is not None
        }
