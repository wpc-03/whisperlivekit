import logging
import sys
import time
import threading
import queue
import os
from enum import Enum
from typing import Optional, Callable, List, Dict, Any
import numpy as np
import sherpa_onnx
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SystemState(Enum):
    """系统状态枚举"""
    WAITING = "waiting"          # 等待唤醒词状态
    WAKEWORD_DETECTED = "wakeword_detected"  # 唤醒词检测成功
    ASR_ACTIVE = "asr_active"    # ASR识别进行中
    ASR_FINISHING = "asr_finishing"  # ASR正在结束
    ERROR = "error"              # 错误状态

class HotwordDetector:
    """
    唤醒词检测器主类
    实现完整的唤醒词检测功能，包括麦克风监听、KWS检测、状态管理
    """
    
    def __init__(self, 
                 model_dir: str,
                 keywords_file: str,
                 threshold: float = 0.6,
                 sample_rate: int = 16000,
                 num_threads: int = 4):
        """
        初始化唤醒词检测器
        
        Args:
            model_dir: 模型目录路径
            keywords_file: 关键词文件路径
            threshold: 唤醒词检测阈值 (0.0-1.0)
            sample_rate: 音频采样率
            num_threads: 推理线程数
        """
        self.model_dir = model_dir
        self.keywords_file = keywords_file
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        
        # 系统状态
        self.state = SystemState.WAITING
        self.state_lock = threading.Lock()
        
        # 音频处理
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.audio_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        # KWS模型
        self.kws: Optional[sherpa_onnx.Kws] = None
        self.kws_stream: Optional[Any] = None
        
        # 回调函数
        self.on_wakeword_detected: Optional[Callable[[str], None]] = None
        self.on_asr_start: Optional[Callable[[], None]] = None
        self.on_asr_result: Optional[Callable[[str], None]] = None
        self.on_asr_finished: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # 统计信息
        self.stats = {
            "wakeword_detections": 0,
            "asr_sessions": 0,
            "total_audio_seconds": 0,
            "last_wakeword_time": 0
        }
        
        # ASR相关（后续集成）
        self.asr_active = False
        self.asr_silence_timeout = 3.0  # ASR静音超时（秒）
        self.last_audio_time = 0
        
        # 文件监控相关
        self.keywords_file_mtime = 0
        self.file_monitor_thread: Optional[threading.Thread] = None
        self._update_file_mtime()
        
        logger.info(f"唤醒词检测器初始化完成，模型目录: {model_dir}")
    
    def _initialize_kws(self) -> bool:
        """初始化KWS模型"""
        try:
            logger.info("正在初始化KWS模型...")
            
            # 构建配置文件
            config = sherpa_onnx.KwsConfig()
            
            # 设置模型文件路径
            config.model_config.encoder_filename = f"{self.model_dir}/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
            config.model_config.decoder_filename = f"{self.model_dir}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
            config.model_config.joiner_filename = f"{self.model_dir}/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
            config.model_config.tokens = f"{self.model_dir}/tokens.txt"
            config.model_config.num_threads = self.num_threads
            
            # 设置KWS配置
            config.kws_conf.keywords_file = self.keywords_file
            config.kws_conf.threshold = self.threshold
            config.kws_conf.min_keywords = 1
            
            # 设置采样率
            config.sample_rate = self.sample_rate
            
            # 创建KWS实例
            self.kws = sherpa_onnx.Kws(config)
            self.kws_stream = self.kws.create_stream()
            
            logger.info(f"KWS模型初始化成功，阈值: {self.threshold}")
            return True
            
        except Exception as e:
            logger.error(f"KWS模型初始化失败: {e}")
            self._set_state(SystemState.ERROR)
            if self.on_error:
                self.on_error(f"KWS模型初始化失败: {e}")
            return False
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """
        音频回调函数，从麦克风接收音频数据
        """
        if status:
            logger.warning(f"音频设备状态: {status}")
        
        try:
            # 转换为单声道浮点格式
            samples = indata[:, 0].astype(np.float32)
            
            # 将音频数据放入队列供处理线程使用
            if not self.audio_queue.full():
                self.audio_queue.put(samples)
            
            # 更新最后音频时间（用于ASR超时检测）
            self.last_audio_time = time.time()
            
        except Exception as e:
            logger.error(f"音频回调处理失败: {e}")
    
    def _process_audio(self) -> None:
        """音频处理线程主函数"""
        logger.info("音频处理线程启动")
        
        while self.is_running:
            try:
                # 从队列获取音频数据（非阻塞，避免线程卡死）
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 根据当前状态处理音频
                with self.state_lock:
                    current_state = self.state
                
                if current_state == SystemState.WAITING:
                    # 等待状态：进行唤醒词检测
                    self._detect_wakeword(audio_data)
                    
                elif current_state == SystemState.WAKEWORD_DETECTED:
                    # 唤醒词检测成功，准备启动ASR
                    self._start_asr()
                    
                elif current_state == SystemState.ASR_ACTIVE:
                    # ASR激活状态：处理ASR音频
                    self._process_asr_audio(audio_data)
                    
                elif current_state == SystemState.ASR_FINISHING:
                    # ASR正在结束：等待清理
                    pass
                    
                elif current_state == SystemState.ERROR:
                    # 错误状态：暂停处理
                    time.sleep(0.1)
                
                # 更新统计信息
                self.stats["total_audio_seconds"] += len(audio_data) / self.sample_rate
                
            except Exception as e:
                logger.error(f"音频处理异常: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("音频处理线程结束")
    
    def _detect_wakeword(self, audio_data: np.ndarray) -> None:
        """检测唤醒词"""
        if not self.kws or not self.kws_stream:
            return
        
        try:
            # 将音频数据送入KWS流
            self.kws_stream.accept_waveform(audio_data)
            
            # 检查是否有检测结果
            while self.kws_stream.is_ready():
                result = self.kws_stream.decode()
                if result:
                    wakeword = str(result).strip()
                    logger.info(f"🔊 检测到唤醒词: {wakeword}")
                    
                    # 更新统计
                    self.stats["wakeword_detections"] += 1
                    self.stats["last_wakeword_time"] = time.time()
                    
                    # 触发回调
                    if self.on_wakeword_detected:
                        self.on_wakeword_detected(wakeword)
                    
                    # 切换到唤醒词检测成功状态
                    self._set_state(SystemState.WAKEWORD_DETECTED)
                    break
                    
        except Exception as e:
            logger.error(f"唤醒词检测失败: {e}")
    
    def _start_asr(self) -> None:
        """启动ASR识别"""
        logger.info("启动ASR识别...")
        
        # 切换到ASR激活状态
        self._set_state(SystemState.ASR_ACTIVE)
        self.asr_active = True
        self.stats["asr_sessions"] += 1
        
        # 触发ASR开始回调
        if self.on_asr_start:
            self.on_asr_start()
        
        # 重置KWS流（开始新的检测周期）
        if self.kws:
            self.kws_stream = self.kws.create_stream()
        
        logger.info("ASR识别已启动")
    
    def _process_asr_audio(self, audio_data: np.ndarray) -> None:
        """处理ASR音频（待集成具体ASR实现）"""
        # 这里将集成实际的ASR处理逻辑
        # 目前只是模拟处理
        pass
    
    def _check_asr_timeout(self) -> bool:
        """检查ASR是否超时（静音超时）"""
        if not self.asr_active:
            return False
        
        current_time = time.time()
        silence_duration = current_time - self.last_audio_time
        
        if silence_duration > self.asr_silence_timeout:
            logger.info(f"ASR静音超时 ({silence_duration:.1f}s > {self.asr_silence_timeout}s)")
            return True
        
        return False
    
    def _finish_asr(self) -> None:
        """结束ASR识别"""
        logger.info("结束ASR识别...")
        
        # 切换到ASR结束状态
        self._set_state(SystemState.ASR_FINISHING)
        
        # 触发ASR结束回调
        if self.on_asr_finished:
            self.on_asr_finished()
        
        # 清理ASR资源
        self.asr_active = False
        
        # 短暂延迟后返回等待状态
        time.sleep(0.5)
        self._set_state(SystemState.WAITING)
        
        logger.info("已返回等待唤醒词状态")
    
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
                        self._reload_kws()
                        self._update_file_mtime()
                time.sleep(2)  # 每2秒检查一次
            except Exception as e:
                logger.error(f"文件监控异常: {e}")
                time.sleep(2)
        
        logger.info("关键词文件监控线程结束")
    
    def _reload_kws(self) -> bool:
        """重新加载KWS模型"""
        try:
            logger.info("正在重新加载KWS模型...")
            
            # 保存当前状态
            current_state = self.get_state()
            
            # 停止当前的KWS实例
            if self.kws:
                del self.kws
                self.kws = None
                self.kws_stream = None
                logger.info("已停止当前KWS实例")
            
            # 重新初始化KWS模型
            success = self._initialize_kws()
            
            if success:
                logger.info("KWS模型重新加载成功")
                # 恢复之前的状态
                if current_state != SystemState.WAITING:
                    self._set_state(current_state)
            else:
                logger.error("KWS模型重新加载失败")
            
            return success
        except Exception as e:
            logger.error(f"重新加载KWS模型失败: {e}")
            return False
    
    def _set_state(self, new_state: SystemState) -> None:
        """安全地更新系统状态"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
        
        if old_state != new_state:
            logger.info(f"系统状态变更: {old_state.value} -> {new_state.value}")
    
    def start(self) -> bool:
        """启动唤醒词检测系统"""
        if self.is_running:
            logger.warning("系统已经在运行中")
            return False
        
        # 初始化KWS模型
        if not self._initialize_kws():
            return False
        
        # 启动音频处理线程
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        
        # 启动文件监控线程
        self.file_monitor_thread = threading.Thread(target=self._monitor_keywords_file, daemon=True)
        self.file_monitor_thread.start()
        
        # 启动麦克风音频流
        try:
            logger.info(f"正在启动麦克风音频流，采样率: {self.sample_rate}Hz")
            
            # 获取可用麦克风设备
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            logger.info(f"使用默认音频输入设备: {devices[default_input]['name']}")
            
            # 创建音频输入流
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=1600,  # 100ms的块大小（16000 * 0.1 = 1600）
                channels=1,
                callback=self._audio_callback,
                dtype='float32'
            )
            
            self.audio_stream.start()
            logger.info("麦克风音频流启动成功")
            
            return True
            
        except Exception as e:
            logger.error(f"启动麦克风音频流失败: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """停止唤醒词检测系统"""
        logger.info("正在停止唤醒词检测系统...")
        
        self.is_running = False
        
        # 停止音频流
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                logger.info("麦克风音频流已停止")
            except Exception as e:
                logger.error(f"停止音频流失败: {e}")
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            if self.processing_thread.is_alive():
                logger.warning("音频处理线程未能正常结束")
        
        # 清理队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("唤醒词检测系统已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return self.stats.copy()
    
    def get_state(self) -> SystemState:
        """获取当前系统状态"""
        with self.state_lock:
            return self.state

# 示例使用代码
if __name__ == "__main__":
    # 配置参数
    MODEL_DIR = r"D:\python\FastWhisperTranscriber\model\sherpa-onnx-kws-zipformer-wenetspeech"
    KEYWORDS_FILE = f"{MODEL_DIR}/keywords.txt"
    
    # 创建检测器实例
    detector = HotwordDetector(
        model_dir=MODEL_DIR,
        keywords_file=KEYWORDS_FILE,
        threshold=0.6,
        sample_rate=16000,
        num_threads=4
    )
    
    # 设置回调函数
    def on_wakeword_detected(wakeword: str):
        print(f"\n🎯 唤醒词检测回调: {wakeword}")
    
    def on_asr_start():
        print(f"\n🚀 ASR识别开始")
    
    def on_asr_result(text: str):
        print(f"📝 ASR结果: {text}")
    
    def on_asr_finished():
        print(f"\n✅ ASR识别完成")
    
    def on_error(error_msg: str):
        print(f"\n❌ 错误: {error_msg}")
    
    detector.on_wakeword_detected = on_wakeword_detected
    detector.on_asr_start = on_asr_start
    detector.on_asr_result = on_asr_result
    detector.on_asr_finished = on_asr_finished
    detector.on_error = on_error
    
    # 启动系统
    print("=" * 60)
    print("唤醒词检测系统启动中...")
    print(f"模型目录: {MODEL_DIR}")
    print(f"关键词文件: {KEYWORDS_FILE}")
    print("=" * 60)
    
    try:
        if detector.start():
            print("\n✅ 系统启动成功！")
            print("🎤 正在监听麦克风，请说出唤醒词...")
            print("按 Ctrl+C 停止系统\n")
            
            # 主循环，定期显示状态
            while True:
                time.sleep(5)
                stats = detector.get_stats()
                state = detector.get_state()
                print(f"[状态: {state.value}] 唤醒词检测次数: {stats['wakeword_detections']}")
                
        else:
            print("\n❌ 系统启动失败")
            
    except KeyboardInterrupt:
        print("\n\n接收到停止信号，正在关闭系统...")
    finally:
        detector.stop()
        print("系统已关闭")