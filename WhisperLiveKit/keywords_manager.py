import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class KeywordsManager:
    """
    关键字管理类，用于读取和解析关键字文件
    """
    
    def __init__(self, keywords_file_path: Optional[str] = None):
        """
        初始化关键字管理器
        
        Args:
            keywords_file_path: 关键字文件路径
        """
        self.keywords_file_path = keywords_file_path
        self.keywords: List[str] = []
        self._load_keywords()
    
    def _load_keywords(self) -> None:
        """
        加载关键字文件
        """
        if not self.keywords_file_path:
            logger.debug("No keywords file specified")
            return
        
        file_path = Path(self.keywords_file_path)
        
        if not file_path.exists():
            logger.warning(f"Keywords file not found: {file_path}")
            return
        
        if not file_path.is_file():
            logger.warning(f"Keywords path is not a file: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.keywords = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    self.keywords.append(line)
            
            logger.info(f"Loaded {len(self.keywords)} keywords from {file_path}")
            
        except UnicodeDecodeError:
            logger.error(f"Failed to decode keywords file: {file_path}, please ensure it's UTF-8 encoded")
        except IOError as e:
            logger.error(f"Failed to read keywords file: {file_path}, error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading keywords file: {file_path}, error: {e}")
    
    def get_keywords(self) -> List[str]:
        """
        获取所有关键字
        
        Returns:
            关键字列表
        """
        return self.keywords.copy()
    
    def get_keywords_as_string(self, separator: str = ", ") -> str:
        """
        将关键字拼接成字符串
        
        Args:
            separator: 关键字之间的分隔符
            
        Returns:
            拼接后的关键字字符串
        """
        return separator.join(self.keywords)
    
    def has_keywords(self) -> bool:
        """
        检查是否有关键字
        
        Returns:
            True如果有关键字，否则False
        """
        return len(self.keywords) > 0
    
    def reload_keywords(self) -> None:
        """
        重新加载关键字文件
        """
        logger.info("Reloading keywords...")
        self.keywords = []
        self._load_keywords()
    
    @staticmethod
    def create_example_file(file_path: str, keywords: Optional[List[str]] = None) -> None:
        """
        创建示例关键字文件
        
        Args:
            file_path: 文件路径
            keywords: 可选的关键字列表，如果不提供则使用默认示例
        """
        if keywords is None:
            keywords = [
                "Whisper",
                "ASR",
                "语音识别",
                "LocalAgreement",
                "SimulStreaming",
                "LiveKit",
            ]
        
        example_content = """# 关键字配置文件
# 每行一个关键字
# 以#开头的行会被忽略
# 空行也会被忽略

"""
        
        for keyword in keywords:
            example_content += f"{keyword}\n"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(example_content)
            logger.info(f"Created example keywords file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create example keywords file: {file_path}, error: {e}")


def load_keywords_from_file(file_path: Optional[str]) -> Tuple[List[str], str]:
    """
    从文件加载关键字的便捷函数
    
    Args:
        file_path: 关键字文件路径
        
    Returns:
        (关键字列表, 拼接后的关键字字符串)
    """
    manager = KeywordsManager(file_path)
    return manager.get_keywords(), manager.get_keywords_as_string()
