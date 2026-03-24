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
            # 创建空的关键字文件
            self._create_empty_file(file_path)
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
    
    def _create_empty_file(self, file_path: Path) -> None:
        """
        创建空的关键字文件
        
        Args:
            file_path: 文件路径
        """
        try:
            os.makedirs(file_path.parent, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# 关键字配置文件\n# 每行一个关键字\n# 以#开头的行会被忽略\n# 空行也会被忽略\n\n")
            logger.info(f"Created empty keywords file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create empty keywords file: {file_path}, error: {e}")
    
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
    
    def add_keyword(self, keyword: str) -> bool:
        """
        添加关键字
        
        Args:
            keyword: 要添加的关键字
            
        Returns:
            True如果添加成功，否则False
        """
        if not self.keywords_file_path:
            logger.error("No keywords file specified")
            return False
        
        if not keyword or keyword.strip() == "":
            logger.error("Empty keyword provided")
            return False
        
        # 移除前后空格
        keyword = keyword.strip()
        
        # 检查是否已存在
        if keyword in self.keywords:
            logger.warning(f"Keyword already exists: {keyword}")
            return False
        
        try:
            # 读取文件内容（包括注释和空行）
            file_path = Path(self.keywords_file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 添加新关键字
            self.keywords.append(keyword)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                # 保留原有内容
                for line in lines:
                    f.write(line)
                # 添加新关键字
                f.write(f"{keyword}\n")
            
            logger.info(f"Added keyword: {keyword}")
            return True
        except Exception as e:
            logger.error(f"Failed to add keyword: {e}")
            # 回滚内存中的关键字列表
            if keyword in self.keywords:
                self.keywords.remove(keyword)
            return False
    
    def update_keyword(self, old_keyword: str, new_keyword: str) -> bool:
        """
        更新关键字
        
        Args:
            old_keyword: 旧关键字
            new_keyword: 新关键字
            
        Returns:
            True如果更新成功，否则False
        """
        if not self.keywords_file_path:
            logger.error("No keywords file specified")
            return False
        
        if not old_keyword or not new_keyword:
            logger.error("Empty keyword provided")
            return False
        
        # 移除前后空格
        old_keyword = old_keyword.strip()
        new_keyword = new_keyword.strip()
        
        # 检查旧关键字是否存在
        if old_keyword not in self.keywords:
            logger.warning(f"Old keyword not found: {old_keyword}")
            return False
        
        # 检查新关键字是否已存在
        if new_keyword != old_keyword and new_keyword in self.keywords:
            logger.warning(f"New keyword already exists: {new_keyword}")
            return False
        
        try:
            # 读取文件内容
            file_path = Path(self.keywords_file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换关键字
            # 确保只替换完整的行
            import re
            pattern = r'^' + re.escape(old_keyword) + r'$' 
            new_content = re.sub(pattern, new_keyword, content, flags=re.MULTILINE)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # 更新内存中的关键字列表
            index = self.keywords.index(old_keyword)
            self.keywords[index] = new_keyword
            
            logger.info(f"Updated keyword: {old_keyword} -> {new_keyword}")
            return True
        except Exception as e:
            logger.error(f"Failed to update keyword: {e}")
            return False
    
    def delete_keyword(self, keyword: str) -> bool:
        """
        删除关键字
        
        Args:
            keyword: 要删除的关键字
            
        Returns:
            True如果删除成功，否则False
        """
        if not self.keywords_file_path:
            logger.error("No keywords file specified")
            return False
        
        if not keyword:
            logger.error("Empty keyword provided")
            return False
        
        # 移除前后空格
        keyword = keyword.strip()
        
        # 检查是否存在
        if keyword not in self.keywords:
            logger.warning(f"Keyword not found: {keyword}")
            return False
        
        try:
            # 读取文件内容（逐行）
            file_path = Path(self.keywords_file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 过滤掉要删除的关键字行
            new_lines = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line != keyword:
                    new_lines.append(line)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # 从内存中删除
            self.keywords.remove(keyword)
            
            logger.info(f"Deleted keyword: {keyword}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete keyword: {e}")
            return False
    
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
