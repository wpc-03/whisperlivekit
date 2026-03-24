import logging
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class WakewordsManager:
    """
    唤醒词管理类
    """
    
    def __init__(self, model_dir: str):
        """
        初始化唤醒词管理器
        
        Args:
            model_dir: 唤醒词模型目录
        """
        self.model_dir = Path(model_dir)
        self.keywords_raw_file = self.model_dir / "keywords_raw.txt"
        self.keywords_file = self.model_dir / "keywords.txt"
        self.tokens_file = self.model_dir / "tokens.txt"
        self._ensure_files_exists()
    
    def _ensure_files_exists(self) -> None:
        """
        确保文件存在
        """
        # 确保模型目录存在
        if not self.model_dir.exists():
            logger.error(f"Model directory not found: {self.model_dir}")
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # 确保tokens.txt文件存在
        if not self.tokens_file.exists():
            logger.error(f"Tokens file not found: {self.tokens_file}")
            raise FileNotFoundError(f"Tokens file not found: {self.tokens_file}")
        
        # 确保keywords_raw.txt文件存在
        if not self.keywords_raw_file.exists():
            # 创建空的keywords_raw.txt文件
            try:
                with open(self.keywords_raw_file, 'w', encoding='utf-8') as f:
                    f.write("# 唤醒词配置文件\n# 每行一个唤醒词\n# 格式: 唤醒词 [:boost值] [#threshold值]\n# 示例: 你好 :2.0 #0.5\n\n")
                logger.info(f"Created empty keywords_raw.txt file: {self.keywords_raw_file}")
            except Exception as e:
                logger.error(f"Failed to create keywords_raw.txt file: {e}")
                raise
    
    def get_wakewords(self) -> List[Dict[str, Any]]:
        """
        获取所有唤醒词
        
        Returns:
            唤醒词列表，每个唤醒词包含word、boost、threshold字段
        """
        wakewords = []
        
        try:
            with open(self.keywords_raw_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析唤醒词行
                import re
                match = re.match(r'^(.+?)(?:\s*[:：]\s*(\d+\.?\d*))?(?:\s*#\s*(\d+\.?\d*))?\s*$', line)
                if match:
                    word = match.group(1).strip()
                    boost = float(match.group(2)) if match.group(2) else None
                    threshold = float(match.group(3)) if match.group(3) else None
                    
                    wakewords.append({
                        "word": word,
                        "boost": boost,
                        "threshold": threshold
                    })
            
            logger.info(f"Loaded {len(wakewords)} wakewords from {self.keywords_raw_file}")
        except Exception as e:
            logger.error(f"Failed to load wakewords: {e}")
        
        return wakewords
    
    def add_wakeword(self, word: str, boost: Optional[float] = None, threshold: Optional[float] = None) -> bool:
        """
        添加唤醒词
        
        Args:
            word: 唤醒词
            boost: boost值
            threshold: threshold值
            
        Returns:
            True如果添加成功，否则False
        """
        if not word or word.strip() == "":
            logger.error("Empty wakeword provided")
            return False
        
        # 移除前后空格
        word = word.strip()
        
        # 检查是否已存在
        existing_wakewords = self.get_wakewords()
        for ww in existing_wakewords:
            if ww["word"] == word:
                logger.warning(f"Wakeword already exists: {word}")
                return False
        
        try:
            # 构建唤醒词行
            parts = [word]
            if boost is not None:
                parts.append(f":{boost}")
            if threshold is not None:
                parts.append(f"#{threshold}")
            wakeword_line = " ".join(parts)
            
            # 追加到文件
            with open(self.keywords_raw_file, 'a', encoding='utf-8') as f:
                f.write(f"{wakeword_line}\n")
            
            # 自动转换唤醒词格式
            if not self.convert_wakewords():
                logger.warning("Failed to convert wakewords after adding")
            
            logger.info(f"Added wakeword: {wakeword_line}")
            return True
        except Exception as e:
            logger.error(f"Failed to add wakeword: {e}")
            return False
    
    def update_wakeword(self, old_word: str, new_word: str, boost: Optional[float] = None, threshold: Optional[float] = None) -> bool:
        """
        更新唤醒词
        
        Args:
            old_word: 旧唤醒词
            new_word: 新唤醒词
            boost: boost值
            threshold: threshold值
            
        Returns:
            True如果更新成功，否则False
        """
        if not old_word or not new_word:
            logger.error("Empty wakeword provided")
            return False
        
        # 移除前后空格
        old_word = old_word.strip()
        new_word = new_word.strip()
        
        # 检查旧唤醒词是否存在
        existing_wakewords = self.get_wakewords()
        old_wakeword = None
        for ww in existing_wakewords:
            if ww["word"] == old_word:
                old_wakeword = ww
                break
        
        if not old_wakeword:
            logger.warning(f"Old wakeword not found: {old_word}")
            return False
        
        # 检查新唤醒词是否已存在（如果与旧唤醒词不同）
        if new_word != old_word:
            for ww in existing_wakewords:
                if ww["word"] == new_word:
                    logger.warning(f"New wakeword already exists: {new_word}")
                    return False
        
        try:
            # 读取文件内容
            with open(self.keywords_raw_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 构建新的唤醒词行
            parts = [new_word]
            if boost is not None:
                parts.append(f":{boost}")
            elif old_wakeword.get("boost") is not None:
                parts.append(f":{old_wakeword['boost']}")
            if threshold is not None:
                parts.append(f"#{threshold}")
            elif old_wakeword.get("threshold") is not None:
                parts.append(f"#{old_wakeword['threshold']}")
            new_wakeword_line = " ".join(parts)
            
            # 替换唤醒词行
            new_lines = []
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    new_lines.append(line)
                    continue
                
                # 检查是否是要更新的唤醒词行
                import re
                match = re.match(r'^(.+?)(?:\s*[:：]\s*(\d+\.?\d*))?(?:\s*#\s*(\d+\.?\d*))?\s*$', stripped_line)
                if match and match.group(1).strip() == old_word:
                    new_lines.append(f"{new_wakeword_line}\n")
                else:
                    new_lines.append(line)
            
            # 写回文件
            with open(self.keywords_raw_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # 自动转换唤醒词格式
            if not self.convert_wakewords():
                logger.warning("Failed to convert wakewords after updating")
            
            logger.info(f"Updated wakeword: {old_word} -> {new_wakeword_line}")
            return True
        except Exception as e:
            logger.error(f"Failed to update wakeword: {e}")
            return False
    
    def delete_wakeword(self, word: str) -> bool:
        """
        删除唤醒词
        
        Args:
            word: 要删除的唤醒词
            
        Returns:
            True如果删除成功，否则False
        """
        if not word:
            logger.error("Empty wakeword provided")
            return False
        
        # 移除前后空格
        word = word.strip()
        
        # 检查是否存在
        existing_wakewords = self.get_wakewords()
        found = False
        for ww in existing_wakewords:
            if ww["word"] == word:
                found = True
                break
        
        if not found:
            logger.warning(f"Wakeword not found: {word}")
            return False
        
        try:
            # 读取文件内容
            with open(self.keywords_raw_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 过滤掉要删除的唤醒词行
            new_lines = []
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    new_lines.append(line)
                    continue
                
                # 检查是否是要删除的唤醒词行
                import re
                match = re.match(r'^(.+?)(?:\s*[:：]\s*(\d+\.?\d*))?(?:\s*#\s*(\d+\.?\d*))?\s*$', stripped_line)
                if not (match and match.group(1).strip() == word):
                    new_lines.append(line)
            
            # 写回文件
            with open(self.keywords_raw_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # 自动转换唤醒词格式
            if not self.convert_wakewords():
                logger.warning("Failed to convert wakewords after deleting")
            
            logger.info(f"Deleted wakeword: {word}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete wakeword: {e}")
            return False
    
    def convert_wakewords(self) -> bool:
        """
        转换唤醒词格式
        
        Returns:
            True如果转换成功，否则False
        """
        try:
            # 构建命令
            cmd = [
                'python', 'wake_keywords_converter.py',
                '--tokens', str(self.tokens_file),
                '--tokens-type', 'ppinyin',
                '--input', str(self.keywords_raw_file),
                '--output', str(self.keywords_file)
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=str(Path(__file__).parent.parent)
            )
            
            if result.returncode != 0:
                logger.error(f"Conversion failed, return code: {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
            
            logger.info(f"Conversion successful, output file: {self.keywords_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to convert wakewords: {e}")
            return False
