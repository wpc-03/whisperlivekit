#!/usr/bin/env python3
"""
WhisperLiveKit 关键词转换工具
将中文关键词文件转换为 sherpa-onnx KWS 模型所需的拼音格式

功能：
1. 支持添加 @注释（推荐，便于阅读和维护）
2. 支持批量添加 boosting score (:N) 和 triggering threshold (#N)
3. 自动调用 sherpa-onnx-cli text2token 命令进行转换
4. 支持多种 tokens type (ppinyin, fpinyin, bpe 等)

使用方法：
python keywords_converter.py --tokens tokens.txt --tokens-type ppinyin --input keywords_raw.txt --output keywords.txt
"""

import argparse
import logging
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeywordsConverter:
    """关键词转换器类"""
    
    def __init__(
        self,
        tokens_file: str,
        tokens_type: str,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        add_annotations: bool = True,
        boost_score: Optional[float] = None,
        threshold: Optional[float] = None,
        bpe_model: Optional[str] = None,
        lexicon: Optional[str] = None
    ):
        """
        初始化关键词转换器
        
        Args:
            tokens_file: tokens.txt 文件路径
            tokens_type: tokens 类型 (ppinyin, fpinyin, bpe, cjkchar, cjkchar+bpe, phone+ppinyin)
            input_file: 输入文件路径 (默认为 keywords_raw.txt)
            output_file: 输出文件路径 (默认为 keywords.txt)
            add_annotations: 是否添加 @注释
            boost_score: 为所有关键词添加的 boosting score (如 :2.0)
            threshold: 为所有关键词添加的 triggering threshold (如 #0.6)
            bpe_model: BPE 模型文件路径 (仅当 tokens_type 为 bpe 或 cjkchar+bpe 时需要)
            lexicon: 词典文件路径 (仅当 tokens_type 为 phone+ppinyin 时需要)
        """
        self.tokens_file = Path(tokens_file)
        self.tokens_type = tokens_type
        self.input_file = Path(input_file) if input_file else None
        self.output_file = Path(output_file) if output_file else None
        self.add_annotations = add_annotations
        self.boost_score = boost_score
        self.threshold = threshold
        self.bpe_model = Path(bpe_model) if bpe_model else None
        self.lexicon = Path(lexicon) if lexicon else None
        
        # 验证文件
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """验证输入参数"""
        if not self.tokens_file.exists():
            raise FileNotFoundError(f"tokens.txt 文件不存在: {self.tokens_file}")
        
        if self.input_file and not self.input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")
        
        if self.bpe_model and not self.bpe_model.exists():
            raise FileNotFoundError(f"BPE 模型文件不存在: {self.bpe_model}")
        
        if self.lexicon and not self.lexicon.exists():
            raise FileNotFoundError(f"词典文件不存在: {self.lexicon}")
        
        # 验证 tokens_type
        valid_types = ['cjkchar', 'bpe', 'cjkchar+bpe', 'fpinyin', 'ppinyin', 'phone+ppinyin']
        if self.tokens_type not in valid_types:
            raise ValueError(f"无效的 tokens_type: {self.tokens_type}，有效值为: {valid_types}")
        
        # 检查依赖项
        if self.tokens_type in ['bpe', 'cjkchar+bpe'] and not self.bpe_model:
            raise ValueError(f"tokens_type 为 {self.tokens_type} 时需要 --bpe-model 参数")
        
        if self.tokens_type == 'phone+ppinyin' and not self.lexicon:
            raise ValueError(f"tokens_type 为 phone+ppinyin 时需要 --lexicon 参数")
    
    def _preprocess_keywords(self, input_file: Path) -> Tuple[List[str], List[str]]:
        """
        预处理关键词文件
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            (keywords, comments) 元组，keywords为要转换的关键词列表，comments为注释行列表
        """
        logger.info(f"读取关键词文件: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            logger.error(f"文件编码错误，请确保 {input_file} 是 UTF-8 编码")
            raise
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            raise
        
        keywords = []
        comments = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # 空行，添加到comments中保留
                comments.append('')
                continue
            
            if line.startswith('#'):
                # 注释行，保留原始内容
                comments.append(line)
                continue
            
            # 解析行，可能包含单独的 boost 值
            # 格式: "关键词" 或 "关键词 :3.0" 或 "关键词 3.0"
            keyword = line
            keyword_boost = self.boost_score  # 默认使用全局 boost 值
            
            # 检查是否指定了单独的 boost 值
            # 支持格式: "关键词 :3.0" 或 "关键词 3.0" 或 "关键词:3.0"
            import re
            match = re.match(r'^(.+?)\s*[:：]\s*(\d+\.?\d*)$', line)
            if match:
                keyword = match.group(1).strip()
                keyword_boost = float(match.group(2))
                logger.debug(f"检测到单独 boost 值: {keyword} -> {keyword_boost}")
            
            # 有效关键词行
            # 构建处理后的行
            parts = [keyword]
            
            # 添加 boosting score
            if keyword_boost is not None:
                parts.append(f":{keyword_boost}")
            
            # 添加 triggering threshold
            if self.threshold is not None:
                parts.append(f"#{self.threshold}")
            
            # 添加 @注释（使用原始关键词，不包含 boost 标识）
            if self.add_annotations:
                parts.append(f"@{keyword}")
            
            keywords.append(' '.join(parts))
        
        logger.info(f"处理了 {len(keywords)} 个关键词，{len(comments)} 个注释/空行")
        return keywords, comments
    
    def _create_temp_input_file(self, keywords: List[str]) -> Path:
        """
        创建临时输入文件（仅包含要转换的关键词，不包括注释）
        
        Args:
            keywords: 要转换的关键词列表
            
        Returns:
            临时文件路径
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            encoding='utf-8', 
            suffix='_keywords.txt',
            delete=False
        )
        
        with temp_file:
            for line in keywords:
                temp_file.write(line + '\n')
        
        logger.debug(f"创建临时文件: {temp_file.name}")
        return Path(temp_file.name)
    
    def _build_command(self, temp_input_file: Path, output_file: Path) -> List[str]:
        """
        构建 sherpa-onnx-cli 命令
        
        Args:
            temp_input_file: 临时输入文件路径
            output_file: 输出文件路径
            
        Returns:
            命令参数列表
        """
        cmd = [
            'sherpa-onnx-cli',
            'text2token',
            '--tokens', str(self.tokens_file),
            '--tokens-type', self.tokens_type
        ]
        
        # 添加可选参数
        if self.bpe_model:
            cmd.extend(['--bpe-model', str(self.bpe_model)])
        
        if self.lexicon:
            cmd.extend(['--lexicon', str(self.lexicon)])
        
        # 添加输入输出文件
        cmd.extend([str(temp_input_file), str(output_file)])
        
        return cmd
    
    def convert(self) -> bool:
        """
        执行关键词转换
        
        Returns:
            转换是否成功
        """
        try:
            # 确定输入输出文件
            input_file = self.input_file or Path('keywords_raw.txt')
            output_file = self.output_file or Path('keywords.txt')
            
            if not input_file.exists():
                logger.error(f"输入文件不存在: {input_file}")
                return False
            
            logger.info(f"开始关键词转换:")
            logger.info(f"  输入文件: {input_file}")
            logger.info(f"  输出文件: {output_file}")
            logger.info(f"  tokens: {self.tokens_file}")
            logger.info(f"  tokens_type: {self.tokens_type}")
            logger.info(f"  添加@注释: {self.add_annotations}")
            if self.boost_score:
                logger.info(f"  boosting score: :{self.boost_score}")
            if self.threshold:
                logger.info(f"  triggering threshold: #{self.threshold}")
            
            # 预处理关键词
            keywords, comments = self._preprocess_keywords(input_file)
            
            # 创建临时文件（仅包含关键词，不包括注释）
            temp_input_file = self._create_temp_input_file(keywords)
            
            try:
                # 构建并执行命令
                cmd = self._build_command(temp_input_file, output_file)
                logger.debug(f"执行命令: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                
                if result.returncode != 0:
                    logger.error(f"转换失败，返回码: {result.returncode}")
                    logger.error(f"错误输出: {result.stderr}")
                    return False
                
                logger.info(f"转换成功，输出文件: {output_file}")
                
                # 如果存在注释，将其添加到输出文件开头
                if comments and output_file.exists():
                    try:
                        # 读取转换后的内容
                        with open(output_file, 'r', encoding='utf-8') as f:
                            converted_content = f.read()
                        
                        # 写入注释和转换后的内容
                        with open(output_file, 'w', encoding='utf-8') as f:
                            # 写入注释行（每行后加换行）
                            for comment in comments:
                                f.write(comment + '\n')
                            # 写入转换后的内容
                            f.write(converted_content)
                        
                        logger.info(f"已将 {len(comments)} 个注释/空行添加到输出文件开头")
                    except Exception as e:
                        logger.warning(f"添加注释到输出文件时出错: {e}")
                
                # 显示转换结果的前几行
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:5]
                    if lines:
                        logger.info("转换结果示例:")
                        for i, line in enumerate(lines, 1):
                            logger.info(f"  {i}: {line.strip()}")
                
                return True
                
            finally:
                # 清理临时文件
                if temp_input_file.exists():
                    temp_input_file.unlink()
                    logger.debug(f"已删除临时文件: {temp_input_file}")
        
        except Exception as e:
            logger.error(f"转换过程中发生错误: {e}", exc_info=True)
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='WhisperLiveKit 关键词转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python keywords_converter.py --tokens model/tokens.txt --tokens-type ppinyin
  
  # 指定输入输出文件
  python keywords_converter.py --tokens model/tokens.txt --tokens-type ppinyin --input my_keywords.txt --output converted.txt
  
  # 添加 boosting score 和 threshold
  python keywords_converter.py --tokens model/tokens.txt --tokens-type ppinyin --boost 2.0 --threshold 0.6
  
  # 不添加 @注释
  python keywords_converter.py --tokens model/tokens.txt --tokens-type ppinyin --no-annotations
  
  # 使用 BPE 模型
  python keywords_converter.py --tokens model/tokens.txt --tokens-type bpe --bpe-model model/bpe.model
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--tokens',
        type=str,
        required=True,
        help='tokens.txt 文件路径'
    )
    
    parser.add_argument(
        '--tokens-type',
        type=str,
        required=True,
        choices=['cjkchar', 'bpe', 'cjkchar+bpe', 'fpinyin', 'ppinyin', 'phone+ppinyin'],
        help='tokens 类型'
    )
    
    # 可选参数
    parser.add_argument(
        '--input',
        type=str,
        default='keywords_raw.txt',
        help='输入文件路径 (默认为 keywords_raw.txt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='keywords.txt',
        help='输出文件路径 (默认为 keywords.txt)'
    )
    
    parser.add_argument(
        '--no-annotations',
        action='store_false',
        dest='add_annotations',
        default=True,
        help='不添加 @注释'
    )
    
    parser.add_argument(
        '--boost',
        type=float,
        help='为所有关键词添加相同的 boosting score (如 2.0 对应 :2.0)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='为所有关键词添加相同的 triggering threshold (如 0.6 对应 #0.6)'
    )
    
    parser.add_argument(
        '--bpe-model',
        type=str,
        help='BPE 模型文件路径 (仅当 tokens-type 为 bpe 或 cjkchar+bpe 时需要)'
    )
    
    parser.add_argument(
        '--lexicon',
        type=str,
        help='词典文件路径 (仅当 tokens-type 为 phone+ppinyin 时需要)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，显示更多详细信息'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建转换器
        converter = KeywordsConverter(
            tokens_file=args.tokens,
            tokens_type=args.tokens_type,
            input_file=args.input,
            output_file=args.output,
            add_annotations=args.add_annotations,
            boost_score=args.boost,
            threshold=args.threshold,
            bpe_model=args.bpe_model,
            lexicon=args.lexicon
        )
        
        # 执行转换
        success = converter.convert()
        
        if success:
            logger.info("关键词转换完成！")
            sys.exit(0)
        else:
            logger.error("关键词转换失败")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()