import logging

logger = logging.getLogger(__name__)

# 尝试导入 opencc
try:
    import opencc
    OPENCC_AVAILABLE = True
    # 创建转换器：繁体转简体
    _t2s_converter = opencc.OpenCC('t2s')  # Traditional to Simplified
    logger.info("Using OpenCC for Chinese text conversion")
except ImportError:
    OPENCC_AVAILABLE = False
    logger.warning("OpenCC not available. Using built-in conversion table. "
                   "For better conversion, install: pip install opencc-python-reimplemented")


def traditional_to_simplified(text: str) -> str:
    """
    将繁体中文转换为简体中文
    
    Args:
        text: 输入文本（可能包含繁体字）
        
    Returns:
        转换后的简体中文文本
    """
    if not text:
        return text
    
    if OPENCC_AVAILABLE:
        try:
            return _t2s_converter.convert(text)
        except Exception as e:
            logger.warning(f"OpenCC conversion failed: {e}")
            return text
    else:
        # OpenCC 不可用时，直接返回原文本
        logger.warning("OpenCC not available, returning original text")
        return text



def contains_traditional(text: str) -> bool:
    """
    检查文本是否包含繁体字
    
    Args:
        text: 输入文本
        
    Returns:
        是否包含繁体字
    """
    if not text:
        return False
    
    if OPENCC_AVAILABLE:
        # 使用 OpenCC 转换后比较
        return traditional_to_simplified(text) != text
    else:
        # OpenCC 不可用时，无法检测，返回 False
        return False

