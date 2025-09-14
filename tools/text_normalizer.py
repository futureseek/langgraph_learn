#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本规范化工具 - 用于在文档分块前对文本进行清洗和标准化处理
支持NFKC归一化、标点符号标准化、控制字符清理等功能
"""

import re
import unicodedata
from typing import Union

class TextNormalizer:
    """文本规范化器"""
    
    def __init__(self):
        # 全角到半角标点映射
        self.full_to_half_punct = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '；': ';',
            '：': ':',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '《': '<',
            '》': '>',
            '、': ',',
            '…': '...',
        }
        
        # 创建正则表达式模式
        self.punct_pattern = re.compile('|'.join(re.escape(key) for key in self.full_to_half_punct.keys()))
    
    def normalize_text(self, text: Union[str, None]) -> Union[str, None]:
        """
        对文本进行完整的规范化处理
        
        Args:
            text: 原始文本
            
        Returns:
            str: 规范化后的文本
        """
        if not text:
            return text
            
        # 1. NFKC归一化
        normalized = unicodedata.normalize('NFKC', text)
        
        # 2. 替换全角标点为半角标点
        normalized = self.punct_pattern.sub(
            lambda match: self.full_to_half_punct[match.group(0)], 
            normalized
        )
        
        # 3. 清理不可见控制字符（保留换行符）
        # 移除除了换行符(\n)、制表符(\t)、回车符(\r)之外的控制字符
        normalized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', normalized)
        
        # 4. 标准化空白字符
        # 将连续的空白字符（空格、制表符等）替换为单个空格
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        
        # 5. 标准化换行符
        # 将不同平台的换行符统一为\n
        normalized = re.sub(r'\r\n|\r', '\n', normalized)
        
        # 6. 移除行首行尾的空白字符
        lines = normalized.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        normalized = '\n'.join(cleaned_lines)
        
        # 7. 移除多余的空行（保留单个空行作为段落分隔）
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        
        return normalized.strip()
    
    def clean_control_characters(self, text: str) -> str:
        """
        清理控制字符
        
        Args:
            text: 文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除除了换行符(\n)、制表符(\t)、回车符(\r)之外的控制字符
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    def standardize_punctuation(self, text: str) -> str:
        """
        标准化标点符号
        
        Args:
            text: 文本
            
        Returns:
            str: 标准化后的文本
        """
        return self.punct_pattern.sub(
            lambda match: self.full_to_half_punct[match.group(0)], 
            text
        )
    
    def standardize_whitespace(self, text: str) -> str:
        """
        标准化空白字符
        
        Args:
            text: 文本
            
        Returns:
            str: 标准化后的文本
        """
        # 将连续的空白字符替换为单个空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 统一换行符
        text = re.sub(r'\r\n|\r', '\n', text)
        return text

def create_text_normalizer() -> TextNormalizer:
    """
    创建文本规范化器实例
    
    Returns:
        TextNormalizer: 文本规范化器实例
    """
    return TextNormalizer()

# 测试代码
if __name__ == "__main__":
    # 创建测试文本（包含各种需要规范化的字符）
    test_text = """
    这是一个　测试文档　。
    它包含　全角空格　、全角标点，还有各种\u000c控制字符\u00ad。
    不同的换行符\r\n和\r应该被统一。
    
    连续的空行
    
    应该被合并。
    """
    
    print("原始文本:")
    print(repr(test_text))
    print("\n" + "="*50 + "\n")
    
    # 创建规范化器
    normalizer = create_text_normalizer()
    
    # 规范化文本
    normalized_text = normalizer.normalize_text(test_text)
    
    print("规范化后的文本:")
    print(repr(normalized_text))
    print("\n" + "="*50 + "\n")
    
    # 显示处理前后的对比
    print("处理前后对比:")
    print("原文本行数:", len(test_text.split('\n')))
    print("规范化后行数:", len(normalized_text.split('\n')))