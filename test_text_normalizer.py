#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextNormalizer 测试脚本
"""

from tools.text_normalizer import TextNormalizer

def test_text_normalizer():
    """测试TextNormalizer的各种功能"""
    normalizer = TextNormalizer()
    
    # 测试用例
    test_cases = [
        # 全角空格测试
        ("这是　一个　测试文档　。", "这是 一个 测试文档 ."),
        # 全角标点测试
        ("包含全角标点，还有各种\u000c控制字符\u00ad。", "包含全角标点,还有各种控制字符\xad."),
        # NFKC归一化测试
        ("ﬃ", "ffi"),  # NFKC归一化
    ]
    
    print("Testing TextNormalizer with various cases:")
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = normalizer.normalize_text(input_text)
        print(f"{i}. Original: {repr(input_text)}")
        print(f"   Expected: {repr(expected)}")
        print(f"   Actual:   {repr(result)}")
        print(f"   Pass: {result == expected}")
        print()

if __name__ == "__main__":
    test_text_normalizer()