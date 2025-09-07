#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DocumentProcessor测试脚本
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.DocumentProcessor import create_document_processor

def main():
    """主函数"""
    print("=== DocumentProcessor 测试 ===\n")
    
    # 测试文本
    test_text = """
    这是一个测试文档。它包含多个句子，用于测试我们的文档处理器。
    文档处理器应该能够正确地将这个文本分割成多个块。
    每个块应该保持语义的完整性。
    
    这是第二个段落。它也应该被正确处理。
    我们可以测试不同的分块策略。
    
    最后一个段落。测试应该覆盖所有情况。
    确保所有的功能都能正常工作。
    """
    
    # 测试默认分块策略
    print("1. 测试默认分块策略:")
    processor = create_document_processor(chunk_size=100, chunk_overlap=20)
    documents = processor.process_text(test_text)
    print(f"生成了 {len(documents)} 个文档块")
    for i, doc in enumerate(documents):
        print(f"  块 {i+1}: {doc.page_content[:50]}...")
    print()
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    main()