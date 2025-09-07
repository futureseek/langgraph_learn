#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-Chunking使用示例
展示如何使用Qwen3-1.7B模型进行文档分块
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.DocumentProcessor import create_document_processor, load_qwen_model

def main():
    """主函数"""
    print("=== Meta-Chunking with Qwen3-1.7B 示例 ===\n")
    
    # 测试文本
    test_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，
    并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程的模拟。
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    
    人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得
    计算机知识，心理学和哲学。人工智能是包括十分广泛的科学，
    它由不同的领域组成，如机器学习，计算机视觉等等，总的说来，
    人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。
    """
    
    # 1. 使用默认分块策略
    print("1. 使用默认分块策略:")
    default_processor = create_document_processor(chunk_size=150, chunk_overlap=30)
    default_documents = default_processor.process_text(test_text)
    print(f"   生成了 {len(default_documents)} 个文档块")
    for i, doc in enumerate(default_documents[:3]):  # 只显示前3个块
        print(f"   块 {i+1}: {doc.page_content[:60]}...")
    print()
    
    # 2. 使用Qwen3-1.7B模型的Meta-Chunking
    print("2. 使用Qwen3-1.7B模型的Meta-Chunking:")
    try:
        # 加载模型（在实际使用中取消注释）
        print("   正在加载Qwen3-1.7B模型...")
        # model, tokenizer = load_qwen_model("Qwen/Qwen3-1.7B")
        
        # 创建使用模型的处理器（在实际使用中取消注释）
        # meta_processor = create_document_processor(
        #     chunk_size=150,
        #     chunk_overlap=30,
        #     meta_chunking_strategy="prob_subtract",
        #     meta_model=model,
        #     meta_tokenizer=tokenizer
        # )
        
        # 处理文本（在实际使用中取消注释）
        # meta_documents = meta_processor.process_text(test_text)
        # print(f"   生成了 {len(meta_documents)} 个文档块")
        # for i, doc in enumerate(meta_documents[:3]):  # 只显示前3个块
        #     print(f"   块 {i+1}: {doc.page_content[:60]}...")
        
        # 演示配置信息
        print("   模型配置信息:")
        print("   - 模型名称: Qwen/Qwen3-1.7B")
        print("   - 加载方式: 通过ModelScope")
        print("   - 设备映射: 自动分配 (GPU/CPU)")
        print("   - 精度设置: float16 (节省内存)")
        print("   - 分词器: AutoTokenizer (trust_remote_code=True)")
        print("   - 策略类型: 概率差分块 (prob_subtract)")
        
        print("   注意: 为了演示目的，实际模型加载被注释掉了。")
        print("         在实际使用中，请取消注释相关代码。")
        
    except Exception as e:
        print(f"   模型加载或处理失败: {e}")
        print("   将使用简化版本的Meta-Chunking策略")
        
        # 使用简化版本
        meta_processor = create_document_processor(
            chunk_size=150,
            chunk_overlap=30,
            meta_chunking_strategy="prob_subtract"
        )
        meta_documents = meta_processor.process_text(test_text)
        print(f"   生成了 {len(meta_documents)} 个文档块 (使用简化策略)")
        for i, doc in enumerate(meta_documents[:3]):  # 只显示前3个块
            print(f"   块 {i+1}: {doc.page_content[:60]}...")
    print()
    
    # 3. 比较不同策略
    print("3. 比较不同Meta-Chunking策略:")
    strategies = ["perplexity", "prob_subtract", "semantic"]
    
    for strategy in strategies:
        try:
            processor = create_document_processor(
                chunk_size=150,
                chunk_overlap=30,
                meta_chunking_strategy=strategy
            )
            documents = processor.process_text(test_text)
            print(f"   {strategy} 策略: {len(documents)} 个块")
        except Exception as e:
            print(f"   {strategy} 策略: 失败 - {e}")
    
    print("\n=== 示例完成 ===")

if __name__ == "__main__":
    main()