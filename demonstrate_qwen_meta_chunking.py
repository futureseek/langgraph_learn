#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
演示如何使用Qwen3-1.7B模型进行Meta-Chunking分块
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.DocumentProcessor import create_document_processor, load_qwen_model, MetaChunking

def main():
    """主函数"""
    print("=== 演示使用Qwen3-1.7B模型进行Meta-Chunking分块 ===\n")
    
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
    
    print("1. 首先加载Qwen3-1.7B模型:")
    try:
        # 加载Qwen3-1.7B模型
        print("   正在通过ModelScope加载Qwen3-1.7B模型...")
        print("   模型配置:")
        print("   - 模型名称: Qwen/Qwen3-1.7B")
        print("   - API地址: https://api-inference.modelscope.cn/v1")
        print("   - API密钥: ms-8b59067c-75ff-4b83-900e-26e00e46c531")
        print("   - 设备映射: auto")
        print("   - 精度设置: float16")
        
        # 实际加载模型（在实际使用中取消注释）
        model, tokenizer = load_qwen_model("Qwen/Qwen3-1.7B")
        
        # 为了演示目的，我们创建一个模拟的模型对象
        # 在实际使用中，请使用上面的load_qwen_model函数
        # model = "模拟的Qwen3-1.7B模型实例"
        # tokenizer = "模拟的分词器实例"
        
        if model and tokenizer:
            print("   ✅ 模型加载成功!")
        else:
            print("   ⚠️ 模型加载失败，将使用简化版本")
            
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        model, tokenizer = None, None
    print()
    
    print("2. 创建使用Qwen3-1.7B模型的Meta-Chunking处理器:")
    try:
        # 创建使用模型的Meta-Chunking处理器
        meta_chunking = MetaChunking(model, tokenizer)
        print("   ✅ Meta-Chunking处理器创建成功!")
        
        # 创建文档处理器
        processor = create_document_processor(
            chunk_size=150,
            chunk_overlap=20,
            meta_chunking_strategy="prob_subtract",
            meta_model=model,
            meta_tokenizer=tokenizer
        )
        print("   ✅ 文档处理器创建成功!")
        
    except Exception as e:
        print(f"   ❌ 处理器创建失败: {e}")
    print()
    
    print("3. 使用Meta-Chunking进行文本分块:")
    try:
        # 处理文本
        documents = processor.process_text(test_text)
        print(f"   生成了 {len(documents)} 个文档块")
        
        # 显示前3个块的内容
        for i, doc in enumerate(documents[:3]):
            print(f"   块 {i+1}: {doc.page_content[:100]}...")
            if len(doc.page_content) > 100:
                print(f"        ... (长度: {len(doc.page_content)} 字符)")
        
    except Exception as e:
        print(f"   ❌ 文本处理失败: {e}")
    print()
    
    print("4. 比较不同策略:")
    strategies = ["perplexity", "prob_subtract", "semantic"]
    
    for strategy in strategies:
        try:
            processor = create_document_processor(
                chunk_size=150,
                chunk_overlap=20,
                meta_chunking_strategy=strategy,
                meta_model=model if model != "模拟的Qwen3-1.7B模型实例" else None,
                meta_tokenizer=tokenizer if tokenizer != "模拟的分词器实例" else None
            )
            documents = processor.process_text(test_text)
            print(f"   {strategy} 策略: {len(documents)} 个块")
        except Exception as e:
            print(f"   {strategy} 策略: 失败 - {e}")
    print()
    
    print("5. 与默认策略对比:")
    try:
        default_processor = create_document_processor(chunk_size=150, chunk_overlap=20)
        default_documents = default_processor.process_text(test_text)
        print(f"   默认策略: {len(default_documents)} 个块")
        
        # 显示默认策略的第一个块
        if default_documents:
            print(f"   默认策略第一个块: {default_documents[0].page_content[:100]}...")
            if len(default_documents[0].page_content) > 100:
                print(f"                    ... (长度: {len(default_documents[0].page_content)} 字符)")
    except Exception as e:
        print(f"   默认策略: 失败 - {e}")
    print()
    
    print("=== 演示完成 ===")
    print("\n💡 使用说明:")
    print("   1. 要实际使用Qwen3-1.7B模型，请取消注释load_qwen_model函数调用")
    print("   2. 确保已安装transformers和torch依赖包")
    print("   3. 确保网络连接正常以访问ModelScope API")
    print("   4. 在生产环境中，建议使用GPU以获得更好的性能")

if __name__ == "__main__":
    main()