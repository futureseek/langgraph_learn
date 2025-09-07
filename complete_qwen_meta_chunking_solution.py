#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整的Qwen3-1.7B模型Meta-Chunking解决方案
包含错误处理和重试机制
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.DocumentProcessor import create_document_processor, load_qwen_model

def load_model_with_retry(model_name="Qwen/Qwen3-1.7B", max_retries=3):
    """
    带重试机制的模型加载函数
    
    Args:
        model_name: 模型名称
        max_retries: 最大重试次数
        
    Returns:
        tuple: (model, tokenizer) 或 (None, None) 如果加载失败
    """
    for attempt in range(max_retries):
        try:
            print(f"   尝试加载模型 (第 {attempt + 1}/{max_retries} 次)...")
            model, tokenizer = load_qwen_model(model_name)
            if model and tokenizer:
                print("   ✅ 模型加载成功!")
                return model, tokenizer
            else:
                print("   ⚠️ 模型加载返回空值")
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            if attempt < max_retries - 1:
                print(f"   等待5秒后重试...")
                time.sleep(5)
            else:
                print("   已达到最大重试次数")
    
    print("💡 将使用简化版本的Meta-Chunking策略")
    return None, None

def demonstrate_complete_solution():
    """演示完整的解决方案"""
    print("=== Qwen3-1.7B模型Meta-Chunking完整解决方案 ===\n")
    
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
    
    print("📋 解决方案概述:")
    print("   本方案演示如何使用Qwen3-1.7B模型进行高级文本分块处理")
    print("   通过Meta-Chunking策略，实现更智能的文本分割")
    print()
    
    # 1. 配置模型参数
    print("1. 配置模型参数:")
    model_config = {
        "model_name": "Qwen/Qwen3-1.7B",
        "base_url": "https://api-inference.modelscope.cn/v1",
        "api_key": "ms-8b59067c-75ff-4b83-900e-26e00e46c531",
        "device_map": "auto",
        "torch_dtype": "float16"
    }
    
    print(f"   模型名称: {model_config['model_name']}")
    print(f"   API地址: {model_config['base_url']}")
    print(f"   设备映射: {model_config['device_map']}")
    print(f"   精度设置: {model_config['torch_dtype']}")
    print()
    
    # 2. 加载模型（带重试机制）
    print("2. 加载Qwen3-1.7B模型:")
    model, tokenizer = load_model_with_retry("Qwen/Qwen3-1.7B", max_retries=2)
    print()
    
    # 3. 创建Meta-Chunking处理器
    print("3. 创建Meta-Chunking处理器:")
    try:
        # 创建使用Qwen3-1.7B模型的文档处理器
        processor = create_document_processor(
            chunk_size=200,
            chunk_overlap=30,
            meta_chunking_strategy="prob_subtract",  # 使用概率差策略
            meta_model=model,
            meta_tokenizer=tokenizer
        )
        print("   ✅ Meta-Chunking处理器创建成功")
        print("   策略类型: 概率差分块 (prob_subtract)")
        print("   块大小: 200字符")
        print("   重叠大小: 30字符")
    except Exception as e:
        print(f"   ❌ 处理器创建失败: {e}")
        return
    print()
    
    # 4. 处理文本
    print("4. 使用Meta-Chunking处理文本:")
    try:
        documents = processor.process_text(test_text)
        print(f"   ✅ 文本处理完成，生成了 {len(documents)} 个文档块")
        
        # 显示前3个块的内容
        print("   文档块预览:")
        for i, doc in enumerate(documents[:3]):
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"     块 {i+1}: {content_preview}...")
            print(f"         长度: {len(doc.page_content)} 字符")
        print()
    except Exception as e:
        print(f"   ❌ 文本处理失败: {e}")
        return
    
    # 5. 对比不同策略
    print("5. 对比不同Meta-Chunking策略:")
    strategies = [
        ("perplexity", "困惑度分块"),
        ("prob_subtract", "概率差分块"),
        ("semantic", "语义分块")
    ]
    
    for strategy_key, strategy_name in strategies:
        try:
            # 创建使用不同策略的处理器
            strategy_processor = create_document_processor(
                chunk_size=200,
                chunk_overlap=30,
                meta_chunking_strategy=strategy_key,
                meta_model=model,
                meta_tokenizer=tokenizer
            )
            
            # 处理文本
            strategy_documents = strategy_processor.process_text(test_text)
            print(f"   {strategy_name}: {len(strategy_documents)} 个块")
        except Exception as e:
            print(f"   {strategy_name}: 处理失败 - {e}")
    print()
    
    # 6. 与默认策略对比
    print("6. 与默认策略对比:")
    try:
        # 创建默认处理器
        default_processor = create_document_processor(
            chunk_size=200,
            chunk_overlap=30
        )
        
        # 处理文本
        default_documents = default_processor.process_text(test_text)
        print(f"   默认策略: {len(default_documents)} 个块")
        
        # 显示默认策略的第一个块
        if default_documents:
            content_preview = default_documents[0].page_content[:100].replace('\n', ' ')
            print(f"   默认策略第一个块: {content_preview}...")
            print(f"                    长度: {len(default_documents[0].page_content)} 字符")
    except Exception as e:
        print(f"   默认策略: 处理失败 - {e}")
    print()
    
    print("=== 解决方案演示完成 ===")
    print()
    print("💡 实际使用说明:")
    print("   1. 安装必要的依赖包:")
    print("      uv pip install transformers torch accelerate")
    print()
    print("   2. 如果遇到网络问题，可以:")
    print("      - 检查网络连接")
    print("      - 使用代理或VPN")
    print("      - 增加重试次数")
    print("      - 手动下载模型文件")
    print()
    print("   3. 在代码中实际加载模型:")
    print("      from tools.DocumentProcessor import load_qwen_model")
    print("      model, tokenizer = load_qwen_model('Qwen/Qwen3-1.7B')")
    print()
    print("   4. 创建处理器时传入实际的模型对象:")
    print("      processor = create_document_processor(")
    print("          meta_chunking_strategy='prob_subtract',")
    print("          meta_model=model,")
    print("          meta_tokenizer=tokenizer")
    print("      )")
    print()
    print("   5. 确保网络连接正常以访问ModelScope API")
    print("   6. 建议使用GPU以获得更好的性能")

def explain_troubleshooting():
    """解释故障排除方法"""
    print("🔧 常见问题及解决方案:\n")
    
    issues = {
        "网络连接问题": """
        问题: 下载模型时出现连接中断
        解决方案:
        1. 检查网络连接是否稳定
        2. 使用代理或VPN连接
        3. 增加重试次数
        4. 手动下载模型文件到本地
        """,
        
        "依赖包缺失": """
        问题: 缺少必要的依赖包
        解决方案:
        1. 安装transformers: uv pip install transformers
        2. 安装torch: uv pip install torch
        3. 安装accelerate: uv pip install accelerate
        """,
        
        "内存不足": """
        问题: 加载大模型时内存不足
        解决方案:
        1. 使用设备映射: device_map="auto"
        2. 降低精度: torch_dtype=torch.float16
        3. 使用量化模型
        4. 增加虚拟内存
        """,
        
        "权限问题": """
        问题: 无法写入缓存目录
        解决方案:
        1. 在Windows上激活开发者模式
        2. 以管理员身份运行
        3. 更改缓存目录: HF_HOME环境变量
        """
    }
    
    for issue, solution in issues.items():
        print(f"🔹 {issue}:")
        print(f"   {solution.strip()}")
        print()

if __name__ == "__main__":
    demonstrate_complete_solution()
    print("\n" + "="*60 + "\n")
    explain_troubleshooting()