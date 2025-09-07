#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于ModelScope API的Meta-Chunking解决方案
使用Qwen3-1.7B模型的API接口而不是直接加载模型
"""

import os
import sys
import json
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from tools.DocumentProcessor import create_document_processor, MetaChunking

class APIMetaChunking(MetaChunking):
    """基于API的Meta-Chunking实现"""
    
    def __init__(self, api_key=None, base_url=None):
        """
        初始化API Meta-Chunking
        
        Args:
            api_key: ModelScope API密钥
            base_url: API基础URL
        """
        self.api_key = api_key or "ms-8b59067c-75ff-4b83-900e-26e00e46c531"
        self.base_url = base_url or "https://api-inference.modelscope.cn/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        print(f"✅ API客户端初始化成功")
        print(f"   Base URL: {self.base_url}")
    
    def get_prob_subtract(self, sentence1, sentence2, language):
        """使用API计算两个句子的概率差"""
        try:
            if language == 'zh':
                query = '''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
                1. 将"{}"分割成"{}"与"{}"两部分；
                2. 将"{}"不进行分割，保持原形式；
                请回答1或2。'''.format(sentence1 + sentence2, sentence1, sentence2, sentence1 + sentence2)
            else:
                query = '''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
                1. Split "{}" into "{}" and "{}" two parts;
                2. Keep "{}" unsplit in its original form;
                Please answer 1 or 2.'''.format(sentence1 + ' ' + sentence2, sentence1, sentence2, sentence1 + ' ' + sentence2)
            
            response = self.client.chat.completions.create(
                model="Qwen/Qwen3-1.7B",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"API响应: {answer}")
            
            # 简化处理，如果回答包含"1"则返回正值，否则返回负值
            if "1" in answer:
                return 0.5  # 正值表示应该分割
            else:
                return -0.5  # 负值表示不应该分割
                
        except Exception as e:
            print(f"API调用失败: {e}")
            return 0  # 返回默认值
    
    def prob_subtract_chunking(self, text, threshold=0, language='zh'):
        """基于API的概率差分块策略"""
        try:
            # 分割文本为句子
            segments = self.split_text_by_punctuation(text, language)
            segments = [item for item in segments if item.strip()]
            
            if len(segments) <= 1:
                return [text]
            
            chunks = []
            current_chunk = ""
            
            for i, segment in enumerate(segments):
                if current_chunk == "":
                    current_chunk = segment
                else:
                    # 使用API计算概率差
                    prob_diff = self.get_prob_subtract(current_chunk, segment, language)
                    
                    if prob_diff > threshold:
                        # 不分割，继续添加到当前块
                        if language == 'zh':
                            current_chunk += segment
                        else:
                            current_chunk += " " + segment
                    else:
                        # 分割，将当前块添加到结果中
                        chunks.append(current_chunk)
                        current_chunk = segment
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks if chunks else [text]
        except Exception as e:
            print(f"API概率差分块失败: {e}")
            return self._fallback_chunking(text)

def demonstrate_api_solution():
    """演示基于API的解决方案"""
    print("=== 基于ModelScope API的Meta-Chunking解决方案 ===\n")
    
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
    
    print("1. 初始化API客户端:")
    try:
        api_meta_chunking = APIMetaChunking()
        print("   ✅ API客户端初始化成功")
    except Exception as e:
        print(f"   ❌ API客户端初始化失败: {e}")
        return
    print()
    
    print("2. 创建基于API的Meta-Chunking处理器:")
    try:
        processor = create_document_processor(
            chunk_size=200,
            chunk_overlap=30,
            meta_chunking_strategy="prob_subtract",
            meta_model=api_meta_chunking,  # 传入API客户端
            meta_tokenizer=None
        )
        print("   ✅ 处理器创建成功")
    except Exception as e:
        print(f"   ❌ 处理器创建失败: {e}")
        return
    print()
    
    print("3. 使用API进行文本分块:")
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
    
    print("4. 与默认策略对比:")
    try:
        default_processor = create_document_processor(chunk_size=200, chunk_overlap=30)
        default_documents = default_processor.process_text(test_text)
        print(f"   默认策略: {len(default_documents)} 个块")
        
        if default_documents:
            content_preview = default_documents[0].page_content[:100].replace('\n', ' ')
            print(f"   默认策略第一个块: {content_preview}...")
            print(f"                    长度: {len(default_documents[0].page_content)} 字符")
    except Exception as e:
        print(f"   默认策略: 处理失败 - {e}")
    print()
    
    print("=== API解决方案演示完成 ===")
    print()
    print("💡 优势:")
    print("   1. 无需本地加载大模型")
    print("   2. 减少内存占用")
    print("   3. 避免网络下载问题")
    print("   4. 更稳定的性能")
    print()
    print("⚠️ 注意事项:")
    print("   1. 需要有效的API密钥")
    print("   2. 网络连接是必需的")
    print("   3. API调用可能产生费用")
    print("   4. 响应时间取决于网络延迟")

if __name__ == "__main__":
    demonstrate_api_solution()