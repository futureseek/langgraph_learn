#!/usr/bin/env python3
"""
RAG功能测试脚本
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.RAGRetriever import create_rag_tools
from langchain_openai import ChatOpenAI

def test_rag_functionality():
    """测试RAG功能"""
    print("🧪 开始测试RAG功能...")
    
    try:
        # 创建模型实例
        model = ChatOpenAI(
                model='Qwen/Qwen3-1.7B',
                base_url='https://api-inference.modelscope.cn/v1',
                api_key='ms-8b59067c-75ff-4b83-900e-26e00e46c531',
                streaming=True  # 使用流式调用，可能不需要enable_thinking参数
            )
        
        # 创建RAG工具
        rag_tools = create_rag_tools(model=model)
        
        # 查找工具
        add_doc_tool = None
        qa_tool = None
        stats_tool = None
        
        for tool in rag_tools:
            if tool.name == "add_document_to_rag":
                add_doc_tool = tool
            elif tool.name == "rag_question_answer":
                qa_tool = tool
            elif tool.name == "get_rag_stats":
                stats_tool = tool
        
        if not all([add_doc_tool, qa_tool, stats_tool]):
            print("❌ 未找到所需的RAG工具")
            return False
        
        print("✅ RAG工具创建成功")
        
        # 测试1: 查看初始状态
        print("\n📊 查看知识库初始状态...")
        stats_result = stats_tool.func()
        print(stats_result)
        
        # 测试2: 添加测试文档
        print("\n📄 添加测试文档...")
        test_doc_path = "./test_document.md"
        if os.path.exists(test_doc_path):
            add_result = add_doc_tool.func(test_doc_path)
            print(add_result)
        else:
            print(f"❌ 测试文档不存在: {test_doc_path}")
            return False
        
        # 测试3: 查看添加后的状态
        print("\n📊 查看添加文档后的状态...")
        stats_result = stats_tool.func()
        print(stats_result)
        
        # 测试4: 测试问答功能
        print("\n🤖 测试RAG问答功能...")
        questions = [
            "这个系统支持哪些智能体？",
            "如何安装这个系统？",
            "RAG功能有什么特点？"
        ]
        
        for question in questions:
            print(f"\n❓ 问题: {question}")
            answer = qa_tool.func(question)
            print(f"🤖 回答: {answer}")
            print("-" * 50)
        
        print("✅ RAG功能测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_functionality()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败！")
        sys.exit(1)