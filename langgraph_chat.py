import os
from langchain_openai import ChatOpenAI
from typing import TypedDict
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage  # 添加 ToolMessage
import operator
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from tools import MessagerManager
from tools.document_exporter import create_document_export_tool
from tools.DocumentReader import create_document_reader_tool



class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], MessagerManager(max_woking_memory=100,max_history=500)]

 
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.message_manager = MessagerManager(max_woking_memory=100, max_history=500)
        # 绑定工具到模型
        self.model = self.model.bind_tools(tools)
        
        # 构建工作流图
        graph = StateGraph(AgentState)  
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {
                True: "action",
                False: END
            }
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")  # 正确的入口点设置
        self.graph = graph.compile()
    
    def exists_action(self,state:AgentState):
        """
        检查工具调用
        """

        if not state["messages"]:
            return False
        last_message = state["messages"][-1]
        if isinstance(last_message,AIMessage):
            return hasattr(last_message,'tool_calls') and len(last_message.tool_calls) > 0
        if isinstance(last_message, ToolMessage):
            return False
        return False

    def call_openai(self, state:AgentState):
        messages = state["messages"]
        if self.system:
            # 确保系统消息在最前面
            system_msg = [SystemMessage(content=self.system)]
            non_system_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]
            messages = system_msg + non_system_msgs
        
        print(f"发送给模型的消息数量: {len(messages)}")
        response = self.model.invoke(messages)
        return {"messages": [response]}
    def take_action(self , state:AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


if __name__ == "__main__":

    os.environ["TAVILY_API_KEY"] = "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"
    
    # 创建工具列表
    search_tool = TavilySearchResults(max_results=2)
    document_export_tool = create_document_export_tool()
    document_reader_tool = create_document_reader_tool()
    tools = [search_tool, document_export_tool, document_reader_tool]

    prompt ="""
    你是一名智能机器人助手。你有以下能力：
    
    🔍 信息获取：
    1. 搜索引擎：搜索最新的网络信息
    2. 文档读取：读取本地文件内容（支持 txt, md, py, js, json 等多种格式）
    
    📝 内容处理：
    3. 文档导出：将内容保存为 Markdown 格式文档
    
    💡 使用指南：
    - 当用户询问本地文件内容时，使用文档读取工具
    - 当用户需要网络搜索时，使用搜索工具
    - 当用户要求保存、导出、整理到文档时，使用文档导出工具
    - 可以组合使用：读取本地文档 → 分析内容 → 搜索相关信息 → 导出综合报告
    
    📄 文档读取支持：
    - 单个文件：直接提供文件路径
    - 目录：提供目录路径，会读取所有支持的文件
    - 通配符：如 "*.py" 匹配所有Python文件
    
    📤 文档导出格式：
    - 带标题：使用 "标题|内容" 格式
    - 无标题：直接传入内容
    
    你可以智能判断用户需求，主动使用合适的工具组合。
    """.strip()

    model = ChatOpenAI(
        model = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        api_key = 'ms-15b6023d-3719-4505-ac95-ebffd78deec5',
        base_url = 'https://api-inference.modelscope.cn/v1/'
    )

    abot = Agent(model, tools, prompt)

    print("🤖 AI助手已启动！输入 'quit' 或 'exit'或者'q' 退出对话\n")

    state = {"messages":[]}
    while True:
        try:
            user_input = input("👤 用户: ").strip()

            if user_input.lower() in ['quit', 'exit','q']:
                print("👋 再见！")
                break
            if not user_input:
                continue

            state["messages"].append(HumanMessage(content=user_input))

            result = abot.graph.invoke(state)

            ai_res = result['messages'][-1].content
            print(f"🤖 AI: {ai_res}")

            state["messages"] = result['messages']
        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重试或输入 'quit' 退出")