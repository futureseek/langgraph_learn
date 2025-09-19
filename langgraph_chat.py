import os
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from agents.BaseAgent import BaseAgent,TaskEvaluatorAgent,TaskExecutorAgent,TaskPlannerAgent
from agents.MessageManager import MessagerManager
from agents.MultiAgentState import MultiAgentState
from tools.TavilySearcher import create_tavily_search_reader_tool
from tools.document_exporter import create_document_export_tool
from tools.DocumentReader import create_document_reader_tool
from tools.Path_Acquire import create_path_acquire_tool
<<<<<<< HEAD

=======
from tools.RAGRetriever import create_rag_tools
from tools.clean_think import clean_response
>>>>>>> ollama_use_meta_chunk

# 多智能体管理器
class MultiAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        
        # 创建三个专门化智能体
        self.planner = TaskPlannerAgent(model, tools)  # 传递工具信息但不绑定
        self.executor = TaskExecutorAgent(model, tools) 
        self.evaluator = TaskEvaluatorAgent(model, [])
        
        self.message_manager = MessagerManager(max_woking_memory=100, max_history=500)
        
<<<<<<< HEAD
        # 创建检查点保存器实现记忆功能
        self.checkpointer = InMemorySaver()
        
=======
>>>>>>> ollama_use_meta_chunk
        # 构建多智能体工作流图
        self.graph = self._build_workflow()
    
    def _build_workflow(self):
        """构建多智能体工作流 - 支持多工具循环执行"""
        workflow = StateGraph(MultiAgentState)
        
        # 添加智能体节点
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        
        # 设置工作流程 - 支持循环执行
        workflow.add_edge("planner", "executor")
        
        # 执行者可能需要工具调用
        workflow.add_conditional_edges(
            "executor",
            self._needs_tool_execution,
            {
                True: "tool_execution",
                False: "evaluator"
            }
        )
        
        # 工具执行后，检查是否需要继续执行更多工具
        workflow.add_conditional_edges(
            "tool_execution",
            self._should_continue_execution,
            {
                True: "executor",      # 继续执行下一个工具
                False: "evaluator"     # 所有工具执行完成，进入评估
            }
        )
        
        workflow.add_edge("evaluator", END)
        
        # 设置入口点
        workflow.set_entry_point("planner")
        
<<<<<<< HEAD
        return workflow.compile(checkpointer=self.checkpointer)
=======
        return workflow.compile()
>>>>>>> ollama_use_meta_chunk
    
    def _planner_node(self, state: MultiAgentState) -> Dict:
        """任务规划节点"""
        print(f"\n🎯 {self.planner.name} 开始分析任务...")
        
<<<<<<< HEAD
        result = self.planner.process(state)
        response = result["response"]
        
        # 分析任务计划中需要的工具调用
        planned_tool_calls = self._extract_planned_tool_calls(response.content)
        planned_tools = list(set([call['name'] for call in planned_tool_calls]))  # 去重获取工具名称列表
=======
        # 获取用户查询
        user_query = state.get("user_query", "")
        
        # 直接检查是否是工具调用
        available_tools = {tool.name: tool for tool in self.tools}
        direct_tool_call = None
        
        # 检查用户查询是否直接是工具名称
        if user_query.strip() in available_tools:
            direct_tool_call = {
                'name': user_query.strip(),
                'params': {},
                'step': 1
            }
        else:
            # 检查是否包含工具名称
            for tool_name in available_tools.keys():
                if tool_name in user_query:
                    # 尝试提取参数
                    params = {}
                    tool_part = user_query.split(tool_name, 1)[1].strip() # 获取工具名后的部分
                    if tool_part:
                        # 简单处理：如果后面跟的是文件路径或简单字符串，作为主要参数
                        # 这里可以根据工具的具体需求进行更复杂的解析
                        # 例如，对于 add_document_to_rag，期望一个 file_path 参数
                        # 对于 rag_question_answer，期望一个 question 参数
                        # 我们可以做一个通用的处理，把剩余部分作为 'input' 参数
                        # 或者根据工具名做特定处理
                        if tool_name == "rag_question_answer":
                             # 对于 rag_question_answer，剩余部分是问题
                             params["question"] = tool_part
                        elif tool_name in ["add_document_to_rag", "add_directory_to_rag", "delete_rag_document"]:
                            # 对于这些工具，剩余部分很可能是路径
                            # 移除可能的引号
                            path = tool_part.strip('\'"')
                            params["file_path"] = path # 注意：add_directory_to_rag 实际需要 directory_path, 这里简化处理或需要更精确的映射
                        elif tool_name == "export_document":
                            # 对于 export_document，剩余部分可能是内容
                            params["content"] = tool_part
                        else:
                            # 通用处理，将剩余部分作为 'input' 参数
                            params["input"] = tool_part
                    
                    direct_tool_call = {
                        'name': tool_name,
                        'params': params, # 使用解析出的参数
                        'step': 1
                    }
                    break
        
        if direct_tool_call:
            # 直接工具调用，跳过AI规划
            response_content = f"直接调用工具: {direct_tool_call['name']}"
            response = AIMessage(content=response_content)
            planned_tool_calls = [direct_tool_call]
            planned_tools = [direct_tool_call['name']]
        else:
            # 正常AI规划流程
            result = self.planner.process(state)
            response = result["response"]
            # 输出加工
            clean_response(response)
            # 分析任务计划中需要的工具调用
            planned_tool_calls = self._extract_planned_tool_calls(response.content)
            planned_tools = list(set([call['name'] for call in planned_tool_calls]))  # 去重获取工具名称列表
>>>>>>> ollama_use_meta_chunk
        
        # 更新状态
        state["messages"].append(response)
        state["current_agent"] = self.planner.name
        state["task_plan"] = response.content
        state["step"] = "planning_complete"
<<<<<<< HEAD
        state["agent_history"].append(result["agent_record"])
=======
        if not direct_tool_call:
            state["agent_history"].append(result["agent_record"])
>>>>>>> ollama_use_meta_chunk
        state["planned_tools"] = planned_tools
        state["executed_tools"] = []
        state["planned_tool_calls"] = planned_tool_calls
        state["executed_tool_calls"] = []
        state["execution_count"] = 0
        state["current_tool_call_index"] = 0
        
        print(f"📋 任务计划：\n{response.content}")
        print(f"🛠️ 计划使用的工具：{planned_tools}")
        return state
    
    def _executor_node(self, state: MultiAgentState) -> Dict:
        """任务执行节点"""
        print(f"\n⚡ {self.executor.name} 开始执行任务...")
        
        # 更新执行计数
        state["execution_count"] = state.get("execution_count", 0) + 1
        
        # 确定下一个要执行的工具调用
        next_tool_call = self._determine_next_tool_call(state)
        executed_calls = state.get("executed_tool_calls", [])
        
        # 为执行者提供明确的指导
        if next_tool_call:
            # 格式化参数字符串
            params_str = ", ".join([f'{k}="{v}"' for k, v in next_tool_call['params'].items()])
            
            guided_context = f"""
根据任务计划，你需要按顺序执行以下工具调用：

当前状态：
- 已执行的工具调用：{len(executed_calls)} 个
- 下一个要执行的工具调用：{next_tool_call['name']}({params_str})

重要指示：
1. 现在只执行这一个工具调用：{next_tool_call['name']}
2. 使用以下参数：{next_tool_call['params']}
3. 不要同时调用多个工具
4. 专注于当前步骤，执行完成后系统会自动让你继续下一步

请立即调用 {next_tool_call['name']} 工具，使用指定的参数。
"""
            # 创建指导消息
            guidance_message = HumanMessage(content=guided_context)
            temp_state = state.copy()
            temp_state["messages"] = state["messages"] + [guidance_message]
            
            result = self.executor.process(temp_state)
            response = result["response"]
            
            # 添加指导消息到状态
            state["messages"].append(guidance_message)
        else:
            # 没有更多工具调用需要执行，让执行者总结当前状态
            executed_calls = state.get("executed_tool_calls", [])
            planned_calls = state.get("planned_tool_calls", [])
            
            summary_context = f"""
所有计划的工具调用都已执行完成。

执行总结：
- 计划的工具调用：{len(planned_calls)} 个
- 已执行的工具调用：{len(executed_calls)} 个

请总结执行结果，为评估阶段做准备。
"""
            guidance_message = HumanMessage(content=summary_context)
            temp_state = state.copy()
            temp_state["messages"] = state["messages"] + [guidance_message]
            
            result = self.executor.process(temp_state)
            response = result["response"]
<<<<<<< HEAD
=======
            # 输出加工
            clean_response(response)
>>>>>>> ollama_use_meta_chunk
            
            # 添加指导消息到状态
            state["messages"].append(guidance_message)
        
        # 更新状态
        state["messages"].append(response)
        state["current_agent"] = self.executor.name
        
        # 累积执行结果
        if state.get("execution_result"):
            state["execution_result"] += f"\n\n步骤 {state['execution_count']}:\n{response.content}"
        else:
            state["execution_result"] = response.content
            
        state["step"] = "execution_complete"
        state["agent_history"].append(result["agent_record"])
        
        print(f"🔧 执行过程：\n{response.content}")
        return state
    
    def _tool_execution_node(self, state: MultiAgentState) -> Dict:
        """工具执行节点"""
        print(f"\n🛠️ 执行工具调用...")
        
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_results = []
            executed_tools = state.get("executed_tools", [])
            executed_tool_calls = state.get("executed_tool_calls", [])
            
            for tool_call in last_message.tool_calls:
                print(f"调用工具: {tool_call['name']} 参数: {tool_call['args']}")
                
                # 记录已执行的工具名称（去重）
                if tool_call['name'] not in executed_tools:
                    executed_tools.append(tool_call['name'])
                
                # 记录详细的工具调用信息（不去重，允许同一工具多次调用）
                executed_tool_calls.append({
                    'name': tool_call['name'],
                    'args': tool_call['args'],
                    'call_id': tool_call['id'],
                    'step': len(executed_tool_calls) + 1
                })
                
                if tool_call['name'] in self.executor.tools:
                    try:
<<<<<<< HEAD
                        result = self.executor.tools[tool_call['name']].invoke(tool_call['args'])
=======
                        # 对于无参数的工具，传入空字符串
                        if not tool_call['args'] or tool_call['args'] == {}:
                            result = self.executor.tools[tool_call['name']].invoke("")
                        else:
                            result = self.executor.tools[tool_call['name']].invoke(tool_call['args'])
>>>>>>> ollama_use_meta_chunk
                        tool_results.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            name=tool_call['name'],
                            content=str(result)
                        ))
                        print(f"✅ 工具 {tool_call['name']} 执行成功")
<<<<<<< HEAD
                        print(f"工具结果: {str(result)[:200]}...")
=======
                        # 对于 get_rag_stats 工具，显示完整结果
                        if tool_call['name'] == 'get_rag_stats':
                            print(f"工具结果:\n{str(result)}")
                        else:
                            print(f"工具结果: {str(result)[:500]}...")
>>>>>>> ollama_use_meta_chunk
                    except Exception as e:
                        tool_results.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            name=tool_call['name'],
                            content=f"工具执行失败: {str(e)}"
                        ))
                        print(f"❌ 工具 {tool_call['name']} 执行失败: {str(e)}")
                else:
                    tool_results.append(ToolMessage(
                        tool_call_id=tool_call['id'],
                        name=tool_call['name'],
                        content="未知工具"
                    ))
                    print(f"❌ 未知工具: {tool_call['name']}")
            
            # 更新已执行工具列表和调用列表
            state["executed_tools"] = executed_tools
            state["executed_tool_calls"] = executed_tool_calls
            
            # 更新当前工具调用索引
            state["current_tool_call_index"] = state.get("current_tool_call_index", 0) + len(last_message.tool_calls)
            
            # 使用 MessageManager 智能管理消息添加
            current_messages = state["messages"]
            managed_messages = self.message_manager(current_messages, tool_results)
            state["messages"] = managed_messages
            state["step"] = "tool_execution_complete"
            
            # 显示执行进度
            self._show_tool_call_progress(state)
        
        return state
    
    def _evaluator_node(self, state: MultiAgentState) -> Dict:
        """结果评估节点"""
        print(f"\n🔍 {self.evaluator.name} 开始评估结果...")
        
        result = self.evaluator.process(state)
        response = result["response"]
<<<<<<< HEAD
        
=======
        # 输出加工
        clean_response(response)
>>>>>>> ollama_use_meta_chunk
        # 更新状态
        state["messages"].append(response)
        state["current_agent"] = self.evaluator.name
        state["evaluation_result"] = response.content
        state["step"] = "evaluation_complete"
        state["completed"] = True
        state["agent_history"].append(result["agent_record"])
        
        print(f"📊 评估结果：\n{response.content}")
        return state
    
    def _extract_planned_tool_calls(self, task_plan: str) -> List[Dict]:
        """从任务计划中提取详细的工具调用信息"""
        import re
        
        # 获取所有可用工具的名称
        available_tools = {tool.name: tool for tool in self.tools}
        tool_calls = []
        
        # 方法1：从"工具调用清单："部分提取
        calls_section_pattern = r'工具调用清单[：:]\s*\n((?:- .+\n?)*)'
        calls_match = re.search(calls_section_pattern, task_plan, re.MULTILINE)
        
        if calls_match:
            calls_text = calls_match.group(1)
            # 解析每个工具调用 - 修复正则表达式以支持下划线
            call_pattern = r'- ([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)'
            for match in re.finditer(call_pattern, calls_text):
                tool_name = match.group(1)
                params_str = match.group(2)
                
                if tool_name in available_tools:
                    # 解析参数 - 改进参数解析逻辑
                    params = {}
                    
                    # 处理多种参数格式
                    if '=' in params_str:
                        # 格式: key="value", key2="value2"
                        param_pattern = r'(\w+)="([^"]*)"'
                        for param_match in re.finditer(param_pattern, params_str):
                            param_name = param_match.group(1)
                            param_value = param_match.group(2)
                            params[param_name] = param_value
                    else:
                        # 简单格式，可能只有一个参数值
                        # 去掉引号
                        clean_params = params_str.strip().strip('"\'')
                        # 根据工具类型推断参数名
                        if tool_name == 'tavily_search_results_json':
                            params['query'] = clean_params
                        elif tool_name == 'document_reader':
                            params['file_path'] = clean_params
                        elif tool_name == 'export_document':
                            params['content'] = clean_params
                    
                    tool_calls.append({
                        'name': tool_name,
                        'params': params,
                        'step': len(tool_calls) + 1
                    })
        
        # 方法2：从"工具调用："部分提取（作为备选）
        if not tool_calls:
            call_pattern = r'工具调用[：:]\s*([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)'
            for match in re.finditer(call_pattern, task_plan):
                tool_name = match.group(1)
                params_str = match.group(2)
                
                if tool_name in available_tools:
                    params = {}
                    
                    # 处理多种参数格式
                    if '=' in params_str:
                        param_pattern = r'(\w+)="([^"]*)"'
                        for param_match in re.finditer(param_pattern, params_str):
                            param_name = param_match.group(1)
                            param_value = param_match.group(2)
                            params[param_name] = param_value
                    else:
                        # 简单格式处理
                        clean_params = params_str.strip().strip('"\'')
                        if tool_name == 'tavily_search_results_json':
                            params['query'] = clean_params
                        elif tool_name == 'document_reader':
                            params['file_path'] = clean_params
                        elif tool_name == 'export_document':
                            params['content'] = clean_params
                    
                    tool_calls.append({
                        'name': tool_name,
                        'params': params,
                        'step': len(tool_calls) + 1
                    })
        
<<<<<<< HEAD
=======
        # 方法3：直接匹配用户输入中的工具名称（新增）
        if not tool_calls:
            user_input = task_plan.strip()
            # 检查用户输入是否直接是工具名称
            if user_input in available_tools:
                tool_calls.append({
                    'name': user_input,
                    'params': {},
                    'step': 1
                })
            else:
                # 检查是否包含工具名称
                for tool_name in available_tools.keys():
                    if tool_name in user_input:
                        tool_calls.append({
                            'name': tool_name,
                            'params': {},
                            'step': len(tool_calls) + 1
                        })
                        break
        
>>>>>>> ollama_use_meta_chunk
        print(f"🔍 从任务计划中提取的工具调用: {len(tool_calls)} 个")
        for i, call in enumerate(tool_calls, 1):
            print(f"   {i}. {call['name']}({call['params']})")
        
        return tool_calls
    
    def _show_tool_call_progress(self, state: MultiAgentState):
        """显示工具调用执行进度"""
        planned_calls = state.get("planned_tool_calls", [])
        executed_calls = state.get("executed_tool_calls", [])
        
        if planned_calls:
            print(f"\n📊 工具调用执行进度:")
            for i, call in enumerate(planned_calls, 1):
                status = "✅" if i <= len(executed_calls) else "⏳"
                params_str = ", ".join([f'{k}="{v}"' for k, v in call['params'].items()])
                print(f"   {i}. {call['name']}({params_str}) {status}")
            
            remaining = len(planned_calls) - len(executed_calls)
            if remaining > 0:
                print(f"   剩余 {remaining} 个工具调用待执行")
            else:
                print(f"   🎉 所有工具调用执行完成!")
    
    def _get_executed_tools(self, state: MultiAgentState) -> List[str]:
        """获取已执行的工具列表"""
        executed = []
        for msg in state["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call['name'] not in executed:
                        executed.append(tool_call['name'])
        return executed
    
    def _determine_next_tool_call(self, state: MultiAgentState) -> Dict:
        """确定下一个应该执行的工具调用"""
        planned_calls = state.get("planned_tool_calls", [])
        current_index = state.get("current_tool_call_index", 0)
        
        if current_index < len(planned_calls):
            return planned_calls[current_index]
        
        return None
    
    def _should_continue_execution(self, state: MultiAgentState) -> bool:
        """判断是否需要继续执行任务"""
        # 检查执行次数，避免无限循环
        execution_count = state.get("execution_count", 0)
        if execution_count >= 5:
            print("⚠️ 达到最大执行次数限制，转入评估阶段")
            return False
        
        # 检查是否还有未执行的工具调用
        planned_calls = state.get("planned_tool_calls", [])
        executed_calls = state.get("executed_tool_calls", [])
        
        remaining_calls = len(planned_calls) - len(executed_calls)
        
        if remaining_calls > 0:
            print(f"🔄 还有 {remaining_calls} 个工具调用待执行")
            return True
        
        return False
    
    def _needs_tool_execution(self, state: MultiAgentState) -> bool:
        """判断是否需要执行工具"""
        if not state["messages"]:
            return False
        
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0
        
        return False
    
<<<<<<< HEAD
    def process_query(self, user_query: str, thread_id: str = "default") -> Dict:
        """处理用户查询 - 支持会话记忆"""
=======
    def process_query(self, user_query: str) -> Dict:
        """处理用户查询"""
>>>>>>> ollama_use_meta_chunk
        # 初始化状态
        initial_state = MultiAgentState(
            messages=[HumanMessage(content=user_query)],
            current_agent="",
            task_plan="",
            execution_result="",
            evaluation_result="",
            user_query=user_query,
            agent_history=[],
            step="start",
            completed=False,
            execution_count=0,
            planned_tools=[],
            executed_tools=[],
            planned_tool_calls=[],
            executed_tool_calls=[],
            current_tool_call_index=0
        )
        
<<<<<<< HEAD
        # 配置会话记忆
        config = {"configurable": {"thread_id": thread_id}}
        
        # 运行工作流，支持记忆功能
        final_state = initial_state
        for output in self.graph.stream(initial_state, config=config):
=======
        # 运行工作流
        final_state = initial_state
        for output in self.graph.stream(initial_state):
>>>>>>> ollama_use_meta_chunk
            if isinstance(output, dict):
                final_state.update(output)
        
        return final_state



def run_multi_agent_mode() -> bool:
    """运行多智能体模式""" 
<<<<<<< HEAD
    import uuid
    
    # 创建工具列表
=======
    # 创建工具列表
    print("创建工具列表")
>>>>>>> ollama_use_meta_chunk
    search_tool = create_tavily_search_reader_tool()
    document_export_tool = create_document_export_tool()
    document_reader_tool = create_document_reader_tool()
    path_ac_tool = create_path_acquire_tool()
<<<<<<< HEAD
    tools = [search_tool,document_export_tool, document_reader_tool,path_ac_tool]

    model = ChatOpenAI(
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        api_key='ms-15b6023d-3719-4505-ac95-ebffd78deec5',
        base_url='https://api-inference.modelscope.cn/v1/'
    )

    # 创建多智能体系统
    multi_agent = MultiAgent(model, tools)

    # 生成会话ID，实现记忆功能
    session_id = str(uuid.uuid4())[:8]  # 使用短的会话ID
    
=======
    print("创建工具列表完成")
    # 创建模型实例
    model = ChatOpenAI(
            # model='Qwen/Qwen3-1.7B',
            # base_url='https://api-inference.modelscope.cn/v1',
            # api_key='ms-8b59067c-75ff-4b83-900e-26e00e46c531',
            # streaming=True  # 使用流式调用，可能不需要enable_thinking参数
            
            model='qwen3:1.7b',
            base_url='http://localhost:11434/v1',
            api_key='ollama',
            streaming=True
        )
        
    # 创建RAG工具（传入模型实例）
    print("创建RAG工具")
    rag_tools = create_rag_tools(model=model)
    print("创建RAG工具完成")
    # 合并所有工具
    tools = [search_tool, document_export_tool, document_reader_tool, path_ac_tool] + rag_tools
    
    # 创建多智能体系统
    multi_agent = MultiAgent(model, tools)

>>>>>>> ollama_use_meta_chunk
    print("🤖 多智能体协作系统已启动！")
    print("📋 系统包含三个专门化智能体：")
    print("   🎯 TaskPlanner - 任务拆解专家")
    print("   ⚡ TaskExecutor - 任务执行专家") 
    print("   🔍 TaskEvaluator - 结果评估专家")
<<<<<<< HEAD
    print(f"\n🧠 当前会话ID: {session_id} (支持记忆功能)")
    print("📝 输入 'new' 创建新会话, '查看记忆' 查看对话历史")
    print("输入 'quit' 或 'exit' 退出对话\n")

    while True:
        try:
            user_input = input(f"👤 用户({session_id[:4]}): ").strip()
=======
    print("\n🛠️ 可用工具：")
    print("   🔍 搜索工具 - 网络信息检索")
    print("   📄 文档工具 - 文件读取和导出")
    print("   📁 路径工具 - 文件路径获取")
    print("   🧠 RAG工具 - 智能文档问答系统")
    print("     • add_document_to_rag ./doc/中华人民共和国证券法(2019修订).pdf - 添加文档到知识库")
    print("     • add_directory_to_rag ./docs/ - 批量添加目录文档")
    print("     • rag_question_answer 您的问题 - 基于知识库问答")
    print("     • get_rag_stats - 查看知识库统计")
    print("     • delete_rag_document ./path/to/file.md - 删除指定文档")
    print("     • clear_rag_knowledge_base - 清空整个知识库")
    print("\n输入 'quit' 或 'exit' 退出对话\n")

    while True:
        try:
            user_input = input("👤 用户: ").strip()
>>>>>>> ollama_use_meta_chunk

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            if not user_input:
                continue
<<<<<<< HEAD
                
            # 特殊命令处理
            if user_input.lower() == 'new':
                # 创建新会话
                session_id = str(uuid.uuid4())[:8]
                print(f"🆕 已创建新会话: {session_id}")
                continue
            elif user_input in ['查看记忆', 'memory', 'history']:
                # 查看对话历史
                try:
                    config = {"configurable": {"thread_id": session_id}}
                    history = multi_agent.checkpointer.list(config)
                    if history:
                        print(f"\n📜 会话 {session_id} 的历史记忆:")
                        for i, checkpoint in enumerate(history):
                            print(f"  {i+1}. 检查点 {checkpoint}")
                    else:
                        print(f"\n💭 会话 {session_id} 暂无历史记忆")
                except Exception as e:
                    print(f"\n⚠️ 无法获取历史记忆: {e}")
                continue
=======
>>>>>>> ollama_use_meta_chunk

            print(f"\n{'='*60}")
            print(f"🚀 开始处理任务: {user_input}")
            print(f"{'='*60}")

<<<<<<< HEAD
            # 处理用户查询，传入会话ID
            final_state = multi_agent.process_query(user_input, session_id)
=======
            # 处理用户查询
            final_state = multi_agent.process_query(user_input)
>>>>>>> ollama_use_meta_chunk

            print(f"\n{'='*60}")
            print("✅ 任务处理完成！")
            print(f"{'='*60}")
            
            # 显示最终结果
            if final_state.get("evaluation_result"):
                print(f"\n📊 最终评估：\n{final_state['evaluation_result']}")
            
            # 显示智能体协作历史
            """
            print(f"\n🤝 智能体协作历史：")
            for record in final_state.get("agent_history", []):
                print(f"   {record['agent']} ({record['role']}): {record['content'][:100]}...")
            """
        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重试或输入 'quit' 退出")

    return False


if __name__ == "__main__":
    print("🤖 LangGraph 智能助手系统")
    
    while True:
        if not run_multi_agent_mode():
            break
