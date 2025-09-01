import os
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from datetime import datetime
from langgraph.graph import StateGraph, END
from tools import MessagerManager
from tools.TavilySearcher import create_tavily_search_reader_tool
from tools.document_exporter import create_document_export_tool
from tools.DocumentReader import create_document_reader_tool
from tools.Path_Acquire import create_path_acquire_tool



# 多智能体状态定义
class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], MessagerManager(max_woking_memory=100, max_history=500)]
    current_agent: str
    task_plan: str
    execution_result: str
    evaluation_result: str
    user_query: str
    agent_history: List[Dict[str, Any]]
    step: str
    completed: bool
    execution_count: int
    planned_tools: List[str]
    executed_tools: List[str]
    planned_tool_calls: List[Dict[str, Any]]
    executed_tool_calls: List[Dict[str, Any]]
    current_tool_call_index: int

# 单独的智能体基类
class BaseAgent:
    def __init__(self, name: str, role: str, system_prompt: str, model, tools: List = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.tools = {t.name: t for t in (tools or [])}
        self.message_manager = MessagerManager(max_woking_memory=50, max_history=200)
        for name in tools:
            print(name)
        
        # 如果有工具，绑定到模型
        if tools:
            self.model = self.model.bind_tools(tools)
    
    def get_context(self, state: MultiAgentState) -> str:
        """获取智能体的上下文信息"""
        # 使用 MessageManager 智能管理消息
        all_messages = state["messages"]
        if len(all_messages) > 10:  # 只有消息较多时才使用 MessageManager
            managed_messages = self.message_manager([], all_messages[-20:])  # 从最近20条中智能选择
        else:
            managed_messages = all_messages
        
        recent_messages = []
        for msg in managed_messages:
            if isinstance(msg, HumanMessage):
                recent_messages.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                recent_messages.append(f"AI: {msg.content}")
            elif isinstance(msg, ToolMessage):
                recent_messages.append(f"工具结果: {msg.content}")
        
        context = f"""
                    {self.system_prompt}

                    当前任务: {state.get('user_query', '')}
                    当前步骤: {state.get('step', '')}
                    任务计划: {state.get('task_plan', '')}
                    执行结果: {state.get('execution_result', '')}

                    最近对话:
                    {chr(10).join(recent_messages)}

                    请根据你的角色职责，继续处理当前任务。
                    """
        return context
    
    def process(self, state: MultiAgentState) -> Dict[str, Any]:
        """处理状态并返回结果"""
        context = self.get_context(state)
        
        try:
            response = self.model.invoke([HumanMessage(content=context)])
            
            # 记录智能体活动
            agent_record = {
                "agent": self.name,
                "role": self.role,
                "content": response.content,
                "timestamp": datetime.now().isoformat(),
                "step": state.get("step", "")
            }
            
            return {
                "response": response,
                "agent_record": agent_record
            }
        except Exception as e:
            return {
                "response": AIMessage(content=f"处理失败: {str(e)}"),
                "agent_record": {
                    "agent": self.name,
                    "role": self.role,
                    "content": f"错误: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "step": state.get("step", "")
                }
            }


# 三个专门化的智能体类
class TaskPlannerAgent(BaseAgent):
    def __init__(self, model, available_tools):
        # 获取工具信息但不绑定工具
        tool_descriptions = []
        if available_tools:
            for tool in available_tools:
                tool_descriptions.append(f"- {tool.name}: {getattr(tool, 'description', '工具')}")
        
        tools_info = "\n".join(tool_descriptions) if tool_descriptions else "暂无可用工具"
        
        super().__init__(
            name="TaskPlanner",
            role="任务拆解专家",
            system_prompt=f"""
你是一名任务拆解专家。你的职责是：

🎯 核心任务：
1. 分析用户的请求和需求
2. 将复杂任务拆解为具体的执行步骤
3. 详细规划每个工具调用的参数
4. 制定清晰的执行计划

🔧 可用工具：
{tools_info}

📋 工作流程：
- 理解用户意图和目标
- 识别任务的复杂度和类型
- 分解为可执行的子任务
- 为每个步骤指定具体的工具调用和参数
- 预估执行顺序和依赖关系

💡 输出格式：
请按以下格式输出任务计划：
```
任务分析：[对用户需求的理解]

执行步骤：
1. [步骤描述] 
   工具调用：工具名称(参数="具体参数值")
2. [步骤描述]
   工具调用：工具名称(参数="具体参数值")
3. [步骤描述]
   工具调用：工具名称(参数="具体参数值")
...

工具调用清单：
- 工具名称(参数="参数值1")
- 工具名称(参数="参数值2")
- 工具名称(参数="参数值3")

预期结果：[期望达成的目标]
```

⚠️ 重要提示：
1. 同一个工具可以多次调用，使用不同的参数
2. 每次工具调用都要明确指定参数值
3. 搜索时要针对不同方面使用不同的关键词
4. 确保工具调用的参数格式正确
5. 只制定计划，不要实际调用工具

注意：你只负责制定计划，不执行具体操作。
            """,
            model=model,
            tools=[]  # 不绑定工具，只制定计划
        )


class TaskExecutorAgent(BaseAgent):
    def __init__(self, model, tools):
        super().__init__(
            name="TaskExecutor", 
            role="任务执行专家",
            system_prompt="""
你是一名任务执行专家。你的职责是：

⚡ 核心任务：
1. 根据TaskPlanner的计划执行具体操作
2. 调用相应的工具完成任务
3. 处理执行过程中的问题
4. 收集和整理执行结果

🔧 工作原则：
- 严格按照计划执行，不偏离既定步骤
- 主动使用工具获取信息或处理数据
- 遇到问题时尝试解决或报告具体错误
- 详细记录执行过程和结果

📊 可用工具：
- 搜索引擎：获取最新信息
- 文档读取：读取本地文件
- 文档导出：保存处理结果

💼 执行策略：
- 按步骤顺序执行
- 每个步骤完成后确认结果
- 如需调用工具，立即执行
- 整理最终执行结果

注意：专注执行，不重新制定计划。
            """,
            model=model,
            tools=tools
        )


class TaskEvaluatorAgent(BaseAgent):
    def __init__(self, model, tools):
        super().__init__(
            name="TaskEvaluator",
            role="结果评估专家", 
            system_prompt="""
你是一名结果评估专家。你的职责是：

🔍 核心任务：
1. 评估TaskExecutor的执行结果
2. 检查是否完成了用户的原始需求
3. 识别潜在问题和改进空间
4. 提供质量评估和建议

📈 评估维度：
- 完整性：是否完全满足用户需求
- 准确性：信息和结果是否正确
- 质量：输出内容的质量如何
- 效率：执行过程是否合理

✅ 评估流程：
1. 对比原始需求和执行结果
2. 检查每个步骤的完成情况
3. 验证工具使用的合理性
4. 评估最终输出的价值

📋 输出格式：
```
评估结果：[通过/需要改进/失败]
完成度：[百分比]
质量评分：[1-10分]
主要成果：[列出关键成果]
存在问题：[如有问题，详细说明]
改进建议：[具体的改进建议]
```

注意：客观评估，不执行新任务。
            """,
            model=model,
            tools=[]  # 评估者通常不需要工具
        )


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
        
        return workflow.compile()
    
    def _planner_node(self, state: MultiAgentState) -> Dict:
        """任务规划节点"""
        print(f"\n🎯 {self.planner.name} 开始分析任务...")
        
        result = self.planner.process(state)
        response = result["response"]
        
        # 分析任务计划中需要的工具调用
        planned_tool_calls = self._extract_planned_tool_calls(response.content)
        planned_tools = list(set([call['name'] for call in planned_tool_calls]))  # 去重获取工具名称列表
        
        # 更新状态
        state["messages"].append(response)
        state["current_agent"] = self.planner.name
        state["task_plan"] = response.content
        state["step"] = "planning_complete"
        state["agent_history"].append(result["agent_record"])
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
                        result = self.executor.tools[tool_call['name']].invoke(tool_call['args'])
                        tool_results.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            name=tool_call['name'],
                            content=str(result)
                        ))
                        print(f"✅ 工具 {tool_call['name']} 执行成功")
                        print(f"工具结果: {str(result)[:200]}...")
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
    
    def process_query(self, user_query: str) -> Dict:
        """处理用户查询"""
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
        
        # 运行工作流
        final_state = initial_state
        for output in self.graph.stream(initial_state):
            if isinstance(output, dict):
                final_state.update(output)
        
        return final_state



def run_multi_agent_mode() -> bool:
    """运行多智能体模式""" 
    # 创建工具列表
    search_tool = create_tavily_search_reader_tool()
    document_export_tool = create_document_export_tool()
    document_reader_tool = create_document_reader_tool()
    path_ac_tool = create_path_acquire_tool()
    tools = [search_tool,document_export_tool, document_reader_tool,path_ac_tool]

    model = ChatOpenAI(
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        api_key='ms-15b6023d-3719-4505-ac95-ebffd78deec5',
        base_url='https://api-inference.modelscope.cn/v1/'
    )

    # 创建多智能体系统
    multi_agent = MultiAgent(model, tools)

    print("🤖 多智能体协作系统已启动！")
    print("📋 系统包含三个专门化智能体：")
    print("   🎯 TaskPlanner - 任务拆解专家")
    print("   ⚡ TaskExecutor - 任务执行专家") 
    print("   🔍 TaskEvaluator - 结果评估专家")
    print("\n输入 'quit' 或 'exit' 退出对话\n")

    while True:
        try:
            user_input = input("👤 用户: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            if not user_input:
                continue

            print(f"\n{'='*60}")
            print(f"🚀 开始处理任务: {user_input}")
            print(f"{'='*60}")

            # 处理用户查询
            final_state = multi_agent.process_query(user_input)

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
