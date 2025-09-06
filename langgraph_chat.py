import os
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from agents.BaseAgent import BaseAgent,TaskEvaluatorAgent,TaskExecutorAgent,TaskPlannerAgent,HandlerAgent
from agents.MessageManager import MessagerManager
from agents.MultiAgentState import MultiAgentState
from tools.TavilySearcher import create_tavily_search_reader_tool
from tools.document_exporter import create_document_export_tool
from tools.DocumentReader import create_document_reader_tool
from tools.Path_Acquire import create_path_acquire_tool


# 多智能体管理器
class MultiAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        
        # 创建四个专门化智能体
        self.planner = TaskPlannerAgent(model, tools) 
        self.executor = TaskExecutorAgent(model, tools) 
        self.evaluator = TaskEvaluatorAgent(model, [])
        self.handler = HandlerAgent(model) 
        
        self.message_manager = MessagerManager(max_woking_memory=100, max_history=500)
        
        # 创建检查点保存器实现记忆功能
        self.checkpointer = InMemorySaver()
        
        # 构建多智能体工作流图
        self.graph = self._build_workflow()
    
    def _build_workflow(self):
        """构建散状分发工作流 - 以handler为中心的分发架构"""
        workflow = StateGraph(MultiAgentState)
        
        # 添加中央分发器节点
        workflow.add_node("handler", self._handler_node)
        
        # 添加功能节点
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        
        # 设置散状分发流程 - 所有节点都通过handler进行分发
        workflow.set_entry_point("handler")
        
        # handler根据状态分发到不同节点
        workflow.add_conditional_edges(
            "handler",
            self._handler_decision,
            {
                "planner": "planner",
                "executor": "executor", 
                "evaluator": "evaluator",
                "tool_execution": "tool_execution",
                "END": END
            }
        )
        
        # 所有功能节点执行完成后都返回handler进行下一步分发
        workflow.add_edge("planner", "handler")
        workflow.add_edge("executor", "handler")
        workflow.add_edge("evaluator", "handler")
        workflow.add_edge("tool_execution", "handler")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _handler_node(self, state: MultiAgentState) -> Dict:
        """Handler节点 - 使用HandlerAgent进行流程控制"""
        print(f"\n🎯 {self.handler.name} 分析当前状态...")
        self._diagnose_state(state, "Handler")
        print(f"📊 Handler输入状态调试:")
        print(f"   - step: {state.get('step', 'None')}")
        print(f"   - current_agent: {state.get('current_agent', 'None')}")
        print(f"   - messages数量: {len(state.get('messages', []))}")
        print(f"   - 最后一条消息类型: {type(state.get('messages', [])[-1]).__name__ if state.get('messages') else 'None'}")
        
        try:
            result = self.handler.process(state)
            response = result["response"]
            
            print(f"✅ Handler处理成功")
            print(f"🤖 Handler原始响应: '{response.content}'")
            
            # 提取Handler的决策结果并存储到实例变量中
            next_node = response.content.strip()
            self._current_decision = next_node  # 存储决策结果
            
            # 添加循环检测
            if not hasattr(self, '_decision_history'):
                self._decision_history = []
            self._decision_history.append((next_node, state.get("step", "")))
            
            # 检查是否出现循环
            if len(self._decision_history) > 20:
                self._decision_history = self._decision_history[-10:]  # 保持最近10次决策
                
            # 检测严重循环模式（更宽松的条件）
            recent_decisions = [d[0] for d in self._decision_history[-6:]]
            if len(recent_decisions) >= 6:
                # 只有在完全相同的模式重复3次以上才认为是循环
                pattern_counts = {}
                for i in range(len(recent_decisions) - 1):
                    pattern = f"{recent_decisions[i]}->{recent_decisions[i+1]}"
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                max_repeats = max(pattern_counts.values()) if pattern_counts else 0
                if max_repeats >= 3:
                    print(f"⚠️ 检测到严重循环模式 (重复{max_repeats}次)，强制转入评估阶段")
                    self._current_decision = "evaluator"
            
            # 更新状态
            state["messages"].append(response)
            state["current_agent"] = self.handler.name
            state["agent_history"].append(result["agent_record"])
            state["handler_decision"] = f"Handler决策: {next_node}"
            
            print(f"🔀 {self.handler.name} 决策: 下一个节点 -> {next_node}")
            print(f"📊 当前状态: step={state.get('step')}, execution_count={state.get('execution_count', 0)}")
            
        except Exception as e:
            print(f"❌ Handler处理失败: {str(e)}")
            # 设置默认决策
            self._current_decision = "END"
            state["handler_decision"] = f"Handler错误: {str(e)}"
            
        return state
    
    def _handler_decision(self, state: MultiAgentState) -> str:
        """获取Handler的决策结果"""
        decision = getattr(self, '_current_decision', 'END')
        
        # 验证节点名称的有效性
        valid_nodes = ["planner", "executor", "evaluator", "tool_execution", "END"]
        if decision not in valid_nodes:
            print(f"⚠️ 无效的节点名称: {decision}, 默认结束")
            return "END"
        
        # 额外的安全检查：避免循环调用
        step = state.get("step", "")
        
        # 检查是否需要工具执行（但不强制覆盖Handler的决策）
        if decision == "tool_execution":
            needs_execution = self._needs_tool_execution(state)
            print(f"🔍 工具执行检查: needs_execution={needs_execution}")
            if state.get("messages"):
                # 查找最近的AIMessage
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage):
                        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                        print(f"🔍 最近AIMessage工具调用: {has_tool_calls}, 类型: {type(msg).__name__}")
                        if has_tool_calls:
                            print(f"🔍 工具调用详情: {msg.tool_calls}")
                        break
            
            # 注释掉强制转换，让Handler的决策生效
            # if not needs_execution:
            #     print("⚠️ 无需工具执行，转入评估阶段")
            #     return "evaluator"
        
        # 检查是否需要继续执行
        if decision == "executor" and step == "tool_execution_complete":
            if not self._should_continue_execution(state):
                print("⚠️ 无需继续执行，转入评估阶段")
                return "evaluator"
        
        return decision
    
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
        print(f"📊 Executor输入状态调试:")
        print(f"   - step: {state.get('step', 'None')}")
        print(f"   - current_agent: {state.get('current_agent', 'None')}")
        print(f"   - messages数量: {len(state.get('messages', []))}")
        print(f"   - execution_count: {state.get('execution_count', 0)}")
        
        # 更新执行计数
        state["execution_count"] = state.get("execution_count", 0) + 1
        
        # 确定下一个要执行的工具调用
        next_tool_call = self._determine_next_tool_call(state)
        executed_calls = state.get("executed_tool_calls", [])
        
        print(f"🔍 工具调用状态:")
        print(f"   - 下一个工具调用: {next_tool_call}")
        print(f"   - 已执行调用数: {len(executed_calls)}")
        
        try:
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
                
                print(f"📝 发送给Executor的指导消息长度: {len(guided_context)}")
                
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
                
                print(f"📝 发送给Executor的总结消息长度: {len(summary_context)}")
                
                result = self.executor.process(temp_state)
                response = result["response"]
                
                # 添加指导消息到状态
                state["messages"].append(guidance_message)
            
            print(f"✅ Executor处理成功")
            
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
            
        except Exception as e:
            print(f"❌ Executor处理失败: {str(e)}")
            # 设置默认执行结果
            state["messages"].append(AIMessage(content=f"执行失败: {str(e)}"))
            state["current_agent"] = self.executor.name
            state["execution_result"] = f"执行失败: {str(e)}"
            state["step"] = "execution_complete"
            
        return state
    
    def _tool_execution_node(self, state: MultiAgentState) -> Dict:
        """工具执行节点"""
        print(f"\n🛠️ 执行工具调用...")
        
        # 从后往前查找最近的包含工具调用的AIMessage
        ai_message_with_tools = None
        print(f"🔍 查找工具调用，消息总数: {len(state['messages'])}")
        
        for i, message in enumerate(reversed(state["messages"])):
            msg_index = len(state["messages"]) - 1 - i
            print(f"🔍 检查消息[{msg_index}]: {type(message).__name__}")
            
            if isinstance(message, AIMessage):
                has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                print(f"🔍 AIMessage[{msg_index}] 有工具调用: {has_tool_calls}")
                
                if has_tool_calls:
                    print(f"🔍 工具调用内容: {message.tool_calls}")
                    ai_message_with_tools = message
                    break
        
        if ai_message_with_tools:
            tool_results = []
            executed_tools = state.get("executed_tools", [])
            executed_tool_calls = state.get("executed_tool_calls", [])
            
            for tool_call in ai_message_with_tools.tool_calls:
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
                        # 处理工具调用参数格式问题
                        args = tool_call['args']
                        print(f"🔍 原始参数: {args}")
                        
                        # 如果参数是 {'__arg1': value} 格式，需要转换
                        if len(args) == 1 and '__arg1' in args:
                            tool_name = tool_call['name']
                            if tool_name == 'get_current_directory':
                                args = {'method': args['__arg1']}
                            elif tool_name == 'tavily_search_reader':
                                args = {'query': args['__arg1']}
                            elif tool_name == 'document_reader':
                                args = {'file_path': args['__arg1']}
                            elif tool_name == 'export_document':
                                args = {'content': args['__arg1']}
                            print(f"🔧 转换后参数: {args}")
                        
                        result = self.executor.tools[tool_call['name']].invoke(args)
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
            state["current_tool_call_index"] = state.get("current_tool_call_index", 0) + len(ai_message_with_tools.tool_calls)
            
            # 使用 MessageManager 智能管理消息添加
            current_messages = state["messages"]
            managed_messages = self.message_manager(current_messages, tool_results)
            state["messages"] = managed_messages
            state["step"] = "tool_execution_complete"
            
            # 显示执行进度
            self._show_tool_call_progress(state)
            
            print(f"🔧 工具执行节点完成，已执行 {len(executed_tool_calls)} 个工具调用")
        else:
            print("⚠️ 未找到需要执行的工具调用")
            state["step"] = "tool_execution_complete"
        
        return state
    
    def _evaluator_node(self, state: MultiAgentState) -> Dict:
        """结果评估节点"""
        print(f"\n🔍 {self.evaluator.name} 开始评估结果...")
        print(f"📊 Evaluator输入状态调试:")
        print(f"   - step: {state.get('step', 'None')}")
        print(f"   - current_agent: {state.get('current_agent', 'None')}")
        print(f"   - messages数量: {len(state.get('messages', []))}")
        print(f"   - execution_result长度: {len(str(state.get('execution_result', '')))}")
        
        try:
            result = self.evaluator.process(state)
            response = result["response"]
            
            print(f"✅ Evaluator处理成功")
            
            # 更新状态
            state["messages"].append(response)
            state["current_agent"] = self.evaluator.name
            state["evaluation_result"] = response.content
            state["step"] = "evaluation_complete"
            state["completed"] = True
            state["agent_history"].append(result["agent_record"])
            
            print(f"📊 评估结果：\n{response.content}")
            
        except Exception as e:
            print(f"❌ Evaluator处理失败: {str(e)}")
            # 设置默认评估结果
            state["messages"].append(AIMessage(content=f"评估失败: {str(e)}"))
            state["current_agent"] = self.evaluator.name
            state["evaluation_result"] = f"评估失败: {str(e)}"
            state["step"] = "evaluation_complete"
            state["completed"] = True
            
        return state
    
    def _extract_planned_tool_calls(self, task_plan: str) -> List[Dict]:
        """从任务计划中提取详细的工具调用信息"""
        import re
        
        print(f"🔍 开始提取工具调用，输入文本长度: {len(task_plan)}")
        
        # 获取所有可用工具的名称
        available_tools = {tool.name: tool for tool in self.tools}
        tool_calls = []
        
        print(f"🔍 可用工具: {list(available_tools.keys())}")
        
        # 方法1：从"工具调用清单："部分提取
        calls_section_pattern = r'工具调用清单[：:]\s*\n((?:- .+\n?)*)'
        calls_match = re.search(calls_section_pattern, task_plan, re.MULTILINE)
        
        if calls_match:
            calls_text = calls_match.group(1)
            print(f"🔍 匹配到工具调用清单: {repr(calls_text)}")
            # 解析每个工具调用 - 修复正则表达式以支持下划线
            call_pattern = r'- ([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)'
            for match in re.finditer(call_pattern, calls_text):
                tool_name = match.group(1)
                params_str = match.group(2)
                
                print(f"🔍 解析工具调用: {tool_name}({params_str})")
                
                if tool_name in available_tools:
                    # 解析参数 - 改进参数解析逻辑
                    params = {}
                    
                    # 处理多种参数格式
                    if '=' in params_str:
                        # 格式: key="value", key2="value2" 或 参数="value"
                        param_pattern = r'([^=]+)="([^"]*)"'
                        for param_match in re.finditer(param_pattern, params_str):
                            param_name = param_match.group(1).strip()
                            param_value = param_match.group(2)
                            # 如果参数名是中文"参数"，根据工具类型转换为正确的英文参数名
                            if param_name == '参数':
                                if tool_name == 'tavily_search_reader':
                                    param_name = 'query'
                                elif tool_name == 'document_reader':
                                    param_name = 'file_path'
                                elif tool_name == 'export_document':
                                    param_name = 'content'
                                elif tool_name == 'get_current_directory':
                                    param_name = 'method'
                            # 处理特定工具的参数名映射
                            elif tool_name == 'get_current_directory' and param_name == 'type':
                                param_name = 'method'
                            params[param_name] = param_value
                    else:
                        print(f"🔍 使用简单格式解析参数: {repr(params_str)}")
                        # 简单格式，可能只有一个参数值
                        # 去掉引号
                        clean_params = params_str.strip().strip('"\'')
                        print(f"🔍 清理后的参数值: {repr(clean_params)}")
                        # 根据工具类型推断参数名
                        if tool_name == 'tavily_search_reader':
                            params['query'] = clean_params
                        elif tool_name == 'document_reader':
                            params['file_path'] = clean_params
                        elif tool_name == 'export_document':
                            params['content'] = clean_params
                        elif tool_name == 'get_current_directory':
                            params['method'] = clean_params
                        print(f"🔍 最终参数: {params}")
                    
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
                        print(f"🔍 使用等号格式解析参数: {repr(params_str)}")
                        # 格式: key="value", key2="value2" 或 参数="value"
                        param_pattern = r'([^=]+)="([^"]*)"'
                        for param_match in re.finditer(param_pattern, params_str):
                            param_name = param_match.group(1).strip()
                            param_value = param_match.group(2)
                            print(f"🔍 解析到参数: {repr(param_name)} = {repr(param_value)}")
                            # 如果参数名是中文"参数"，根据工具类型转换为正确的英文参数名
                            if param_name == '参数':
                                print(f"🔧 检测到中文参数名，工具: {tool_name}")
                                if tool_name == 'tavily_search_reader':
                                    param_name = 'query'
                                elif tool_name == 'document_reader':
                                    param_name = 'file_path'
                                elif tool_name == 'export_document':
                                    param_name = 'content'
                                elif tool_name == 'get_current_directory':
                                    param_name = 'method'
                                print(f"🔧 转换后参数名: {param_name}")
                            # 处理特定工具的参数名映射
                            elif tool_name == 'get_current_directory' and param_name == 'type':
                                param_name = 'method'
                                print(f"🔧 映射参数名: type -> method")
                            params[param_name] = param_value
                    else:
                        # 简单格式处理
                        clean_params = params_str.strip().strip('"\'')
                        if tool_name == 'tavily_search_reader':
                            params['query'] = clean_params
                        elif tool_name == 'document_reader':
                            params['file_path'] = clean_params
                        elif tool_name == 'export_document':
                            params['content'] = clean_params
                        elif tool_name == 'get_current_directory':
                            params['method'] = clean_params
                    
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
        
        # 检查是否有需要处理的工具调用结果
        if state.get("step") == "tool_execution_complete":
            # 如果工具执行完成，但没有更多计划的工具调用，则不需要继续
            print("✅ 工具执行完成，准备转入评估阶段")
            return False
        
        return False
    
    def _needs_tool_execution(self, state: MultiAgentState) -> bool:
        """判断是否需要执行工具"""
        if not state["messages"]:
            return False
        
        # 从后往前查找最近的AIMessage，因为Handler的消息可能在最后
        for i, message in enumerate(reversed(state["messages"])):
            if isinstance(message, AIMessage):
                has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls and len(message.tool_calls) > 0
                print(f"🔍 检查AIMessage[{len(state['messages'])-1-i}]工具调用: {has_tool_calls}")
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print(f"🔍 工具调用详情: {message.tool_calls}")
                    
                # 检查这个工具调用是否已经被执行过
                if has_tool_calls:
                    executed_calls = state.get("executed_tool_calls", [])
                    # 如果有工具调用且未被执行，返回True
                    for tool_call in message.tool_calls:
                        already_executed = any(
                            exec_call.get('call_id') == tool_call['id'] 
                            for exec_call in executed_calls
                        )
                        if not already_executed:
                            print(f"🔍 发现未执行的工具调用: {tool_call['name']}")
                            return True
                    print(f"🔍 所有工具调用已执行")
                    return False
                else:
                    # 如果是第一个AIMessage且没有工具调用，继续查找
                    continue
        
        print(f"🔍 未找到需要执行的工具调用")
        return False
    
    def _diagnose_state(self, state: MultiAgentState, node_name: str):
        """诊断当前状态，帮助调试"""
        print(f"\n🔍 [{node_name}] 状态诊断:")
        print(f"   - step: {state.get('step')}")
        print(f"   - execution_count: {state.get('execution_count', 0)}")
        print(f"   - 计划工具调用: {len(state.get('planned_tool_calls', []))}")
        print(f"   - 已执行工具调用: {len(state.get('executed_tool_calls', []))}")
        print(f"   - 消息数量: {len(state.get('messages', []))}")
        
        # 详细检查最近的AIMessage
        ai_messages = [msg for msg in state.get('messages', []) if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_msg = ai_messages[-1]
            has_tool_calls = hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls
            print(f"   - 最后AIMessage有工具调用: {has_tool_calls}")
            if has_tool_calls:
                print(f"   - 工具调用数量: {len(last_ai_msg.tool_calls)}")
        
        if state.get('messages'):
            last_msg = state['messages'][-1]
            print(f"   - 最后消息类型: {type(last_msg).__name__}")
        print()
    
    def process_query(self, user_query: str, thread_id: str = "default") -> Dict:
        """处理用户查询 - 支持会话记忆"""
        # 重置决策历史，避免跨查询的循环检测干扰
        if hasattr(self, '_decision_history'):
            self._decision_history = []
        
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
        
        # 配置会话记忆
        config = {"configurable": {"thread_id": thread_id}}
        
        # 运行工作流，支持记忆功能
        final_state = initial_state
        for output in self.graph.stream(initial_state, config=config):
            if isinstance(output, dict):
                for node_name, node_state in output.items():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)
        
        return final_state



def run_multi_agent_mode() -> bool:
    """运行多智能体模式""" 
    import uuid
    
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

    # 生成会话ID，实现记忆功能
    session_id = str(uuid.uuid4())[:8]  # 使用短的会话ID
    
    print("🤖 多智能体协作系统已启动！")
    print(f"\n🧠 当前会话ID: {session_id} (支持记忆功能)")
    print("📝 输入 'new' 创建新会话, '查看记忆' 查看对话历史")
    print("输入 'quit' 或 'exit' 退出对话\n")

    while True:
        try:
            user_input = input(f"👤 用户({session_id[:4]}): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            if not user_input:
                continue
                
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

            print(f"\n{'='*60}")
            print(f"🚀 开始处理任务: {user_input}")
            print(f"{'='*60}")

            # 处理用户查询，传入会话ID
            final_state = multi_agent.process_query(user_input, session_id)

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
