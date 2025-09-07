from .MessageManager import MessagerManager
from .MultiAgentState import MultiAgentState
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, List, Dict, Any
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


class BaseAgent:
    """
    智能体基类
    """
    def __init__(self, name: str, role: str, system_prompt: str, model, tools: List = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.tools = {t.name: t for t in (tools or [])}
        checkpointer = InMemorySaver()
        self.message_manager = MessagerManager(max_woking_memory=50, max_history=200)
        # for name in tools:
        #     print(name)    # 输出工具名称
        
        # 如果有工具，绑定到模型
        if tools:
            self.model = self.model.bind_tools(tools)
    
    def get_context(self, state: MultiAgentState) -> str:
        """获取智能体的上下文信息"""
        # 使用 MessageManager 智能管理消息
        all_messages = state["messages"]
        if len(all_messages) > 100:  # 只有消息较多时才使用 MessageManager
            print("使用 MessageManager 智能管理消息")
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
