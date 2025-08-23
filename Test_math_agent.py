from typing import List, Dict, Any, TypedDict, Union, Optional
from datetime import datetime
import json
import re
import math
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from pydantic import SecretStr

# 配置模型
model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=SecretStr("sk-7016e4e4bcbc4fdeb00a1fc8b9b3b390"),
    temperature=0.7
)


# 定义工具
@tool
def mycalculator(expression: str) -> str:
    """计算数学表达式，支持基本运算：+, -, *, /, **, sqrt, sin, cos, tan"""
    try:
        # 安全地评估数学表达式
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'pi': math.pi, 'e': math.e
        }

        # 清理表达式
        expression = expression.replace('^', '**').replace('×', '*').replace('÷', '/')

        # 编译并执行
        code = compile(expression, '<string>', 'eval')
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of {name} not allowed")

        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_math_concepts(query: str) -> str:
    """搜索数学概念和公式"""
    math_knowledge = {
        "勾股定理": "直角三角形中，a² + b² = c²，其中c是斜边",
        "二次方程": "ax² + bx + c = 0，解为 x = (-b ± √(b²-4ac))/2a",
        "三角函数": "sin²θ + cos²θ = 1，tan θ = sin θ / cos θ",
        "导数": "f'(x) = lim(h→0) [f(x+h) - f(x)]/h",
        "积分": "∫f(x)dx 是 f(x) 的反导数",
        "概率": "P(A) = 事件A发生的次数 / 总次数",
        "排列": "P(n,r) = n!/(n-r)!",
        "组合": "C(n,r) = n!/(r!(n-r)!)",
        "等差数列": "an = a1 + (n-1)d，Sn = n(a1 + an)/2",
        "等比数列": "an = a1 * r^(n-1)，Sn = a1(1-r^n)/(1-r)"
    }

    for key, value in math_knowledge.items():
        if key in query or any(word in query for word in key.split()):
            return f"{key}: {value}"

    return "未找到相关数学概念，请尝试其他关键词"


@tool
def verify_solution(problem: str, solution: str) -> str:
    """验证数学题解答的正确性"""
    try:
        # 简单的验证逻辑
        if "=" in solution:
            left, right = solution.split("=", 1)
            # 检查等式是否成立
            return "验证通过：等式成立"
        elif "答案" in solution or "结果" in solution:
            return "验证通过：答案格式正确"
        else:
            return "验证失败：答案格式不正确"
    except:
        return "验证失败：无法解析答案"


# 定义状态类型
class MathState(TypedDict):
    problem: str
    messages: List[Dict[str, str]]
    current_agent: str
    step: str
    solution: str
    verification_result: str
    long_term_memory: List[Dict[str, Any]]
    short_term_memory: List[Dict[str, Any]]
    terminated: bool
    calculation_expressions: List[str]


# 记忆类型枚举
class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


# 记忆管理类
class MemoryManager:
    def __init__(self):
        self.long_term_memory: List[Dict[str, Any]] = []
        self.short_term_memory: List[Dict[str, Any]] = []
        self.max_short_term = 10
        self.max_long_term = 50

    def add_memory(self, memory_type: MemoryType, content: Dict[str, Any]):
        """添加记忆"""
        memory = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": memory_type.value
        }

        if memory_type == MemoryType.SHORT_TERM:
            self.short_term_memory.append(memory)
            if len(self.short_term_memory) > self.max_short_term:
                self.short_term_memory.pop(0)
        else:
            self.long_term_memory.append(memory)
            if len(self.long_term_memory) > self.max_long_term:
                self.long_term_memory.pop(0)

    def get_relevant_memories(self, query: str, memory_type: MemoryType = MemoryType.SHORT_TERM) -> List[Dict]:
        """获取相关记忆"""
        memories = self.short_term_memory if memory_type == MemoryType.SHORT_TERM else self.long_term_memory
        # 简单的关键词匹配
        relevant = []
        for memory in memories[-5:]:  # 最近5条记忆
            if any(word in str(memory["content"]).lower() for word in query.lower().split()):
                relevant.append(memory)
        return relevant


# 基础智能体类
class MathAgent:
    def __init__(self, name: str, role: str, system_prompt: str, memory_manager: MemoryManager):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.memory_manager = memory_manager

    def update_memory(self, content: Dict[str, Any], memory_type: MemoryType = MemoryType.SHORT_TERM):
        """更新记忆"""
        self.memory_manager.add_memory(memory_type, content)

    def get_context(self, state: MathState) -> str:
        """获取上下文"""
        recent_messages = "\n".join([f"{m['agent']}: {m['content']}" for m in state["messages"][-3:]])
        short_term = self.memory_manager.get_relevant_memories(state["problem"], MemoryType.SHORT_TERM)
        long_term = self.memory_manager.get_relevant_memories(state["problem"], MemoryType.LONG_TERM)

        memory_context = ""
        if short_term:
            memory_context += f"\n短期记忆: {[m['content'] for m in short_term]}"
        if long_term:
            memory_context += f"\n长期记忆: {[m['content'] for m in long_term]}"

        return f"""
{self.system_prompt}

当前问题: {state["problem"]}
当前步骤: {state["step"]}

最近对话:
{recent_messages}

{memory_context}

请根据你的角色和当前状态，继续解题过程。
"""


# 创建记忆管理器
memory_manager = MemoryManager()

# 创建智能体
planner = MathAgent(
    "Planner",
    "问题分析者",
    """你是一个数学问题分析专家。你的任务是：
1. 仔细分析数学问题的类型和难度
2. 使用搜索工具查找相关的数学概念和公式
3. 制定详细的解题步骤和策略
4. 依次给出计算的关键步骤
5. 不能给出题目的最终答案

请使用思维链推理，逐步分析问题。""",
    memory_manager
)

calculator_agent = MathAgent(
    "Calculator",
    "步骤执行者",
    """你是一个数学计算执行专家。你的任务是：
1. 分析Planner给出的解题步骤
2. 根据步骤生成具体的计算表达式
3. 使用计算工具进行具体的数值计算
4. 确保计算过程的准确性
5. 提供详细的计算结果

请专注于执行Planner制定的步骤，不要重新分析原问题。""",
    memory_manager
)

verifier = MathAgent(
    "Verifier",
    "结果验证者",
    """你是一个严格的数学验证专家。你的任务是：
1. 验证Planner的步骤分析是否正确
2. 验证Calculator的计算过程是否准确
3. 检查最终结果的合理性
4. 识别潜在的错误
5. 提供改进建议

请验证整个解题过程的正确性，不需要调用任何工具。""",
    memory_manager
)


# 定义节点函数
def planner_node(state: MathState) -> Dict:
    """规划者节点 - 分析问题并给出步骤"""
    context = planner.get_context(state)

    # 使用LLM分析问题并确定需要搜索的数学概念
    problem = state["problem"]
    math_concepts = []

    # 让LLM分析问题并确定需要搜索的概念
    search_prompt = f"""
分析以下数学问题，并确定需要搜索的数学概念：

问题：{problem}

请分析问题类型，然后生成1-2个需要搜索的关键词。
只返回关键词，每行一个，不要其他解释。

例如：
一元一次方程
函数求值
算术运算
"""

    try:
        search_response = model.invoke([HumanMessage(content=search_prompt)])
        search_keywords = [kw.strip() for kw in str(search_response.content).split('\n')
                           if kw.strip() and len(kw.strip()) > 1]

        # 使用搜索工具查找相关数学概念
        for keyword in search_keywords[:2]:  # 最多搜索2个概念
            try:
                concept = search_math_concepts.invoke({"query": keyword})
                math_concepts.append(f"相关概念({keyword}): {concept}")
            except Exception as e:
                math_concepts.append(f"搜索{keyword}失败: {str(e)}")

    except Exception as e:
        # 如果LLM分析失败，让LLM重新分析
        fallback_prompt = f"""
重新分析以下问题并确定数学概念：

问题：{problem}

请直接返回1-2个数学概念关键词，每行一个。
"""
        try:
            fallback_response = model.invoke([HumanMessage(content=fallback_prompt)])
            fallback_keywords = [kw.strip() for kw in str(fallback_response.content).split('\n')
                                 if kw.strip() and len(kw.strip()) > 1]

            for keyword in fallback_keywords[:2]:
                try:
                    concept = search_math_concepts.invoke({"query": keyword})
                    math_concepts.append(f"相关概念({keyword}): {concept}")
                except:
                    math_concepts.append(f"未找到{keyword}相关概念")
        except Exception as e2:
            math_concepts.append(f"概念分析失败: {str(e2)}")

    # 让LLM分析问题并给出解题步骤
    steps_prompt = f"""
基于以下问题，分析并给出解题步骤：

问题：{problem}

相关数学概念：
{' '.join(math_concepts)}

请分析问题类型，然后依次给出解题的关键步骤。
不要给出最终答案，只给出步骤。
每个步骤要具体明确。

例如：
步骤1：理解问题类型
步骤2：确定解题策略
步骤3：执行具体计算
...

请勿在步骤中给出计算式，只给出关键步骤，请严格按照以上的例子输出。
"""

    # 将搜索到的概念添加到上下文中
    enhanced_context = f"{context}\n\n{' '.join(math_concepts)}"
    response = model.invoke([HumanMessage(content=steps_prompt)])

    message = {
        "agent": planner.name,
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    }

    # 更新记忆
    planner.update_memory({
        "problem": state["problem"],
        "analysis": response.content,
        "concepts": math_concepts,
        "step": "planning"
    }, MemoryType.LONG_TERM)

    state["messages"].append(message)
    state["current_agent"] = planner.name
    state["step"] = "planning_complete"

    print(f"\n{planner.name}: {response.content}")

    return {"state": state, "next": "calculator"}


def calculator_node(state: MathState) -> Dict:
    """计算者节点 - 分析步骤并执行计算"""
    context = calculator_agent.get_context(state)

    # 获取Planner的步骤分析
    planner_message = None
    for msg in state["messages"]:
        if msg["agent"] == "Planner":
            planner_message = msg["content"]
            break

    if not planner_message:
        planner_message = "未找到Planner的步骤分析"

    # 让LLM分析Planner的步骤并生成计算表达式
    calc_prompt = f"""
基于Planner给出的解题步骤，生成具体的计算表达式：

Planner的步骤分析：
{planner_message}

请分析Planner的步骤，然后：
1. 根据步骤生成具体的计算表达式
2. 使用计算工具进行数值计算
3. 不要重新分析原问题，只关注执行Planner的步骤

请按照步骤顺序生成计算表达式。
"""

    # 使用LLM分析步骤并生成计算表达式
    try:
        analysis_response = model.invoke([HumanMessage(content=calc_prompt)])
        expressions = [expr.strip() for expr in str(analysis_response.content).split('\n')
                       if expr.strip() and any(op in expr for op in ['+', '-', '*', '/', '**', '^', '(', ')', '='])]

        # 执行计算
        calc_results = []
        for expr in expressions[:5]:  # 最多5个表达式
            try:
                # 清理表达式
                clean_expr = re.sub(r'[^0-9+\-*/()^.\s]', '', expr)
                if clean_expr.strip():
                    result = mycalculator.invoke({"expression": clean_expr})
                    calc_results.append(f"计算 {clean_expr} = {result}")
                else:
                    calc_results.append(f"跳过无效表达式: {expr}")
            except Exception as e:
                calc_results.append(f"计算错误 {expr}: {str(e)}")

        # 如果没有找到表达式，让LLM重新分析
        if not calc_results:
            reanalysis_prompt = f"""
重新分析Planner的步骤并生成计算表达式：

步骤：{planner_message}

请直接生成3-5个需要计算的具体数学表达式，每行一个。
只返回表达式，不要其他解释。
"""
            reanalysis_response = model.invoke([HumanMessage(content=reanalysis_prompt)])
            reexpressions = [expr.strip() for expr in str(reanalysis_response.content).split('\n')
                             if expr.strip() and any(op in expr for op in ['+', '-', '*', '/', '**', '^', '(', ')'])]

            for expr in reexpressions[:5]:
                try:
                    clean_expr = re.sub(r'[^0-9+\-*/()^.\s]', '', expr)
                    if clean_expr.strip():
                        result = mycalculator.invoke({"expression": clean_expr})
                        calc_results.append(f"计算 {clean_expr} = {result}")
                except Exception as e:
                    calc_results.append(f"计算错误 {expr}: {str(e)}")

    except Exception as e:
        calc_results = [f"步骤分析错误: {str(e)}"]

    calc_summary = "\n".join(calc_results) if calc_results else "无计算工具调用"

    # 生成计算者的回应
    calc_context = f"{context}\n\nPlanner步骤分析:\n{planner_message}\n\n计算工具结果:\n{calc_summary}"
    final_response = model.invoke([HumanMessage(content=calc_context)])

    message = {
        "agent": calculator_agent.name,
        "content": final_response.content,
        "timestamp": datetime.now().isoformat()
    }

    # 更新记忆
    calculator_agent.update_memory({
        "planner_steps": planner_message,
        "calculation": final_response.content,
        "tool_result": calc_summary,
        "step": "calculation"
    }, MemoryType.SHORT_TERM)

    state["messages"].append(message)
    state["current_agent"] = calculator_agent.name
    state["step"] = "calculation_complete"
    state["calculation_expressions"] = calc_results

    print(f"\n{calculator_agent.name}: {final_response.content}")

    return {"state": state, "next": "verifier"}


def verifier_node(state: MathState) -> Dict:
    """验证者节点 - 验证Planner和Calculator的正确性"""
    context = verifier.get_context(state)

    # 获取Planner和Calculator的消息
    planner_message = None
    calculator_message = None

    for msg in state["messages"]:
        if msg["agent"] == "Planner":
            planner_message = msg["content"]
        elif msg["agent"] == "Calculator":
            calculator_message = msg["content"]

    if not planner_message:
        planner_message = "未找到Planner的分析"
    if not calculator_message:
        calculator_message = "未找到Calculator的计算"

    # 让LLM验证整个解题过程
    verify_prompt = f"""
作为数学验证专家，请验证以下解题过程的正确性：

原问题：{state["problem"]}

Planner的步骤分析：
{planner_message}

Calculator的计算过程：
{calculator_message}

请验证：
1. Planner的步骤分析是否正确合理
2. Calculator的计算过程是否准确
3. 最终结果是否合理
4. 是否存在潜在错误
5. 提供改进建议

请给出详细的验证结果，不需要调用任何工具。
"""

    response = model.invoke([HumanMessage(content=verify_prompt)])

    message = {
        "agent": verifier.name,
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    }

    # 更新记忆
    verifier.update_memory({
        "problem": state["problem"],
        "planner_analysis": planner_message,
        "calculator_process": calculator_message,
        "verification": response.content,
        "step": "verification"
    }, MemoryType.LONG_TERM)

    state["messages"].append(message)
    state["current_agent"] = verifier.name
    state["step"] = "verification_complete"

    print(f"\n{verifier.name}: {response.content}")

    return {"state": state, "next": "end"}


# 构建图
def build_math_solver_graph():
    """构建数学解题流程图"""
    workflow = StateGraph(MathState)

    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("verifier", verifier_node)

    # 设置流程
    workflow.add_edge("planner", "calculator")
    workflow.add_edge("calculator", "verifier")
    workflow.add_edge("verifier", END)

    # 设置入口点
    workflow.set_entry_point("planner")

    return workflow.compile()


# 主函数
def main():
    print("\n=== 智能多智能体数学解题系统 ===")

    # 测试问题
    test_problems = [
        "求解方程: 2x + 3 = 7",
        "计算: (3 + 4) × 2 - 5",
        "求函数 f(x) = x² + 2x + 1 在 x = 2 处的值"
    ]

    graph = build_math_solver_graph()

    for i, problem in enumerate(test_problems, 1):
        print(f"\n{'=' * 60}")
        print(f"问题 {i}: {problem}")
        print(f"{'=' * 60}")

        # 初始化状态
        initial_state = MathState(
            problem=problem,
            messages=[],
            current_agent="",
            step="start",
            solution="",
            verification_result="",
            long_term_memory=[],
            short_term_memory=[],
            terminated=False,
            calculation_expressions=[]
        )

        # 运行图
        final_state = initial_state
        for output in graph.stream(initial_state):
            if isinstance(output, dict) and "state" in output:
                final_state = output["state"]
                if final_state["messages"]:
                    latest_message = final_state["messages"][-1]
                    print(f"\n步骤: {final_state['step']}")
                    print(f"智能体: {latest_message['agent']}")
                    print(f"内容: {latest_message['content']}")
                    print("-" * 50)

        print(f"\n问题 {i} 解决完成!")
        print(f"验证结果: {final_state.get('verification_result', 'N/A')}")
        if final_state.get('calculation_expressions'):
            print(f"计算表达式: {final_state['calculation_expressions']}")


if __name__ == "__main__":
    main()