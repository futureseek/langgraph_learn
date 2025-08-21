from openai import OpenAI
import re

client = OpenAI(
    api_key = 'ms-15b6023d-3719-4505-ac95-ebffd78deec5',
    base_url = 'https://api-inference.modelscope.cn/v1/'
)

class Agent:
    """
    A class representing an agent
    system : 设定
    messages : 上下文
    """

    def __init__(self, system=""):
        """
        Initialize the Agent with the system.

        """
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self , message):
        """
        Call the agent with a message and return the response.

        """
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    def execute(self):
        """
        Execute the agent's logic to generate a response.

        """
        response = client.chat.completions.create(
            model = 'Qwen/Qwen3-Coder-480B-A35B-Instruct',
            messages=self.messages
        )
        return response.choices[0].message.content

def calculate(what):
    try:
        return str(eval(what))
    except Exception as e:
        return f"计算错误: {str(e)}"

tools_list = {
    "calculate": calculate
}

prompt = """
You are an intelligent assistant that follows the ReAct framework. You think, act, and observe in cycles.
 
Follow this format strictly:
 
Thought: Your reasoning about the problem
Action: calculate["expression"]  # or other available tools
Observation: Result of the action
Result: Final answer to the user's question
 
Available tools:
- calculate: Evaluates mathematical expressions
 
Example:
User Question: What is 4*7+3?
Thought: I need to calculate 4*7+3
Action: calculate["4*7+3"]
Observation: 31
Result: The result of 4*7+3 is 31.
""".strip()


action_re = re.compile(r'^Action:\s*(\w+)\["(.*)"\]$')

def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(f"=== Turn {i} ===")
        print(f"Response:\n{result}\n")
        
        # 解析Action行
        lines = result.split('\n')
        action_match = None
        for line in lines:
            action_match = action_re.match(line.strip())
            if action_match:
                break
        
        if action_match:
            # 有action需要执行
            action_name, action_input = action_match.groups()
            if action_name not in tools_list:
                raise Exception(f"Unknown action: {action_name}")
            
            print(f"Executing: {action_name}[\"{action_input}\"]")
            observation = tools_list[action_name](action_input)
            print(f"Observation: {observation}\n")
            next_prompt = f"Observation: {observation}"
        else:
            # 没有更多action，返回最终结果
            return result


test_question = """
User Question: The result of expression 4*7+3
""".strip()

query(test_question)