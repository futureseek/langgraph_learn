# from openai import OpenAI
# import re

# client = OpenAI(
#     base_url='http://localhost:11434/v1',
#     api_key='ollama',  # required, but unused
# )

# response = client.chat.completions.create(
#     model="qwen3:1.7b",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你好，你是谁？"},
#         # {"role": "assistant", "content": "The LA Dodgers won in 2020."},
#         # {"role": "user", "content": "Where was it played?"}
#     ]
# )

# # 正确获取返回内容
# content = response.choices[0].message.content

# # 去掉 <think> 标签
# clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

# print(clean_content)

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
# from tools.clean_think import clean_response  # 刚才写的工具
chat1 = ChatOpenAI(
    model='qwen3:1.7b',
    base_url='http://localhost:11434/v1',
    api_key='ollama',
    extra_body={
        "enable_thinking": False
    },
    streaming=True
    # model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    # api_key='ms-15b6023d-3719-4505-ac95-ebffd78deec5',
    # base_url='https://api-inference.modelscope.cn/v1/',
    # streaming=True
)
response = chat1.invoke([HumanMessage(content="你好啊, AI小助手/no_think")])
print("\n使用 qwen3:1.7b 模型")
print(response)
print("\n")
print("type", type(response))


# print("\n清理后的输出:")
# print(clean_response(response))


# chat2 = ChatOpenAI(
#     model='jingyaogong/minimind2:latest',
#     base_url='http://localhost:11434/v1',
#     api_key='ollama',
#     streaming=True
# )
# response = chat2.invoke([HumanMessage(content="你好啊, AI小助手")])
# print("\n使用 jingyaogong/minimind2:latest 模型")
# print(response)
# print("type", type(response))

# # 正确获取返回内容
# content = response.choices[0].message.content

# # 去掉 <think> 标签
# clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

# print(clean_content)
