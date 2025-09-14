import re
from langchain_core.messages import AIMessage

def clean_response(response):
    if isinstance(response, AIMessage):  
        # 去掉 <think> 标签里的内容
        response.content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        return response  # 直接返回修改后的 AIMessage 对象

    elif isinstance(response, str):
        # 如果是字符串，清理后返回字符串
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    else:
        raise TypeError(f"Unsupported response type: {type(response)}")
