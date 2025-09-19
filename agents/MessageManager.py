from langchain_core.messages import SystemMessage,HumanMessage,AnyMessage,AIMessage,ToolMessage



class MessagerManager:
    """
    分层消息管理器
    """
 
<<<<<<< HEAD
    def __init__(self, max_woking_memory=1000, max_history=5000):
=======
    def __init__(self, max_woking_memory=100, max_history=500):
>>>>>>> ollama_use_meta_chunk
        self.max_woking_memory = max_woking_memory
        self.max_history = max_history
        self.history = []
 
    def __call__(self, current_messages: list, new_messages: list) -> list:
        """
        主要调用接口
        """
        # 添加到历史记录
        self.history.extend(new_messages)
 
        # 保持历史记录长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # 合并当前消息和新消息
        all_messages = current_messages + new_messages
 
        # 准备工作内存
        working_memory = self._prepare_working_memory(all_messages)
        return working_memory
 
    def _prepare_working_memory(self, all_messages: list) -> list:
        """
        准备工作内存 - 保留最重要的消息
        """
        
        # 按类型分类并保持原始顺序
        system_msgs = []
        human_msgs = []
        ai_msgs = []
        tool_msgs = []
        
        # 分类消息但保持顺序信息
        for i, msg in enumerate(all_messages):
            if isinstance(msg, SystemMessage):
                system_msgs.append((i, msg))
            elif isinstance(msg, HumanMessage):
                human_msgs.append((i, msg))
            elif isinstance(msg, AIMessage):
                ai_msgs.append((i, msg))
            elif isinstance(msg, ToolMessage):
                tool_msgs.append((i, msg))
 
        working_memory = []
 
        # 始终保留系统消息（通常只需要最新的一个）
        if system_msgs:
            working_memory.append(system_msgs[-1][1]) 
 
        # 找出AI消息和对应的工具消息配对
        message_pairs = []
        
        # 按时间顺序重组消息
        all_sorted = sorted(
            system_msgs + human_msgs + ai_msgs + tool_msgs, 
            key=lambda x: x[0]
        )
        
        # 保持工具调用链的完整性：AIMessage -> ToolMessage 的顺序
        i = 0
        while i < len(all_sorted):
            msg = all_sorted[i][1]
            
            # 如果是 AIMessage 且包含工具调用，寻找对应的 ToolMessage
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                working_memory.append(msg)
                i += 1
                
                # 寻找紧随其后的 ToolMessage
                while i < len(all_sorted) and isinstance(all_sorted[i][1], ToolMessage):
                    working_memory.append(all_sorted[i][1])
                    i += 1
                    
            # 如果是 HumanMessage，添加最近的几个
            elif isinstance(msg, HumanMessage):
                # 只保留最近的5个 HumanMessage
                recent_humans = [m[1] for m in human_msgs[-5:]]
                if msg in recent_humans:
                    working_memory.append(msg)
                i += 1
                
            # 其他消息直接添加（但要考虑数量限制）
            else:
                working_memory.append(msg)
                i += 1
 
        # 如果工作内存过长，智能裁剪
        if len(working_memory) > self.max_woking_memory:
            # 保留系统消息
            final_memory = [msg for msg in working_memory if isinstance(msg, SystemMessage)]
            
            # 按类型分组
            humans = [msg for msg in working_memory if isinstance(msg, HumanMessage)]
            ais = [msg for msg in working_memory if isinstance(msg, AIMessage)]
            tools = [msg for msg in working_memory if isinstance(msg, ToolMessage)]
            
            # 保留最近的消息
<<<<<<< HEAD
            final_memory.extend(humans[-10:])  # 最近10个用户消息
            final_memory.extend(ais[-10:])     # 最近10个AI消息
            final_memory.extend(tools[-10:])   # 最近10个工具消息
=======
            final_memory.extend(humans[-2:])  # 最近2个用户消息
            final_memory.extend(ais[-2:])     # 最近2个AI消息
            final_memory.extend(tools[-2:])   # 最近2个工具消息
>>>>>>> ollama_use_meta_chunk
            
            # 去重并保持顺序
            seen = set()
            unique_memory = []
            for msg in final_memory:
                msg_id = id(msg)
                if msg_id not in seen:
                    seen.add(msg_id)
                    unique_memory.append(msg)
            
            working_memory = unique_memory[:self.max_woking_memory]
 
        # 确保最终顺序正确
        working_memory = self._maintain_proper_order(working_memory)
        
        return working_memory
 
    def _maintain_proper_order(self, messages: list) -> list:
        """
        确保消息顺序正确：System -> Human -> AI -> Tool 的循环
        """
        # 移除重复消息
        seen_ids = set()
        unique_messages = []
        for msg in messages:
            if id(msg) not in seen_ids:
                seen_ids.add(id(msg))
                unique_messages.append(msg)
        
        return unique_messages
 
    def get_statistics(self) -> dict:
        """
        获取消息统计
        """
        return {
            "total_history": len(self.history),
            "system_messages": len([m for m in self.history if isinstance(m, SystemMessage)]),
            "human_messages": len([m for m in self.history if isinstance(m, HumanMessage)]),
            "ai_messages": len([m for m in self.history if isinstance(m, AIMessage)]),
            "tool_messages": len([m for m in self.history if isinstance(m, ToolMessage)])
        }


if __name__ == "__main__":
    mm = MessagerManager(max_woking_memory=10, max_history=10)
    
    # 测试1: 空消息
    print("测试1 - 空消息:")
    result = mm([], [])
    print(f"结果: {result}")  # []
    
    # 测试2: 系统消息 + 用户消息
    print("\n测试2 - 系统 + 用户消息:")
    current = [SystemMessage(content="你是助手")]
    new = [HumanMessage(content="你好")]
    result = mm(current, new)
    print(f"结果数量: {len(result)}")
    for msg in result:
        print(f"  - {type(msg).__name__}: {msg.content}")
    
    # 测试3: 多轮对话模拟
    print("\n测试3 - 多轮对话:")
    current = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="问题1"),
        AIMessage(content="回答1"),
        HumanMessage(content="问题2"),
        AIMessage(content="回答2")
    ]
    new = [HumanMessage(content="新问题")]
    result = mm(current, new)
    print(f"结果数量: {len(result)}")
    for msg in result:
        print(f"  - {type(msg).__name__}: {msg.content}")