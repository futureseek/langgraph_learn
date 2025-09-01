import os
import glob
from pathlib import Path
from typing import List, Optional
from langchain_tavily import TavilySearch
from langchain_community.tools import Tool


class TavilySearcher:
    """
    Tavily 搜索引擎 - 支持web搜索，返回结果
    """
    def __init__(self):
        api_key = "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"
        self.search_tool = TavilySearch(tavily_api_key = api_key)

    def search(self,query:str)->str:
        """
        执行搜索并且返回结果
        Args:
            query: str:搜索关键词

        Returns: 
            str:搜索结果

        """
        try:
            results = self.search_tool.run(query)
            if not results:
                return "no match result"
            '''
            print(type(results))
            print(results)
            print("Results keys:", results.keys())
            '''
            result_str = f"查询: {query}\n"
            result_str += f"结果数量: {len(results)}\n"
            results = results['results']

            for i, result in enumerate(results, start=1):
                result_str += f"标题: {result['title']}\n"
                result_str += f"链接: {result['url']}\n"
                result_str += f"描述: {result['content']}\n"
            
            return result_str
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"

def create_tavily_search_reader_tool() -> Tool:
    """创建 Tavily 搜索读取工具"""
    reader = TavilySearcher()
    
    def search(query: str) -> str:
        """
        执行 Web 搜索工具函数
        
        Args:
            query: 搜索查询字符串
            
        Examples:
            - "Python programming tutorials" - 搜索 Python 编程教程
            - "best pizza recipe" - 搜索最佳比萨食谱
        """
        return reader.search(query)
 
    return Tool(
        name="tavily_search_reader",
        description="""
        使用 Tavily 搜索引擎执行 Web 搜索并返回结果。
        
        eg:
        输入格式:
        - 搜索查询字符串: "Python programming tutorials"
        
        使用场景:
        - 搜索技术文档
        - 查找新闻和文章
        - 搜索食谱和生活小技巧
        - 批量搜索多个关键词
        """,
        func=search
    )

# 示例用法
if __name__ == "__main__":
    tavily_search_tool = create_tavily_search_reader_tool()
    
    # 使用工具进行搜索
    query = "Python programming tutorials"
    results = tavily_search_tool.func(query)
    
    # 打印结果
    print(results)
