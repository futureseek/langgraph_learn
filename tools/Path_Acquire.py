import os
import glob
from pathlib import Path
from typing import List, Optional
from langchain.tools import Tool

def create_path_acquire_tool()->Tool:
    """
    创建获取当前目录的工具
    """
    def get_current_directory(method='cwd'):
        """
        Args:
            method: 'cwd' - 工作目录, 'script' - 脚本目录
    
        Returns:
            str: 目录路径

        """
        if method == 'cwd':
            return os.getcwd()
        elif method == 'script':
            return os.path.dirname(os.path.abspath(__file__))
        else:
            return os.getcwd()
    

    return Tool(
        name="get_current_directory",
        description="""
            使用这个工具可以获取当前目录的工作路径或者目录
            传入str类型的作为参数,str为cwd时候获取工作目录,str为script时候获取脚本目录,默认参数为cwd
            返回值是str类型的
            当你需要访问文件路径或者获取项目路径的时候可以调用这个函数。
        """,
        func=get_current_directory
    )

if __name__ == "__main__":
    tool = create_path_acquire_tool()
    result = tool.invoke("cwd")
    print(result)

    result = tool.invoke("script")
    print(result)
