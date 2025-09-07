import os
import glob
from pathlib import Path
from typing import List, Optional
from langchain.tools import Tool


class DocumentReader:
    """
    文档读取器 - 支持读取各种文本格式文件
    """
    
    def __init__(self):
        # 支持的文件扩展名
        self.supported_extensions = {
            '.txt', '.md', '.py', '.js', '.html', '.css', 
            '.json', '.xml', '.csv', '.log', '.cfg', '.ini',
            '.cpp', '.c', '.h', '.java', '.go', '.rs', '.php'
        }
    
    def read_document(self, path: str) -> str:
        """
        读取指定路径的文档内容
        
        Args:
            path: 文件路径或目录路径，支持通配符
            
        Returns:
            str: 文档内容或错误信息
        """
        try:
            # 处理路径
            path = path.strip()
            
            # 如果是目录，列出所有支持的文件
            if os.path.isdir(path):
                return self._read_directory(path)
            
            # 如果包含通配符，处理模式匹配
            if '*' in path or '?' in path:
                return self._read_pattern(path)
            
            # 单个文件处理
            if os.path.isfile(path):
                return self._read_single_file(path)
            
            return f"❌ 路径不存在: {path}"
            
        except Exception as e:
            return f"❌ 读取失败: {str(e)}"
    
    def _read_single_file(self, file_path: str) -> str:
        """读取单个文件"""
        try:
            # 检查文件扩展名
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                return f"❌ {os.path.basename(file_path)}是不支持的文件类型"
            
            # 获取文件信息
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024 * 10:  # 10MB 限制
                return f"❌ {os.path.basename(file_path)}文件过大"
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 格式化输出
            result = f"📄 文件: {file_path}\n"
            result += f"📊 大小: {len(content)} 字符\n"
            result += f"🔤 类型: {file_ext}\n"
            result += "=" * 50 + "\n"
            result += content
            
            return result
            
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                result = f"📄 文件: {file_path} (GBK编码)\n"
                result += f"📊 大小: {len(content)} 字符\n"
                result += "=" * 50 + "\n"
                result += content
                return result
            except:
                return f"❌ 无法读取文件编码: {file_path}"
        
        except Exception as e:
            return f"❌ 读取文件失败: {file_path} - {str(e)}"
    
    def _read_directory(self, dir_path: str) -> str:
        """读取目录下的所有支持文件"""
        try:
            files = []
            for root, dirs, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_ext = Path(filename).suffix.lower()
                    if file_ext in self.supported_extensions:
                        files.append(os.path.join(root, filename))
            
            if not files:
                return f"📁 目录 {dir_path} 中没有找到支持的文件类型"
            
            # 限制文件数量
            if len(files) > 10:
                result = f"📁 目录: {dir_path}\n"
                result += f"⚠️  找到 {len(files)} 个文件，只显示前10个:\n\n"
                files = files[:10]
            else:
                result = f"📁 目录: {dir_path}\n"
                result += f"📄 找到 {len(files)} 个文件:\n\n"
            
            # 读取每个文件
            for i, file_path in enumerate(files, 1):
                result += f"\n{'='*20} 文件 {i} {'='*20}\n"
                file_content = self._read_single_file(file_path)
                result += file_content + "\n"
            
            return result
            
        except Exception as e:
            return f"❌ 读取目录失败: {dir_path} - {str(e)}"
    
    def _read_pattern(self, pattern: str) -> str:
        """根据通配符模式读取文件"""
        try:
            files = glob.glob(pattern, recursive=True)
            
            # 过滤支持的文件类型
            supported_files = []
            for file_path in files:
                if os.path.isfile(file_path):
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in self.supported_extensions:
                        supported_files.append(file_path)
            
            if not supported_files:
                return f"🔍 模式 '{pattern}' 没有匹配到支持的文件"
            
            # 限制文件数量
            if len(supported_files) > 5:
                result = f"🔍 模式: {pattern}\n"
                result += f"⚠️  匹配到 {len(supported_files)} 个文件，只显示前5个:\n\n"
                supported_files = supported_files[:5]
            else:
                result = f"🔍 模式: {pattern}\n"
                result += f"📄 匹配到 {len(supported_files)} 个文件:\n\n"
            
            # 读取每个文件
            for i, file_path in enumerate(supported_files, 1):
                result += f"\n{'='*20} 文件 {i} {'='*20}\n"
                file_content = self._read_single_file(file_path)
                result += file_content + "\n"
            
            return result
            
        except Exception as e:
            return f"❌ 模式匹配失败: {pattern} - {str(e)}"


def create_document_reader_tool() -> Tool:
    """创建文档读取工具"""
    reader = DocumentReader()
    
    def read_document(path: str) -> str:
        """
        读取文档工具函数
        
        Args:
            path: 文件路径、目录路径或通配符模式
            
        Examples:
            - "README.md" - 读取单个文件
            - "src/" - 读取目录下所有支持的文件
            - "*.py" - 读取当前目录下所有Python文件
            - "src/**/*.js" - 递归读取src目录下所有JS文件
        """
        return reader.read_document(path)

    return Tool(
        name="document_reader",
        description="""
        读取各种文本格式的文档文件。
        
        支持的文件类型:
        - 文本文件: .txt, .md, .log, .cfg, .ini
        - 代码文件: .py, .js, .html, .css, .cpp, .c, .h, .java, .go, .rs, .php
        - 数据文件: .json, .xml, .csv
        
        输入格式:
        - 单个文件: "path/to/file.txt"
        - 目录: "path/to/directory/"
        - 通配符: "*.py" 或 "src/**/*.js"
        
        使用场景:
        - 分析代码文件
        - 读取配置文件
        - 查看日志文件
        - 批量处理文档
        """,
        func=read_document
    )
    