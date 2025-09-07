"""
文档导出工具
用于将对话内容和搜索结果导出为 Markdown 格式文档
"""

import os
import re
from datetime import datetime
from typing import Optional, Dict, Any
from langchain.tools import Tool


class DocumentExporter:
    """文档导出器类"""
    
    def __init__(self, output_dir: str = "exports"):
        """
        初始化文档导出器
        
        Args:
            output_dir: 输出目录，默认为 'exports'
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除不合法字符"""
        # 移除或替换不合法字符
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 限制长度
        if len(filename) > 100:
            filename = filename[:100]
        return filename
    
    def _generate_filename(self, title: Optional[str] = None) -> str:
        """生成文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if title:
            clean_title = self._sanitize_filename(title)
            return f"{timestamp}_{clean_title}.md"
        else:
            return f"document_{timestamp}.md"
    
    def export_to_markdown(self, content: str, title: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        导出内容到 Markdown 文件
        
        Args:
            content: 要导出的内容
            title: 文档标题（可选）
            metadata: 元数据信息（可选）
            
        Returns:
            str: 生成的文件路径
        """
        filename = self._generate_filename(title)
        filepath = os.path.join(self.output_dir, filename)
        
        # 构建 Markdown 内容
        md_content = self._build_markdown_content(content, title, metadata)
        
        # 写入文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            return f"✅ 文档已成功导出到: {filepath}"
        except Exception as e:
            return f"❌ 文档导出失败: {str(e)}"
    
    def _build_markdown_content(self, content: str, title: Optional[str] = None, 
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """构建 Markdown 内容"""
        lines = []
        
        # 添加标题
        if title:
            lines.append(f"# {title}")
            lines.append("")
        
        # 添加元数据
        if metadata:
            lines.append("## 文档信息")
            lines.append("")
            for key, value in metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # 添加生成时间
        lines.append("## 生成信息")
        lines.append("")
        lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **生成工具**: LangGraph 聊天机器人")
        lines.append("")
        
        # 添加主要内容
        lines.append("## 内容")
        lines.append("")
        lines.append(content)
        
        return "\n".join(lines)


def create_document_export_tool() -> Tool:
    """创建文档导出工具"""
    exporter = DocumentExporter()
    
    def export_document(input_str: str) -> str:
        """
        导出文档工具函数
        
        输入格式: "标题|内容" 或者直接是内容
        """
        try:
            # 解析输入
            if "|" in input_str:
                parts = input_str.split("|", 1)
                title = parts[0].strip()
                content = parts[1].strip()
            else:
                title = None
                content = input_str.strip()
            
            # 添加一些元数据
            metadata = {
                "内容长度": f"{len(content)} 字符",
                "内容类型": "AI 生成内容"
            }
            
            return exporter.export_to_markdown(content, title, metadata)
            
        except Exception as e:
            return f"❌ 文档导出工具执行失败: {str(e)}"
    
    return Tool(
        name="export_document",
        description="""
        将内容导出为 Markdown 文档。
        使用场景：当用户要求将信息保存到文档、整理到文件、导出报告等时使用。
        
        输入格式：
        1. 只有内容："这是要导出的内容"
        2. 标题和内容："文档标题|这是要导出的内容"
        
        示例：
        - "明天天气预报|明天北京天气晴朗，温度15-25度"
        - "市场调研报告|根据搜索结果，当前市场趋势..."
        """,
        func=export_document
    )


# 为了方便直接使用
def export_to_markdown(content: str, title: Optional[str] = None) -> str:
    """便捷函数：直接导出内容到 Markdown"""
    exporter = DocumentExporter()
    return exporter.export_to_markdown(content, title)


if __name__ == "__main__":
    # 测试代码
    print("测试文档导出工具...")
    
    # 创建工具
    tool = create_document_export_tool()
    
    # 测试1: 只有内容
    result1 = tool.func("这是一个测试文档的内容，包含了一些重要信息。")
    print(f"测试1结果: {result1}")
    
    # 测试2: 标题和内容
    result2 = tool.func("天气预报|明天北京天气晴朗，气温15-25摄氏度，适合外出活动。")
    print(f"测试2结果: {result2}")
    
    print("测试完成！")