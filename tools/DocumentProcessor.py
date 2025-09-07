import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    CharacterTextSplitter
)
import pypdf
import docx


class DocumentProcessor:
    """
    文档处理器 - 支持多种文档格式的解析和分块
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 文档块大小
            chunk_overlap: 块之间的重叠
            separators: 自定义分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 支持的文件格式
        self.supported_formats = {
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.py': self._process_python,
            '.js': self._process_javascript,
            '.java': self._process_java,
            '.cpp': self._process_cpp,
            '.c': self._process_cpp,
            '.h': self._process_cpp,
            '.hpp': self._process_cpp,
            '.html': self._process_html,
            '.xml': self._process_xml,
            '.json': self._process_json,
            '.csv': self._process_csv,
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.log': self._process_text,
            '.cfg': self._process_text,
            '.ini': self._process_text,
            '.yaml': self._process_text,
            '.yml': self._process_text,
        }
        
        # 初始化不同类型的文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.code_splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Document]: 处理后的文档块列表
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 获取文件扩展名
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in self.supported_formats:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            # 获取文件基本信息
            file_stats = os.stat(file_path)
            base_metadata = {
                "source": os.path.abspath(file_path),
                "file_name": os.path.basename(file_path),
                "file_type": file_ext,
                "file_size": file_stats.st_size,
                "modified_time": file_stats.st_mtime
            }
            
            # 调用对应的处理函数
            processor_func = self.supported_formats[file_ext]
            documents = processor_func(file_path, base_metadata)
            
            print(f"✅ 成功处理文件: {os.path.basename(file_path)} - {len(documents)} 个块")
            return documents
            
        except Exception as e:
            print(f"❌ 处理文件失败 {file_path}: {e}")
            return []
    
    def process_directory(self, directory_path: str, 
                         recursive: bool = True,
                         file_pattern: Optional[str] = None) -> List[Document]:
        """
        处理目录下的所有支持文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            file_pattern: 文件名模式过滤（正则表达式）
            
        Returns:
            List[Document]: 所有文档块列表
        """
        all_documents = []
        
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"目录不存在或不是目录: {directory_path}")
            
            # 获取文件列表
            if recursive:
                files = directory.rglob("*")
            else:
                files = directory.glob("*")
            
            processed_files = 0
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                # 检查文件扩展名
                if file_path.suffix.lower() not in self.supported_formats:
                    continue
                
                # 应用文件名模式过滤
                if file_pattern and not re.match(file_pattern, file_path.name):
                    continue
                
                # 处理文件
                documents = self.process_file(str(file_path))
                all_documents.extend(documents)
                processed_files += 1
            
            print(f"✅ 目录处理完成: {processed_files} 个文件, 共 {len(all_documents)} 个文档块")
            return all_documents
            
        except Exception as e:
            print(f"❌ 处理目录失败 {directory_path}: {e}")
            return []
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        直接处理文本内容
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            List[Document]: 文档块列表
        """
        try:
            if not text.strip():
                return []
            
            # 分割文本
            chunks = self.text_splitter.split_text(text)
            
            # 创建文档对象
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata.update({
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            return documents
            
        except Exception as e:
            print(f"❌ 处理文本失败: {e}")
            return []
    
    def _process_text(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理纯文本文件"""
        encodings = ['utf-8', 'gbk', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法解码文件: {file_path}")
        
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_markdown(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理Markdown文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.markdown_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_python(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理Python代码文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.code_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_javascript(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理JavaScript文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 对于JavaScript，使用通用的代码分割器
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_java(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理Java文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_cpp(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理C/C++文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_html(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理HTML文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单去除HTML标签（可以后续使用BeautifulSoup改进）
        import re
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_xml(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理XML文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_json(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理JSON文件"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                content = json.dumps(data, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # 如果JSON解析失败，作为普通文本处理
                f.seek(0)
                content = f.read()
        
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_csv(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理CSV文件"""
        import csv
        
        content_lines = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                reader = csv.reader(f)
                for row in reader:
                    content_lines.append(",".join(row))
            except Exception:
                # 如果CSV解析失败，作为普通文本处理
                f.seek(0)
                content_lines = f.readlines()
        
        content = "\n".join(content_lines)
        chunks = self.text_splitter.split_text(content)
        return self._create_documents(chunks, base_metadata)
    
    def _process_pdf(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理PDF文件"""
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                content = ""
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        content += f"\n--- 第 {page_num + 1} 页 ---\n"
                        content += page_text + "\n"
            
            if not content.strip():
                raise ValueError("PDF文件无法提取文本内容")
            
            chunks = self.text_splitter.split_text(content)
            return self._create_documents(chunks, base_metadata)
            
        except Exception as e:
            print(f"❌ PDF处理失败: {e}")
            return []
    
    def _process_docx(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """处理Word文档"""
        try:
            doc = docx.Document(file_path)
            content = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content += paragraph.text + "\n"
            
            # 处理表格
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        content += row_text + "\n"
            
            if not content.strip():
                raise ValueError("Word文档无法提取文本内容")
            
            chunks = self.text_splitter.split_text(content)
            return self._create_documents(chunks, base_metadata)
            
        except Exception as e:
            print(f"❌ Word文档处理失败: {e}")
            return []
    
    def _create_documents(self, chunks: List[str], base_metadata: Dict) -> List[Document]:
        """创建文档对象列表"""
        documents = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            doc_metadata = base_metadata.copy()
            doc_metadata.update({
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表"""
        return list(self.supported_formats.keys())
    
    def estimate_chunks(self, text_length: int) -> int:
        """估算文本块数量"""
        return max(1, text_length // (self.chunk_size - self.chunk_overlap))


def create_document_processor(chunk_size: int = 1000,
                            chunk_overlap: int = 200) -> DocumentProcessor:
    """
    创建文档处理器实例
    
    Args:
        chunk_size: 文档块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        DocumentProcessor: 文档处理器实例
    """
    return DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)