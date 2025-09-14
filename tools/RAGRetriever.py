import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from langchain.tools import Tool, StructuredTool
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .VectorDatabase import VectorDatabase, create_vector_database
from .DocumentProcessor import DocumentProcessor, create_document_processor, create_api_client


class RAGRetriever:
    """
    RAG检索器 - 实现完整的RAG问答功能
    """
    
    def __init__(self, 
                 vector_db: Optional[VectorDatabase] = None,
                 doc_processor: Optional[DocumentProcessor] = None,
                 model=None):
        """
        初始化RAG检索器
        
        Args:
            vector_db: 向量数据库实例
            doc_processor: 文档处理器实例
            model: 语言模型实例
        """
        self.vector_db = vector_db or create_vector_database()
        self.doc_processor = doc_processor or create_document_processor()
        self.model = model
        
        # RAG配置
        self.config = {
            "retrieval_k": 4,  # 检索文档数量
            "score_threshold": 0.3,  # 相似度阈值（降低以提高召回率）
            "max_context_length": 4000,  # 最大上下文长度
            "enable_reranking": True,  # 是否启用重新排序
        }
    
    def add_document_to_knowledge_base(self, file_path: str) -> Dict[str, Any]:
        """
        将文档添加到知识库
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            Dict: 添加结果
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}",
                    "file_path": file_path
                }
            
            # 处理文档
            print(f"📄 正在处理文档: {os.path.basename(file_path)}")
            documents = self.doc_processor.process_file(file_path)
            
            if not documents:
                return {
                    "success": False,
                    "error": "文档处理失败或文档为空",
                    "file_path": file_path
                }
            
            # 添加到向量数据库
            success = self.vector_db.add_documents(documents, file_path)
            
            if success:
                stats = self.vector_db.get_stats()
                return {
                    "success": True,
                    "message": f"成功添加文档到知识库",
                    "file_path": file_path,
                    "chunks_added": len(documents),
                    "total_chunks": stats.get("total_chunks", 0),
                    "total_documents": stats.get("total_documents", 0)
                }
            else:
                return {
                    "success": False,
                    "error": "添加到向量数据库失败",
                    "file_path": file_path
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"处理文档时出错: {str(e)}",
                "file_path": file_path
            }
    
    def add_directory_to_knowledge_base(self, directory_path: str, 
                                      recursive: bool = True) -> Dict[str, Any]:
        """
        将目录下的所有文档添加到知识库
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            
        Returns:
            Dict: 添加结果
        """
        try:
            if not os.path.isdir(directory_path):
                return {
                    "success": False,
                    "error": f"目录不存在: {directory_path}"
                }
            
            print(f"📁 正在处理目录: {directory_path}")
            documents = self.doc_processor.process_directory(directory_path, recursive)
            
            if not documents:
                return {
                    "success": False,
                    "error": "目录中没有可处理的文档",
                    "directory_path": directory_path
                }
            
            # 按文件分组并逐个添加
            file_groups = {}
            for doc in documents:
                file_path = doc.metadata.get("source", "unknown")
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append(doc)
            
            total_added = 0
            processed_files = 0
            errors = []
            
            for file_path, file_docs in file_groups.items():
                try:
                    success = self.vector_db.add_documents(file_docs, file_path)
                    if success:
                        total_added += len(file_docs)
                        processed_files += 1
                    else:
                        errors.append(f"添加失败: {os.path.basename(file_path)}")
                except Exception as e:
                    errors.append(f"处理错误: {os.path.basename(file_path)} - {str(e)}")
            
            stats = self.vector_db.get_stats()
            
            return {
                "success": processed_files > 0,
                "message": f"目录处理完成",
                "directory_path": directory_path,
                "processed_files": processed_files,
                "total_chunks_added": total_added,
                "total_chunks": stats.get("total_chunks", 0),
                "total_documents": stats.get("total_documents", 0),
                "errors": errors
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"处理目录时出错: {str(e)}",
                "directory_path": directory_path
            }
    
    def retrieve_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            List[Document]: 相关文档列表
        """
        try:
            k = k or self.config["retrieval_k"]
            
            # 执行相似性搜索
            results = self.vector_db.similarity_search_with_score(query, k=k * 2)  # 获取更多候选
            
            # 过滤低分文档
            filtered_results = []
            for doc, score in results:
                if score >= self.config["score_threshold"]:
                    doc.metadata["relevance_score"] = score
                    filtered_results.append(doc)
            
            # 重新排序（如果启用）
            if self.config["enable_reranking"] and len(filtered_results) > k:
                filtered_results = self._rerank_documents(query, filtered_results)
            
            return filtered_results[:k]
            
        except Exception as e:
            print(f"❌ 检索文档失败: {e}")
            return []
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        重新排序文档（简单的基于关键词匹配的排序）
        """
        try:
            query_words = set(query.lower().split())
            
            def calculate_keyword_score(doc: Document) -> float:
                content_words = set(doc.page_content.lower().split())
                intersection = query_words.intersection(content_words)
                return len(intersection) / len(query_words) if query_words else 0
            
            # 结合相似度分数和关键词分数
            scored_docs = []
            for doc in documents:
                similarity_score = doc.metadata.get("relevance_score", 0)
                keyword_score = calculate_keyword_score(doc)
                combined_score = 0.7 * similarity_score + 0.3 * keyword_score
                
                doc.metadata["combined_score"] = combined_score
                scored_docs.append((doc, combined_score))
            
            # 按综合分数排序
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs]
            
        except Exception as e:
            print(f"⚠️ 重新排序失败: {e}")
            return documents
    
    def answer_question(self, question: str, context_docs: Optional[List[Document]] = None) -> Dict[str, Any]:
        """
        基于知识库回答问题
        
        Args:
            question: 用户问题
            context_docs: 上下文文档（可选，如果不提供会自动检索）
            
        Returns:
            Dict: 回答结果
        """
        try:
            # 如果没有提供上下文文档，先检索
            if context_docs is None:
                context_docs = self.retrieve_relevant_documents(question)
            
            if not context_docs:
                return {
                    "answer": "抱歉，我在知识库中没有找到相关信息来回答您的问题。您可以尝试添加更多相关文档到知识库中。",
                    "sources": [],
                    "confidence": 0.0,
                    "retrieved_docs": 0
                }
            
            # 构建上下文
            context = self._build_context(context_docs)
            
            # 生成回答
            if self.model:
                answer = self._generate_answer_with_model(question, context)
                confidence = self._estimate_confidence(answer, context_docs)
            else:
                answer = self._generate_simple_answer(question, context_docs)
                confidence = 0.5
            
            # 提取来源信息
            sources = self._extract_sources(context_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "retrieved_docs": len(context_docs),
                "context_length": len(context)
            }
            
        except Exception as e:
            return {
                "answer": f"回答问题时出现错误: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "retrieved_docs": 0
            }
    
    def _build_context(self, documents: List[Document]) -> str:
        """构建上下文字符串"""
        context_parts = []
        current_length = 0
        max_length = self.config["max_context_length"]
        
        for i, doc in enumerate(documents):
            source_info = f"[来源 {i+1}: {os.path.basename(doc.metadata.get('source', 'unknown'))}]"
            content = f"{source_info}\n{doc.page_content}\n"
            
            if current_length + len(content) > max_length:
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n---\n".join(context_parts)
    
    def _generate_answer_with_model(self, question: str, context: str) -> str:
        """使用语言模型生成回答"""
        try:
            prompt = f"""根据以下上下文信息回答问题。如果上下文中没有相关信息，请明确说明。

上下文信息：
{context}

问题：{question}

请基于上下文信息提供准确、详细的回答，并在适当位置引用来源。如果信息不足，请说明需要补充哪些信息。"""

            messages = [
                SystemMessage(content="你是一个智能文档问答助手，专门基于提供的上下文信息回答用户问题。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.model.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"❌ 生成回答失败: {e}")
            return f"生成回答时出现错误: {str(e)}"
    
    def _generate_simple_answer(self, question: str, documents: List[Document]) -> str:
        """生成简单回答（当没有模型时）"""
        # 简单的关键词匹配回答
        question_words = set(question.lower().split())
        relevant_snippets = []
        
        for doc in documents:
            content_words = set(doc.page_content.lower().split())
            if question_words.intersection(content_words):
                snippet = doc.page_content[:200] + "..."
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                relevant_snippets.append(f"来自 {source}: {snippet}")
        
        if relevant_snippets:
            return "基于知识库中的相关信息：\n\n" + "\n\n".join(relevant_snippets[:3])
        else:
            return "未找到直接相关的信息。"
    
    def _estimate_confidence(self, answer: str, documents: List[Document]) -> float:
        """估算回答置信度"""
        try:
            # 简单的置信度估算逻辑
            if "不知道" in answer or "没有信息" in answer or "无法回答" in answer:
                return 0.2
            
            avg_score = sum(doc.metadata.get("relevance_score", 0.5) for doc in documents) / len(documents)
            
            # 基于检索文档数量和平均相似度调整置信度
            doc_count_factor = min(1.0, len(documents) / 3)
            confidence = avg_score * doc_count_factor
            
            return min(0.95, max(0.1, confidence))
            
        except Exception:
            return 0.5
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """提取来源信息"""
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source_path = doc.metadata.get("source", "unknown")
            if source_path not in seen_sources:
                sources.append({
                    "file_name": os.path.basename(source_path),
                    "file_path": source_path,
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "relevance_score": doc.metadata.get("relevance_score", 0.0)
                })
                seen_sources.add(source_path)
        
        return sources
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            stats = self.vector_db.get_stats()
            documents = self.vector_db.list_documents()
            
            return {
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "embedding_model": stats.get("embedding_model", "unknown"),
                "documents": documents,
                "supported_formats": self.doc_processor.get_supported_formats()
            }
        except Exception as e:
            return {"error": f"获取统计信息失败: {str(e)}"}
    
    def delete_document_from_knowledge_base(self, file_path: str) -> Dict[str, Any]:
        """从知识库删除文档"""
        try:
            success = self.vector_db.delete_document(file_path)
            if success:
                return {
                    "success": True,
                    "message": f"成功删除文档: {os.path.basename(file_path)}"
                }
            else:
                return {
                    "success": False,
                    "error": "删除文档失败"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"删除文档时出错: {str(e)}"
            }
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """清空知识库"""
        try:
            success = self.vector_db.clear_all()
            if success:
                return {
                    "success": True,
                    "message": "知识库已清空"
                }
            else:
                return {
                    "success": False,
                    "error": "清空知识库失败"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"清空知识库时出错: {str(e)}"
            }


# 定义输入模型
class EmptyInput(BaseModel):
    """空输入模型，用于不需要参数的工具"""
    pass

class FilePathInput(BaseModel):
    """文件路径输入模型"""
    file_path: str = Field(description="文件路径")

class DirectoryInput(BaseModel):
    """目录输入模型"""
    directory_path: str = Field(description="目录路径")
    recursive: bool = Field(default=True, description="是否递归处理子目录")

class QuestionInput(BaseModel):
    """问题输入模型"""
    question: str = Field(description="要询问的问题")


def create_rag_tools(model=None) -> List[Tool]:
    """
    创建RAG相关的工具集合
    
    Args:
        model: 语言模型实例
        
    Returns:
        List[Tool]: RAG工具列表
    """
    # 创建基于 API 分割决策的 DocumentProcessor（优先使用环境变量配置）
    api_client, api_model = create_api_client(
        api_key="ollama", 
        base_url="http://localhost:11434/v1", 
        model_name="qwen3:1.7b"
        )

    doc_processor = create_document_processor(
        meta_chunking_strategy="prob_subtract",  # 使用 API 决策（失败自动回退默认逻辑）
        api_client=api_client,
        api_model=api_model
    )

    # 创建RAG检索器实例（注入定制的文档处理器）
    rag_retriever = RAGRetriever(model=model, doc_processor=doc_processor)
    
    def add_document_to_rag(file_path: str) -> str:
        """添加文档到RAG知识库"""
        result = rag_retriever.add_document_to_knowledge_base(file_path)
        
        if result["success"]:
            return f"✅ {result['message']}\n📄 文件: {os.path.basename(result['file_path'])}\n📊 添加了 {result['chunks_added']} 个文档块\n🗂️ 知识库现有 {result['total_documents']} 个文档，共 {result['total_chunks']} 个块"
        else:
            return f"❌ 添加文档失败: {result['error']}"
    
    def add_directory_to_rag(directory_path: str, recursive: bool = True) -> str:
        """批量添加目录下的文档到RAG知识库"""
        result = rag_retriever.add_directory_to_knowledge_base(directory_path, recursive)
        
        if result["success"]:
            message = f"✅ {result['message']}\n📁 目录: {result['directory_path']}\n📄 处理了 {result['processed_files']} 个文件\n📊 添加了 {result['total_chunks_added']} 个文档块\n🗂️ 知识库现有 {result['total_documents']} 个文档，共 {result['total_chunks']} 个块"
            
            if result.get("errors"):
                message += f"\n⚠️ 错误: {', '.join(result['errors'])}"
            
            return message
        else:
            return f"❌ 处理目录失败: {result['error']}"
    
    def rag_question_answer(question: str) -> str:
        """基于RAG知识库回答问题"""
        result = rag_retriever.answer_question(question)
        
        response = f"🤖 RAG回答:\n{result['answer']}\n"
        
        if result["sources"]:
            response += f"\n📚 参考来源:\n"
            for i, source in enumerate(result["sources"][:3], 1):
                response += f"{i}. {source['file_name']} (相似度: {source['relevance_score']:.2f})\n"
        
        response += f"\n📊 检索了 {result['retrieved_docs']} 个相关文档块"
        response += f"\n🎯 置信度: {result['confidence']:.2f}"
        
        return response
    
    def get_rag_stats(input_data="") -> str:
        """获取RAG知识库统计信息"""
        stats = rag_retriever.get_knowledge_base_stats()
        
        if "error" in stats:
            return f"❌ 获取统计信息失败: {stats['error']}"
        
        response = f"📊 RAG知识库统计信息:\n"
        response += f"📄 文档总数: {stats['total_documents']}\n"
        response += f"📋 文档块总数: {stats['total_chunks']}\n"
        response += f"🧠 嵌入模型: {stats['embedding_model']}\n"
        response += f"📝 支持格式: {', '.join(stats['supported_formats'])}\n"
        
        if stats['documents']:
            response += f"\n📂 已索引的文档:\n"
            for doc in stats['documents'][:5]:  # 只显示前5个
                response += f"• {doc['file_name']} ({doc['chunks']} 块)\n"
            
            if len(stats['documents']) > 5:
                response += f"... 还有 {len(stats['documents']) - 5} 个文档\n"
        
        return response
    
    def delete_rag_document(file_path: str) -> str:
        """从RAG知识库删除文档"""
        result = rag_retriever.delete_document_from_knowledge_base(file_path)
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ 删除失败: {result['error']}"
    
    def clear_rag_knowledge_base(input_data="") -> str:
        """清空RAG知识库"""
        result = rag_retriever.clear_knowledge_base()
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ 清空失败: {result['error']}"
    
    # 创建工具列表
    tools = [
        Tool(
            name="add_document_to_rag",
            description="""将文档添加到RAG知识库中，用于后续的文档问答。
            
支持的文件格式: .txt, .md, .py, .js, .java, .cpp, .html, .json, .csv, .pdf, .docx 等
            
使用场景:
- 添加技术文档、说明书到知识库
- 建立项目相关的问答系统  
- 创建个人知识管理系统

参数:
- file_path: 要添加的文件路径（必需）

示例: add_document_to_rag("./README.md")""",
            func=add_document_to_rag
        ),
        
        Tool(
            name="add_directory_to_rag",
            description="""批量将目录下的所有支持文档添加到RAG知识库。
            
使用场景:
- 批量导入项目文档
- 建立完整的文档库
- 处理大量历史文档

参数:
- directory_path: 目录路径（必需）
- recursive: 是否递归处理子目录，默认True（可选）

示例: add_directory_to_rag("./docs/")""",
            func=add_directory_to_rag
        ),
        
        Tool(
            name="rag_question_answer",
            description="""基于RAG知识库回答问题。系统会自动检索相关文档并生成回答。
            
使用场景:
- 根据文档内容回答技术问题
- 快速查找项目相关信息
- 基于知识库的智能问答

参数:
- question: 要询问的问题（必需）

示例: rag_question_answer("如何安装这个项目？")""",
            func=rag_question_answer
        ),
        
        Tool(
            name="get_rag_stats",
            description="""获取RAG知识库的统计信息，包括文档数量、块数量、已索引文档列表等。
            
使用场景:
- 查看知识库状态
- 了解已添加的文档
- 监控系统运行状态

无需参数

示例: get_rag_stats()""",
            func=get_rag_stats
        ),
        
        Tool(
            name="delete_rag_document", 
            description="""从RAG知识库中删除指定文档。
            
使用场景:
- 移除过时的文档
- 清理错误添加的文件
- 管理知识库内容

参数:
- file_path: 要删除的文件路径（必需）

示例: delete_rag_document("./old_doc.md")""",
            func=delete_rag_document
        ),
        
        Tool(
            name="clear_rag_knowledge_base",
            description="""清空整个RAG知识库，删除所有已添加的文档。
            
⚠️ 警告: 此操作不可逆，会删除所有数据！

使用场景:
- 重新建立知识库
- 清理测试数据
- 解决数据问题

无需参数

示例: clear_rag_knowledge_base()""",
            func=clear_rag_knowledge_base
        )
    ]
    
    return tools

def main():
    # 初始化工具集合
    tools = create_rag_tools(model=None)  # 如果没有模型，可以传 None

    # 获取具体工具函数
    add_document_to_rag = next(t.func for t in tools if t.name == "add_document_to_rag")
    add_directory_to_rag = next(t.func for t in tools if t.name == "add_directory_to_rag")
    rag_question_answer = next(t.func for t in tools if t.name == "rag_question_answer")
    get_rag_stats = next(t.func for t in tools if t.name == "get_rag_stats")
    delete_rag_document = next(t.func for t in tools if t.name == "delete_rag_document")
    clear_rag_knowledge_base = next(t.func for t in tools if t.name == "clear_rag_knowledge_base")

    # 测试单个文档添加
    file_path = "./doc/rag_usage_guide.md"
    print(add_document_to_rag(file_path))

    # 测试目录添加
    # dir_path = "./docs/"
    # print(add_directory_to_rag(dir_path))

    # 测试问题回答
    question = "证券法规定了哪些信息披露义务？"
    print(rag_question_answer(question))

    # 查看知识库统计信息
    print(get_rag_stats())

    # 测试删除文档
    # print(delete_rag_document(file_path))

    # 测试清空知识库
    # print(clear_rag_knowledge_base())

if __name__ == "__main__":
    main()