import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.embeddings import OllamaEmbeddings
from chromadb.errors import NotFoundError
class VectorDatabase:
    """
    向量数据库管理器 - 基于ChromaDB实现RAG功能
    """
    
    def __init__(self, 
                 persist_directory: str = "./rag_data/vector_db",
                 collection_name: str = "documents",
                 embedding_model: str = "embeddinggemma:300m",
                 metric: str = "cosine"  # 新增参数，用以指定度量空间
                 ):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 数据库持久化目录
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
            metric: 向量相似度度量方式，如 "cosine", "l2", "ip"
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.metric = metric
        self.embedding_model_name = embedding_model
        
        # # 初始化嵌入模型
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=embedding_model,
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        self.embeddings = OllamaEmbeddings(model="embeddinggemma:300m")  # 或者 qwen2-embeddings
        # self.embeddings = embedding_functions.OllamaEmbeddingFunction(
        #     url="http://localhost:11434/api/embeddings",
        #     model_name="embeddinggemma:300m",
        # )
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"✅ 已找到 collection: {self.collection_name}")
            # 检查 metadata / metric 是否一致
            existing_meta = self.collection.metadata or {}
            existing_metric = existing_meta.get("hnsw:space")
            if existing_metric and existing_metric != self.metric:
                raise ValueError(
                    f"已存在 collection '{self.collection_name}'，但 metric 不一致 "
                    f"(已有: {existing_metric}, 期望: {self.metric})"
                )
        except NotFoundError:
            # 如果不存在，就新建
            print(f"⚠️ collection '{self.collection_name}' 不存在，正在新建...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.metric}
            )
        
        # 初始化向量存储
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        # 元数据存储
        self.metadata_file = self.persist_directory / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 加载元数据失败: {e}")
        return {
            "documents": {},
            "collections": {},
            "stats": {
                "total_documents": 0,
                "total_chunks": 0,
                "last_updated": None
            }
        }
    
    def _save_metadata(self):
        """保存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存元数据失败: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def add_documents(self, documents: List[Document], 
                     source_file: Optional[str] = None) -> bool:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表
            source_file: 源文件路径（可选）
            
        Returns:
            bool: 是否成功添加
        """
        try:
            if not documents:
                print("⚠️ 没有文档需要添加")
                return False
            
            # ✅ 确保持久化目录存在
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            # ✅ 确保 collection 已初始化
            if not hasattr(self, "collection") or self.collection is None:
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                except Exception:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    
            # 检查文件是否已更新
            if source_file and os.path.exists(source_file):
                file_hash = self._calculate_file_hash(source_file)
                file_key = os.path.abspath(source_file)
                
                if file_key in self.metadata["documents"]:
                    if self.metadata["documents"][file_key]["hash"] == file_hash:
                        print(f"📄 文件未变更，跳过处理: {os.path.basename(source_file)}")
                        return True
                    else:
                        # 文件已更新，先删除旧的块
                        self._remove_document(file_key)
                
                # 更新元数据
                self.metadata["documents"][file_key] = {
                    "hash": file_hash,
                    "chunks": len(documents),
                    "processed_at": str(Path().absolute())
                }
            
            # 为每个文档添加唯一ID和元数据
            doc_ids = []
            for i, doc in enumerate(documents):
                doc_id = f"{hashlib.md5(doc.page_content.encode()).hexdigest()}_{i}"
                doc_ids.append(doc_id)
                
                # 增强文档元数据
                if source_file:
                    doc.metadata.update({
                        "source_file": os.path.abspath(source_file),
                        "chunk_index": i,
                        "doc_id": doc_id
                    })
                else:
                    doc.metadata.update({
                        "chunk_index": i,
                        "doc_id": doc_id
                    })
            
            # 添加到向量存储
            self.vector_store.add_documents(documents, ids=doc_ids)
            
            # 更新统计信息
            self.metadata["stats"]["total_chunks"] += len(documents)
            if source_file:
                self.metadata["stats"]["total_documents"] += 1
            
            self._save_metadata()
            
            print(f"✅ 成功添加 {len(documents)} 个文档块到向量数据库")
            return True
            
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return False
    
    def _remove_document(self, file_key: str):
        """删除指定文件的所有文档块"""
        try:
            # 查询该文件的所有文档块
            results = self.collection.get(
                where={"source_file": file_key}
            )
            
            if results["ids"]:
                # 删除所有相关的文档块
                self.collection.delete(ids=results["ids"])
                print(f"🗑️ 已删除文件 {os.path.basename(file_key)} 的 {len(results['ids'])} 个旧文档块")
                
                # 更新统计信息
                self.metadata["stats"]["total_chunks"] -= len(results["ids"])
                
        except Exception as e:
            print(f"⚠️ 删除旧文档块失败: {e}")
    
    def similarity_search(self, query: str, k: int = 4, 
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件
            
        Returns:
            List[Document]: 相关文档列表
        """
        try:
            # 使用向量存储进行搜索
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        带相似度分数的搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 相似度分数) 列表
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            collection_count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": len(self.metadata["documents"]),
                "total_chunks": collection_count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有已索引的文档"""
        documents = []
        for file_path, info in self.metadata["documents"].items():
            documents.append({
                "file_path": file_path,
                "chunks": info["chunks"],
                "processed_at": info.get("processed_at", "未知"),
                "file_name": os.path.basename(file_path)
            })
        return documents
    
    def delete_document(self, file_path: str) -> bool:
        """
        删除指定文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否成功删除
        """
        try:
            file_key = os.path.abspath(file_path)
            if file_key not in self.metadata["documents"]:
                print(f"⚠️ 文档不存在: {file_path}")
                return False
            
            self._remove_document(file_key)
            
            # 从元数据中删除
            del self.metadata["documents"][file_key]
            self.metadata["stats"]["total_documents"] -= 1
            
            self._save_metadata()
            
            print(f"✅ 成功删除文档: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            print(f"❌ 删除文档失败: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        清空所有内容：删除 collection 中的所有文档块 + 清空 metadata，
        但保留 collection 本身与 persist 目录。
        """
        try:
            # 1️⃣ 获取 collection
            if not hasattr(self, "collection") or self.collection is None:
                self.collection = self.client.get_collection(name=self.collection_name)
            
            # 2️⃣ 获取 collection 中所有 ids
            results = self.collection.get()   # ✅ 不要传 include=["ids"]
            all_ids = results.get("ids", [])
            
            # 3️⃣ 删除所有文档块
            if all_ids:
                self.collection.delete(ids=all_ids)
                print(f"🗑️ 删除了 {len(all_ids)} 个文档块（chunks）")
            
            # 4️⃣ 重置元数据
            self.metadata = {
                "documents": {},
                "collections": {},
                "stats": {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "last_updated": None
                }
            }
            self._save_metadata()
            
            # 5️⃣ Optionally：重建 vector_store 引用
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )

            print("✅ 已清空 collection 内容和元数据，保留 collection 与目录")
            return True
        except Exception as e:
            print(f"❌ 清空内容失败: {e}")
            return False

        
    def close(self):
        """
        主动释放/关闭 Chroma 客户端，解除 SQLite 文件锁。
        """
        try:
        # 删除引用，等待 GC 回收
            self.client = None
            self.collection = None
            self.vector_store = None

            import gc, time
            gc.collect()
            time.sleep(0.5)  # 给 Windows 一点时间释放文件锁

            print("VectorDatabase 已关闭，SQLite 文件锁已释放。")
        except Exception as e:
            print(f"关闭 VectorDatabase 出错: {e}")

def create_vector_database(persist_directory: str = "./rag_data/vector_db",
                          collection_name: str = "documents") -> VectorDatabase:
    """
    创建向量数据库实例
    
    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称
        
    Returns:
        VectorDatabase: 向量数据库实例
    """
    return VectorDatabase(
        persist_directory=persist_directory,
        collection_name=collection_name
    )

def main():
    # 1️⃣ 初始化数据库
    db = create_vector_database()

    # 2️⃣ 构造示例文档
    docs = [
        Document(page_content="苹果是一种常见的水果，富含维生素。", metadata={"category": "fruit"}),
        Document(page_content="篮球是一项流行的运动，起源于美国。", metadata={"category": "sport"}),
        Document(page_content="Python 是一种流行的编程语言，适合数据科学和人工智能。", metadata={"category": "tech"})
    ]

    # 3️⃣ 添加文档
    print("\n>>> 添加文档")
    db.add_documents(docs, source_file="doc/demo.txt")

    # 4️⃣ 相似性搜索
    print("\n>>> 相似性搜索：'编程语言'")
    results = db.similarity_search("编程语言", k=2)
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content} | metadata={doc.metadata}")

    # 5️⃣ 查看统计信息
    print("\n>>> 数据库统计信息")
    print(db.get_stats())

    # 6️⃣ 列出文档
    print("\n>>> 已索引的文档")
    for doc in db.list_documents():
        print(doc)

    # 7️⃣ 删除文档
    print("\n>>> 删除文档 demo.txt")
    db.delete_document("doc\\demo.txt")

    # 8️⃣ 再次查看统计
    print("\n>>> 删除后的统计信息")
    print(db.get_stats())

    # 9️⃣ 清空所有数据
    print("\n>>> 清空整个数据库")
    db.clear_all()

    # 🔟 关闭数据库连接
    db.close()


if __name__ == "__main__":
    main()