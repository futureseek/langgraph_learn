# quick_test_rag.py
import ollama
import chromadb
from pathlib import Path

# 初始化 Chroma 数据库
chroma_db_path = Path("rag_data/vector_db")
client = chromadb.PersistentClient(path=str(chroma_db_path))
collection = client.get_or_create_collection("docs")

def embed_text(text: str):
    try:
        response = ollama.embed(model="embeddinggemma:300m", input=text)
        # print("Embedding response:", response)  # Debug
        return response["embeddings"][0]
    except Exception as e:
        print("Embed error:", e)
        raise
def quick_test(query: str):
    """执行一次检索测试"""
    print("📂 正在执行查询...")
    query_emb = embed_text(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    if not results["documents"] or len(results["documents"][0]) == 0:
        print("⚠️ 没有找到相关文档")
        return

    print("📊 检索结果:")
    for i, doc in enumerate(results["documents"][0], 1):
        print(f"--- 结果 {i} ---")
        print(doc[:500])  # 截取前500字
        print()

def dump_all_chunks(to_file: bool = False):
    """输出数据库里所有文档块"""
    print("📂 正在获取所有文档块...")
    results = collection.get()
    docs = results["documents"]

    if not docs:
        print("⚠️ 数据库为空")
        return

    print(f"📊 数据库包含 {len(docs)} 个文档块\n")
    for i, doc in enumerate(docs, 1):
        print(f"=== 文档块 {i} ===")
        print(doc[:500])
        print()

    if to_file:
        with open("all_chunks.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs, 1):
                f.write(f"=== 文档块 {i} ===\n")
                f.write(doc + "\n\n")
        print("✅ 已将所有文档块导出到 all_chunks.txt")

if __name__ == "__main__":
    # 测试查询
    test_question = "证券公司承销证券有哪些禁止行为？"
    quick_test(test_question)

    # 输出数据库全部内容
    dump_all_chunks(to_file=True)
