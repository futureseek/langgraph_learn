import os
import shutil
import time
import gc
from pathlib import Path

# 修改为你的数据库持久化目录
PERSIST_DIR = Path("./vector_store")

def clear_db():
    """独立清理 Chroma 向量数据库"""
    try:
        # 1️⃣ 强制释放 Python 引用
        gc.collect()
        time.sleep(0.5)

        # 2️⃣ 删除持久化目录（逐文件删除，避免 WinError 32）
        if PERSIST_DIR.exists():
            for root, dirs, files in os.walk(PERSIST_DIR, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    try:
                        os.remove(file_path)
                        print(f"🗑️ 已删除文件: {file_path}")
                    except Exception as e:
                        print(f"⚠️ 无法删除文件 {file_path}: {e}")
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        os.rmdir(dir_path)
                        print(f"📂 已删除目录: {dir_path}")
                    except Exception as e:
                        print(f"⚠️ 无法删除目录 {dir_path}: {e}")
            try:
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                print(f"✅ 已清空持久化目录: {PERSIST_DIR}")
            except Exception as e:
                print(f"⚠️ 无法删除目录 {PERSIST_DIR}: {e}")
        else:
            print("ℹ️ 持久化目录不存在，无需清理")

        # 3️⃣ 重新创建空目录
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        print("📂 已重建空的持久化目录")

    except Exception as e:
        print(f"❌ 清空数据库失败: {e}")

if __name__ == "__main__":
    clear_db()
