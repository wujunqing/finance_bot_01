import requests
import chromadb
from chromadb.config import Settings
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import CHROMA_SERVER_TYPE, CHROMA_PERSIST_DB_PATH

def check_chroma_status():
    try:
        if CHROMA_SERVER_TYPE == "local":
            # 使用本地模式
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DB_PATH)
            print(f"✅ ChromaDB 本地模式运行正常")
        else:
            # 使用服务器模式
            client = chromadb.HttpClient(host="localhost", port=8000)
            print(f"✅ ChromaDB 服务运行正常")
        
        collections = client.list_collections()
        print(f"📊 集合数量: {len(collections)}")
        for collection in collections:
            print(f"   - {collection.name}: {collection.count()} 条记录")
        return True
    except Exception as e:
        print(f"❌ ChromaDB 服务未运行或连接失败: {e}")
        return False

if __name__ == "__main__":
    check_chroma_status()