import chromadb
from langchain_chroma import Chroma
from langchain_community.vectorstores.milvus import Milvus
from utils.logger_config import LoggerManager

logger = LoggerManager().logger


class VectorDB:
    def add_with_langchain(self, docs):
        """
        将文档添加到数据库
        """
        raise NotImplementedError("Subclasses should implement this method!")

    def get_store(self):
        """
        获得向量数据库的对象实例
        """
        raise NotImplementedError("Subclasses should implement this method!")


class ChromaDB(VectorDB):
    def __init__(self,
                 chroma_server_type="local",
                 host="localhost", port=8000,
                 persist_path="chroma_db",
                 collection_name="langchain",
                 embed=None):

        self.host = host
        self.port = port
        self.path = persist_path
        self.embed = embed
        self.store = None

        if chroma_server_type == "http":
            client = chromadb.HttpClient(host=host, port=port)
            self.store = Chroma(collection_name=collection_name,
                                embedding_function=self.embed,
                                client=client)
            logger.info(f'Chroma数据库连接成功，连接方式：{chroma_server_type}, host:{self.host}, port:{self.port}, embed:{self.embed}')

        elif chroma_server_type == "local":
            self.store = Chroma(collection_name=collection_name,
                                embedding_function=self.embed,
                                persist_directory=persist_path)
            logger.info(f'Chroma数据库连接成功，连接方式：{chroma_server_type}, 本地持久化路径：{persist_path}, embed:{self.embed}')

        if self.store is None:
            raise ValueError("Chroma store init failed!")

        logger.info(f'Chroma数据库的Collectname：{collection_name}')
        logger.info(f'Chroma数据库所使用的embed模型：{self.embed}')

    def add_with_langchain(self, docs):
        self.store.add_documents(documents=docs)

    def get_store(self):
        return self.store


class MilvusDB(VectorDB):
    def __init__(self,
                 milvus_server_type="http",
                 host="localhost", port=19530,
                 collection_name="LangChainCollection",
                 embed=None):

        self.host = host
        self.port = port
        self.embed = embed
        self.collection_name = collection_name

        if milvus_server_type == "http":
            self.store = Milvus(
                collection_name=self.collection_name,
                connection_args={"uri": f"http://{host}:{port}"},
                embedding_function=self.embed,
                auto_id=True
            )
        else:
            # 目前没有找到直接连本地 Milvus 的方法
            self.store = Milvus(
                collection_name=self.collection_name,
                connection_args={"uri": f"tcp://{host}:{port}"},
                embedding_function=self.embed,
                auto_id=True
            )

    def add_with_langchain(self, docs):
        self.store.add_documents(documents=docs)

    def get_store(self):
        return self.store
