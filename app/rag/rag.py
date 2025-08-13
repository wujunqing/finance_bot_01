import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from .retrievers import SimpleRetrieverWrapper
from .vector_db import ChromaDB  # 导入 VectorDB 子类
from .elasticsearch_db import TraditionDB
from utils.logger_config import LoggerManager
import settings
from langchain_community.document_transformers import (
    LongContextReorder,
)
# 配置日志记录
logger = LoggerManager().logger


class RagManager:
    def __init__(self,
                 vector_db_class=ChromaDB,  # 默认使用 ChromaDB
                 es_db=TraditionDB,
                 db_config=None,  # 数据库配置参数
                 llm=None, embed=None,
                 retriever_cls=SimpleRetrieverWrapper, **retriever_kwargs):
        self.llm = llm
        self.embed = embed
        logger.info(f'初始化llm大模型：{self.llm}')
        logger.info(f'初始化embed模型：{self.embed}')

        # 如果没有提供 db_config，使用默认配置
        if db_config is None:
            db_config = {
                "chroma_server_type": settings.CHROMA_SERVER_TYPE,
                "host": settings.CHROMA_HOST,
                "port": settings.CHROMA_PORT,
                "persist_path": settings.CHROMA_PERSIST_DB_PATH,
                "collection_name": settings.CHROMA_COLLECTION_NAME
            }
            logger.info(f'初始化向量数据库配置：{db_config}')

        # 创建向量数据库实例
        self.vector_db = vector_db_class(**db_config, embed=self.embed)
        self.store = self.vector_db.get_store()

        self.retriever_instance = retriever_cls(self.store, self.llm, **retriever_kwargs)
        logger.info(f'使用的检索器类: {retriever_cls.__name__}')

    def get_chain(self, retriever):
        """获取RAG查询链"""
        prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an assistant for question-answering tasks. Use the following pieces 
          of retrieved context to answer the question. 
          If you don't know the answer, just say that you don't know. 
          Use three sentences maximum and keep the answer concise.
          Question: {question} 
          Context: {context} 
          Answer:""")
        ])
        format_docs_runnable = RunnableLambda(self.format_docs)
        rag_chain = (
                {"context": retriever | format_docs_runnable,
                 "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return rag_chain

    def format_docs(self, docs):
        """格式化文档"""

        retrieved_content = "\n\n".join(doc.page_content for doc in docs)
        logger.info(f"检索到的资料为:\n{retrieved_content}")

        # 对于ES查询没有source的情况进行特殊处理
        try:
            retrieved_files = "\n".join([doc.metadata["source"] for doc in docs])
            logger.info(f"资料文件分别是:\n{retrieved_files}")
        except Exception as e:
            logger.info(f"处理查询时没有找到source字段: {e}")

        logger.info(f"检索到资料文件个数：{len(docs)}")

        return retrieved_content

    def get_result(self, question):
        """获取RAG查询结果"""
        retriever = self.retriever_instance.create_retriever()
        rag_chain = self.get_chain(retriever)

        try:
            result = rag_chain.invoke(input=question)
            logger.info(f"RAG查询结果：{result}")

            return result
        except Exception as e:
            logger.error(f"查询时发生错误：{e}")
            return f'{question} 查询时发生错误：{e}'
