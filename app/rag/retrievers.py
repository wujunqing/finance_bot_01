from langchain_core.callbacks import CallbackManagerForRetrieverRun
from utils.logger_config import LoggerManager
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from rag.elasticsearch_db import ElasticsearchDB
# ES需要导入的库
from typing import List
import logging
import settings
from langchain_community.document_transformers import (
    LongContextReorder,
)
from utils.util import get_rerank_model


# 使用llamaindex的向量索引进行增强
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from utils.logger_config import LoggerManager





logger = LoggerManager().logger


class SimpleRetrieverWrapper():
    """自定义检索器实现"""

    def __init__(self, store, llm, **kwargs):
        self.store = store
        self.llm = llm
        logger.info(f'检索器所使用的Chat模型：{self.llm}')

    def create_retriever(self):
        logger.info(f'初始化自定义的Retriever')

        # 初始化一个空的检索器列表
        retrievers = []
        weights = []

        # Step1：创建一个 多路召回检索器 MultiQueryRetriever
        chromadb_retriever = self.store.as_retriever()
        mq_retriever = MultiQueryRetrieverWrapper.from_llm(retriever=chromadb_retriever, llm=self.llm)

        # Step2：创建一个 上下文压缩检索器ContextualCompressionRetriever
        if settings.COMPRESSOR_ENABLE is True:
            compressor = LLMChainExtractor.from_llm(llm=self.llm)
            compression_retriever = ContextualCompressionRetrieverWrapper(
                base_compressor=compressor, base_retriever=mq_retriever
            )
            # 开启开关就使用压缩检索器
            retrievers.append(compression_retriever)
            weights.append(0.5)
            logger.info(f'已启用 ContextualCompressionRetriever')
        else:
            # 关闭开关就使用多路召回检索器
            retrievers.append(mq_retriever)
            weights.append(0.5)
            logger.info(f'已启用 MultiQueryRetriever')

        # Step3：创建一个 ES 检索器
        if settings.ELASTIC_ENABLE_ES is True:
            es_retriever = ElasticsearchRetriever()
            retrievers.append(es_retriever)
            weights.append(0.5)
            logger.info(f'已启用 ElasticsearchRetriever')
        
        # 使用集成检索器，将所有启用的检索器集合在一起
        ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
        return ensemble_retriever

class ElasticsearchRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, ) -> List[Document]:
        """Return the first k documents from the list of documents"""
        es_connector = ElasticsearchDB()
        query_result = es_connector.search(query)
        
        # 增加长上下文重排序
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(query_result)
        # logger.info(f"ElasticSearch检索到的原始文档：")
        # for poriginal in query_result:
        #     logger.info(f"{poriginal}")

        logger.info(f"ElasticSearch检索重排后的文档：")
        for preordered in reordered_docs:
            logger.info(f"{preordered}")

        logger.info(f"ElasticSearch检索到资料文件个数：{len(query_result)}")

        if reordered_docs:
            return [Document(page_content=doc) for doc in reordered_docs]
        return []

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """(Optional) async native implementation."""
        es_connector = ElasticsearchDB()
        query_result = es_connector.search(query)
        if query_result:
            return [Document(page_content=doc) for doc in query_result]
        return []



# class MultiQueryRetrieverWrapper(MultiQueryRetriever):
#     def _get_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun,
#     ) -> List[Document]:
#         """
#         对MultiQueryRetriever进行重写，增加日志打印
#         """
#         queries = self.generate_queries(query, run_manager)
#         if self.include_original:
#             queries.append(query)
#         documents = self.retrieve_documents(queries, run_manager)

#         # 增加长上下文重排序
#         reordering = LongContextReorder()
#         reordered_docs = reordering.transform_documents(documents)

#         logger.info(f'MultiQuery生成的检索语句：')
#         for q in queries:
#             logger.info(f"{q}")
#         logger.info(f'MultiQuery检索到的资料文件：')
#         for doc in documents:
#             logger.info(f"{doc}")
#         logger.info(f"MultiQuery检索到资料文件个数：{len(documents)}")


#         # 使用llamaindex的向量索引进行增强

#         return self.unique_union(reordered_docs)
#         # return self.unique_union(documents)





class MultiQueryRetrieverWrapper(MultiQueryRetriever):
    # 核心修复1：仅保留 model_config，允许动态添加属性（不显式声明属性，避免Pydantic必填校验）
    model_config = {
        "extra": "allow"  # 关键：允许子类动态添加父类未定义的属性，且不触发必填校验
    }

    def __init__(self, 
                 *args, 
                 # 核心修复2：将子类需要的参数设为“默认参数”，避免Pydantic校验必填
                 chroma_db_path: str = "chroma_db",  # 与你之前VectorDB的持久化路径一致
                 llama_collection_name: str = "llama_index",  # 与你之前的集合名一致
                 embed_model=settings.EMBED,  # 接收embedding模型（适配你的DashScope）
                 **kwargs):
        # 1. 先调用父类构造函数（必须放在最前面，初始化父类的核心属性）
        super().__init__(*args, **kwargs)

        # 2. 动态添加子类属性（Pydantic会通过model_config允许这些属性）
        # Chroma数据库相关
        self.persist_path = chroma_db_path
        self.llama_collection_name = llama_collection_name
        
        # 初始化Chroma客户端和集合（复用你之前的本地持久化配置）
        self.llama_client = chromadb.PersistentClient(path=self.persist_path)
        self.llama_collection = self.llama_client.get_or_create_collection(
            name=self.llama_collection_name
        )

        # 3. 初始化llama_index向量存储和索引（传入embedding模型避免维度不匹配）
        self.llama_vector_store = ChromaVectorStore(chroma_collection=self.llama_collection)
        self.llama_index = VectorStoreIndex.from_vector_store(
            vector_store=self.llama_vector_store,
            embed_model=embed_model  # 关键：与Chroma数据库使用相同的embedding模型
        )

        # 4. 创建llama_index检索器（配置检索参数）
        self.llama_retriever = self.llama_index.as_retriever(
            similarity_top_k=5,  # 检索Top5相似结果（可根据需求调整）
            # 可选：添加元数据过滤（如只检索特定PDF）
            # filter={"file_name": {"$in": ["2010_finance_table.pdf"]}}
        )


    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """重写检索逻辑：合并LangChain MultiQuery与llama_index结果"""
        # 1. 原有LangChain MultiQuery检索
        queries = self.generate_queries(query, run_manager)
        if self.include_original:
            queries.append(query)
        langchain_docs = self.retrieve_documents(queries, run_manager)

        # 2. 长上下文重排序（保留原有逻辑）
        reordering = LongContextReorder()
        reordered_langchain_docs = reordering.transform_documents(langchain_docs)

        # 3. llama_index增强检索（异常处理避免流程中断）
        logger.info(f"✅ 使用llama_index增强检索，查询: {query}")
        converted_llama_docs = []
        try:
            llama_docs = self.llama_retriever.retrieve(query)
            logger.info(f"llama_index检索到 {len(llama_docs)} 个文档")

            # 4. 格式转换：llama_index Document → LangChain Document
            for doc in llama_docs:
                metadata = doc.metadata or {}
                # 适配你之前PDF导入的元数据格式（优先取file_name，其次取source）
                source = metadata.get("file_name") or metadata.get("source") or "unknown"

                converted_doc = Document(
                    page_content=doc.text,  # llama_index的text → LangChain的page_content
                    metadata={
                        "source": source,
                        "similarity_score": round(doc.score, 4),  # 保留相似度分数
                        "retriever": "llama_index",  # 标记检索来源
                        "page": metadata.get("page", "unknown")  # 保留页码（若有）
                    }
                )
                converted_llama_docs.append(converted_doc)
        except Exception as e:
            logger.error(f"llama_index检索失败: {str(e)}", exc_info=True)
            converted_llama_docs = []  # 失败时返回空列表，不影响原有检索流程

        # 5. 合并结果并去重（基于内容清洗后的哈希，避免格式差异导致的重复）
        all_docs = reordered_langchain_docs + converted_llama_docs
        unique_docs = self._remove_duplicates(all_docs)

        # 6. 日志打印（增强可观测性）
        logger.info(f"\n===== 检索统计 =====")
        logger.info(f"MultiQuery生成 {len(queries)} 条检索语句: {queries}")
        logger.info(f"LangChain检索文档数: {len(reordered_langchain_docs)}")
        logger.info(f"llama_index检索文档数: {len(converted_llama_docs)}")
        logger.info(f"合并去重后文档数: {len(unique_docs)}")
        logger.info("====================\n")

        return unique_docs


    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """基于内容清洗的去重逻辑（提升准确性）"""
        seen_hashes = set()
        unique_docs = []
        for doc in docs:
            # 清洗内容：去除空格、换行，避免格式差异导致的重复
            cleaned_content = doc.page_content.strip().replace("\n", "").replace(" ", "")
            content_hash = hash(cleaned_content)

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        # 打印去重效果
        duplicate_count = len(docs) - len(unique_docs)
        if duplicate_count > 0:
            logger.info(f"去重完成：移除 {duplicate_count} 个重复文档")
        return unique_docs


from typing import Any, List
class ContextualCompressionRetrieverWrapper(ContextualCompressionRetriever):
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """
        对ContextualCompressionRetriever进行重写，增加日志打印
        """

        docs = self.base_retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        if docs:
            compressed_docs = self.base_compressor.compress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            logger.info(f'压缩后的文档长度：{len(compressed_docs)}')
            logger.info(f'压缩后的文档：{compressed_docs}')
            return list(compressed_docs)
        else:
            return []
    