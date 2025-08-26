import os
import logging
import time
from tqdm import tqdm
# 统一导入所有可能用到的PDF加载器
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PyPDFLoader,
    PDFPlumberLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_config import LoggerManager

logger = LoggerManager().logger


class PDFProcessor:
    def __init__(self, directory, db_type='vector', **kwargs):
        """
        初始化 PDF 处理器
        :param directory: PDF 文件所在目录
        :param db_type: 数据库类型 ('vector' 或 'es')
        :param kwargs: 其他参数
        """
        self.directory = directory  # PDF 文件所在目录
        self.db_type = db_type  # 数据库类型
        self.file_group_num = kwargs.get('file_group_num', 20)  # 每组处理的文件数
        self.batch_num = kwargs.get('batch_num', 6)  # 每次插入的批次数量
        self.chunksize = kwargs.get('chunksize', 1024)  # 切分文本的大小
        self.overlap = kwargs.get('overlap', 100)  # 切分文本的重叠大小
        # 统计信息
        self.success_count = 0
        self.fail_count = 0
        self.failed_files = []
        
        logger.info(f"""
                    初始化PDF文件导入器:
                    配置参数：
                    - 导入的文件路径：{self.directory}
                    - 每次处理文件数：{self.file_group_num}
                    - 每批次处理样本数：{self.batch_num}
                    - 切分文本的大小：{self.chunksize}
                    - 切分文本重叠大小：{self.overlap}
                    """)

        # 根据数据库类型初始化相应的客户端
        if db_type == 'vector':
            self.vector_db = kwargs.get('vector_db')  # 向量数据库实例
            self.es_client = None
            if not self.vector_db:
                raise ValueError("vector_db 参数未提供 for db_type='vector'")
            logger.info(f'导入的目标数据库为：向量数据库')
        elif db_type == 'es':
            self.vector_db = None
            self.es_client = kwargs.get('es_client')  # Elasticsearch 客户端
            if not self.es_client:
                raise ValueError("es_client 参数未提供 for db_type='es'")
            logger.info(f'导入的目标数据库为：ES数据库')
        else:
            raise ValueError("db_type must be either 'vector' or 'es'.")

    def load_pdf_files(self):
        """
        加载目录下的所有PDF文件
        """
        pdf_files = []
        if not os.path.exists(self.directory):
            logger.error(f"目录不存在: {self.directory}")
            return pdf_files
            
        for file in os.listdir(self.directory):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.directory, file)
                pdf_files.append(pdf_path)

        logger.info(f"找到 {len(pdf_files)} 个PDF文件.")
        return pdf_files

    def load_pdf_content(self, pdf_path):
        """
        读取PDF文件内容，使用多种加载器作为备用
        """
        loaders = [
            ("PyMuPDFLoader", PyMuPDFLoader),
            ("PyPDFLoader", PyPDFLoader),
            ("PDFPlumberLoader", PDFPlumberLoader)
        ]
        
        for loader_name, LoaderClass in loaders:
            try:
                loader = LoaderClass(file_path=pdf_path)
                docs = loader.load()
                logger.info(f"使用 {loader_name} 成功加载: {os.path.basename(pdf_path)}")
                self.success_count += 1
                return docs
            except Exception as e:
                logger.warning(f"{loader_name} 加载失败 {os.path.basename(pdf_path)}: {str(e)}")
                continue
                
        # 所有加载器都失败
        logger.error(f"所有加载器均无法加载文件: {os.path.basename(pdf_path)}")
        self.fail_count += 1
        self.failed_files.append(pdf_path)
        return []

    def split_text(self, documents):
        """
        将文本切分成小段
        """
        if not documents:
            logger.warning("没有可切分的文档")
            return []
            
        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunksize,
            chunk_overlap=self.overlap,
            length_function=len,
            add_start_index=True,
        )
        docs = text_splitter.split_documents(documents)

        logger.info(f"将文本切分成 {len(docs)} 个片段")
        return docs

    def insert_docs(self, docs, insert_function, batch_size=None):
        """
        将文档插入到指定的数据库，并显示进度
        :param docs: 要插入的文档列表
        :param insert_function: 插入函数
        :param batch_size: 批次大小
        """
        if not docs:
            logger.warning("没有可插入的文档")
            return
            
        if batch_size is None:
            batch_size = self.batch_num

        logger.info(f"准备插入 {len(docs)} 个文档")
        start_time = time.time()
        total_docs_inserted = 0

        total_batches = (len(docs) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="插入批次进度", unit="批") as pbar:
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                try:
                    insert_function(batch)  # 调用传入的插入函数
                    total_docs_inserted += len(batch)
                    
                    # 计算并显示当前的TPM (每分钟处理数量)
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        tpm = (total_docs_inserted / elapsed_time) * 60
                        pbar.set_postfix({"TPM": f"{tpm:.2f}"})
                        
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"插入批次 {i//batch_size + 1} 失败: {str(e)}")
                    # 可以选择是否重试或跳过
                    pbar.update(1)

        logger.info(f"完成插入，成功插入 {total_docs_inserted}/{len(docs)} 个文档")

    def insert_to_vector_db(self, docs):
        """
        将文档插入到 VectorDB
        """
        try:
            self.vector_db.add_with_langchain(docs)
        except Exception as e:
            logger.error(f"向量数据库插入失败: {str(e)}")
            raise  # 抛出异常让上层处理

    def insert_to_elasticsearch(self, docs):
        """
        将文档插入到 Elasticsearch
        """
        try:
            self.es_client.add_documents(docs)
        except Exception as e:
            logger.error(f"Elasticsearch 插入失败: {str(e)}")
            raise  # 抛出异常让上层处理

    def process_pdfs_group(self, pdf_files_group):
        """处理一组PDF文件"""
        if not pdf_files_group:
            logger.warning("处理的文件组为空")
            return
            
        logger.info(f"开始处理文件组，共 {len(pdf_files_group)} 个文件")
        pdf_contents = []

        for pdf_path in pdf_files_group:
            # 读取PDF文件内容（已包含错误处理）
            documents = self.load_pdf_content(pdf_path)
            pdf_contents.extend(documents)

        # 将文本切分成小段
        docs = self.split_text(pdf_contents)

        if not docs:
            logger.warning("当前文件组没有生成可插入的文档片段")
            return

        # 根据数据库类型插入文档
        try:
            if self.db_type == 'vector':
                self.insert_docs(docs, self.insert_to_vector_db)
            elif self.db_type == 'es':
                self.insert_docs(docs, self.insert_to_elasticsearch)
        except Exception as e:
            logger.error(f"文件组处理失败: {str(e)}")

    def process_pdfs(self):
        """处理所有PDF文件"""
        start_time = time.time()
        # 获取目录下所有的PDF文件
        pdf_files = self.load_pdf_files()
        
        if not pdf_files:
            logger.warning("没有找到任何PDF文件，终止处理")
            return

        group_num = self.file_group_num
        total_groups = (len(pdf_files) + group_num - 1) // group_num
        logger.info(f"将 {len(pdf_files)} 个文件分为 {total_groups} 组进行处理")

        # 按组处理PDF文件
        for i in range(0, len(pdf_files), group_num):
            pdf_files_group = pdf_files[i:i + group_num]
            group_index = i // group_num + 1
            logger.info(f"开始处理第 {group_index}/{total_groups} 组文件")
            self.process_pdfs_group(pdf_files_group)

        # 处理完成后输出统计信息
        elapsed_time = time.time() - start_time
        logger.info(f"""
                    PDF处理完成!
                    总耗时: {elapsed_time:.2f} 秒
                    成功加载: {self.success_count} 个文件
                    加载失败: {self.fail_count} 个文件
                    """)
        
        if self.failed_files:
            logger.warning(f"加载失败的文件列表: {', '.join([os.path.basename(f) for f in self.failed_files])}")
        
        return {
            "total": len(pdf_files),
            "success": self.success_count,
            "fail": self.fail_count,
            "failed_files": self.failed_files,
            "time_seconds": elapsed_time
        }
