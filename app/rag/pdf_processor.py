import os
import logging
import time
import uuid
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_config import LoggerManager
from llama_parse import LlamaParse
from langchain.schema import Document as LangChainDocument  # 导入标准 Document 类
from llama_index.core import Document as LlamaIndexDocument
import settings
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
        self.table_directory = directory + "/table"
        self.db_type = db_type  # 数据库类型
        self.file_group_num = kwargs.get('file_group_num', 20)  # 每组处理的文件数
        self.batch_num = kwargs.get('batch_num', 6)  # 每次插入的批次数量
        self.chunksize = kwargs.get('chunksize', 500)  # 切分文本的大小
        self.overlap = kwargs.get('overlap', 100)  # 切分文本的重叠大小
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

            logger.info(f'导入的目标数据库为：向量数据库')
        elif db_type == 'es':
            self.vector_db = None
            #self.es_client = kwargs.get('es_client')  # Elasticsearch 客户端
            self.es_client = kwargs.get('es_client')  # Elasticsearch 客户端

            logger.info(f'导入的目标数据库为：ES数据库')
        else:
            raise ValueError("db_type must be either 'vector' or 'es'.")
        
        # 通过LlamaParse提取表格类PDF
        os.environ['LLAMA_CLOUD_API_KEY'] = settings.LLAMA_CLOUD_API_KEY
        logger.info(f'✅ Llama API密钥配置完成')

        self.finance_parser = LlamaParse(
                        result_type="markdown",
                        # 启用图表提取
                        extract_charts=True,
    
                        # 启用自动模式（智能选择解析策略）
                        auto_mode=True,
    
                        # 当页面包含图像时触发自动模式
                        auto_mode_trigger_on_image_in_page=True,
    
                        # 当页面包含表格时触发自动模式
                        auto_mode_trigger_on_table_in_page=True,
                        user_prompt="""
                        ## 1. 核心目标
                        将财务报表（含年报、季报、财务附注、审计报告）解析为 Markdown 格式，优先保证**数据准确性、维度完整性、术语标准化**，确保后续检索时能精准定位“指标+时间+业务+单位”四要素。

                        ## 2. 表格处理强制规则（重中之重）
                        ### 2.1 列名必须完整显性化
                        - 所有表格列名需包含「维度+指标+单位」三要素（缺失则补充）：
                        - 时间列：标注具体周期（如“会计期间（年度/季度/月度）”“2024年Q1”而非“期间”）；
                        - 指标列：用财务标准术语（如“归属于母公司股东的净利润（万元）”而非“净利润”“归母净利”）；
                        - 业务列：标注业务/区域/子公司分类（如“业务板块”“子公司名称”而非“分类”）。
                        - 示例：正确列名→“会计期间（季度）、业务板块、营业收入（万元）、毛利率（%）、扣非归母净利润（万元）”；错误列名→“期间、分类、营收、毛利、利润”。

                        ### 2.2 合并单元格/复杂表格处理
                        - 拆分所有合并单元格（如“2023年Q1-Q3”需拆为Q1、Q2、Q3三行，对应数值填写完整，不可留空）；
                        - 跨页表格需在表头标注“（续表）”+原表格标题（如“表3-2023年度营收明细（续表）”），确保表格完整性；
                        - 空白数值需标注原因（如“-”表示无数据，“0”表示数据为0，“未披露”表示未公开），不可留空。

                        ### 2.3 表格标题标准化
                        - 标题格式：「时间范围+数据类型+表格用途」（如“表1-2024年第一季度核心财务指标表”“表5-2023年度各子业务板块毛利率明细”）；
                        - 标题后必须标注单位（如“（单位：万元，百分比保留1位小数）”），若指标单位不同，需在对应列名后单独标注。

                        ## 3. 财务术语强制规范（消除歧义）
                        ### 3.1 核心指标必须用标准全称（附常见别名）
                        - 营收类：“营业收入”（别名：营收、主营业务收入，首次出现需标注）；
                        - 利润类：“归属于母公司股东的净利润”（简称“归母净利润”）、“扣除非经常性损益后的归母净利润”（简称“扣非归母净利润”）；
                        - 比率类：“毛利率（%）”“资产负债率（%）”“同比增长率（%）”“环比增长率（%）”（必须带百分号）；
                        - 严禁使用模糊表述（如“利润”“收益”“增长”需明确为上述标准术语）。

                        ### 3.2 数据口径必须标注
                        - 所有数值需明确“是否合并报表”“是否扣非”“是否经审计”：
                        - 合并报表数据：在表格标题或备注标注“（合并报表数据）”；
                        - 扣非/非扣非：指标列名直接体现（如“扣非归母净利润”“非扣非归母净利润”）；
                        - 审计状态：在表格下方备注“（数据经XX会计师事务所审计，审计意见类型：标准无保留意见）”或“（未经审计）”。

                        ## 4. 附注与数据关联（补充检索上下文）
                        - 财务附注中与表格数据相关的说明（如“营业收入增长因海外订单增加”“资产减值损失来自存货跌价”），需在对应表格下方用「注：」关联（如“注：本表格中2024Q1营业收入同比增长15%，详见附注3.2（页码18）”）；
                        - 特殊数据（如“政府补助计入非经常性损益，金额500万元”）需在对应指标行后用「（）」标注（如“扣非归母净利润：1200万元（已剔除政府补助500万元）”）。

                        ## 5. 图表提取关联规则（适配 extract_charts=True）
                        - 提取的图表（如营收趋势图、毛利率对比图）需在 Markdown 中标注「图X-图表标题（数据来源：表Y-表格标题，页码Z）」；
                        - 图表下方需补充关键数据摘要（如“图2-2023Q1-Q4营收趋势：Q1营收8000万元，Q2 9500万元，Q3 1.1亿元，Q4 1.2亿元，整体呈增长趋势”），避免仅留图片链接无法检索。

                        ## 6. 文本段落处理（辅助表格检索）
                        - 段落中提及的关键财务数据（如“2024年第一季度公司实现归母净利润1.5亿元”），需用「**加粗**」突出，并标注对应表格位置（如“**2024Q1归母净利润1.5亿元**（详见表1-2024Q1核心财务指标，页码5）”）；
                        - 避免大段无结构文本，优先按“指标+数值+维度”拆分短句（如拆分为“1. 2024Q1营业收入：5亿元（同比增长10%）；2. 2024Q1毛利率：35%（环比下降2%）”）。
                                                
                                                """
                        )


    def load_pdf_files(self):
        """
        加载目录下的所有PDF文件
        """
        pdf_files = []
        for file in os.listdir(self.directory):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.directory, file))

        logging.info(f"Found {len(pdf_files)} PDF files.")
        return pdf_files
    

    def load_pdf_table_files(self):
        """
        加载目录下的所有PDF文件
        """
        pdf_files = []
        for file in os.listdir(self.table_directory):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.table_directory, file))

        logging.info(f"Found {len(pdf_files)} PDF files.")
        return pdf_files

    def load_pdf_content(self, pdf_path):
        """
        读取PDF文件内容
        """
        pdf_loader = PyMuPDFLoader(file_path=pdf_path)
        docs = pdf_loader.load()
        logging.info(f"Loading content from {pdf_path}.")

        #将解析结果导出文件，验证解析的效果
        # 尝试多种可能的文本属性提取内容
        content_parts = []
        for doc in docs:
            # 尝试常见的文本属性
            if hasattr(doc, 'page_content') and doc.page_content:
                content_parts.append(doc.page_content)
            elif hasattr(doc, 'content') and doc.content:
                content_parts.append(doc.content)
            elif hasattr(doc, 'text') and doc.text:
                content_parts.append(doc.text)
            elif hasattr(doc, 'page_text') and doc.page_text:
                content_parts.append(doc.page_text)
            else:
                # 如果都没有找到，尝试将对象转换为字符串
                doc_str = str(doc)
                if doc_str:
                    content_parts.append(doc_str)
                else:
                    content_parts.append(f"文档对象 {doc} 没有找到文本内容")
        
        content = "\n\n".join(content_parts)
    
        #print(content)
        with open(file="PyMuPDFLoader.txt", mode="a", encoding="utf8") as f:
            f.write(content)

        return docs
    
    def load_pdf_table_content(self, pdf_path):
        documents = self.finance_parser.load_data(file_path=pdf_path)
        logging.info(f"Loading content from {pdf_path}.")
        
        # 将 LlamaParse 文档转换为标准 LangChain Document 对象
        langchain_docs = []
        for doc in documents:
            # 检查文档对象是否有 text 属性
            if hasattr(doc, 'text'):
                content = doc.text
            elif hasattr(doc, 'get_content'):
                content = doc.get_content()
            else:
                content = str(doc)
                
            # 获取元数据
            metadata = getattr(doc, 'metadata', {})

            # 将解析结果导出文件，验证解析的效果
            with open(file="LlamaParse.txt", mode="a", encoding="utf8") as f:
                f.write(content)

           
            # 创建标准 LangChain Document 对象
            langchain_docs.append(LangChainDocument(
                page_content=content,
                metadata=metadata
            ))
        
        return langchain_docs 
    
    def _convert_langchain_to_llamaindex_docs(self, langchain_docs):
        """
        将LangChain格式的Document列表，转换为llama_index格式的Document列表
        :param langchain_docs: LangChain Document列表
        :return: LlamaIndex Document列表
        """
        llama_docs = []
        for lc_doc in langchain_docs:
            # 1. 生成唯一ID：优先用LangChain文档的id，无则自动生成UUID
            doc_id = lc_doc.id if (hasattr(lc_doc, 'id') and lc_doc.id) else str(uuid.uuid4())
            
            # 2. 转换核心字段：LangChain的page_content → llama_index的text；补充id_属性
            llama_doc = LlamaIndexDocument(
                text=lc_doc.page_content,  # 内容字段对齐
                id_=doc_id,  # 补充llama_index必需的id_属性
                metadata=lc_doc.metadata  # 保留原文档元数据（如文件名、页码）
            )
            llama_docs.append(llama_doc)
        
        logger.debug(f"✅ 完成 {len(langchain_docs)} 个LangChain文档到llama_index格式的转换")
        return llama_docs



    def split_text(self, documents):
        """
        将文本切分成小段
        """
        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunksize,
            chunk_overlap=self.overlap,
            length_function=len,
            add_start_index=True,
        )
        docs = text_splitter.split_documents(documents)

        logging.info("Split text into smaller chunks with RecursiveCharacterTextSplitter.")
        return docs

    def insert_docs(self, docs, insert_function, batch_size=None):
        """
        将文档插入到指定的数据库，并显示进度
        :param docs: 要插入的文档列表
        :param insert_function: 插入函数
        :param batch_size: 批次大小
        """
        if batch_size is None:
            batch_size = self.batch_num

        logging.info(f"Inserting {len(docs)} documents.")
        start_time = time.time()
        total_docs_inserted = 0

        total_batches = (len(docs) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="Inserting batches", unit="batch") as pbar:
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                insert_function(batch)  # 调用传入的插入函数

                total_docs_inserted += len(batch)

                # 计算并显示当前的TPM
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    tpm = (total_docs_inserted / elapsed_time) * 60
                    pbar.set_postfix({"TPM": f"{tpm:.2f}"})

                pbar.update(1)

    def insert_to_vector_db(self, docs):
        """
        将文档插入到 VectorDB
        """
        self.vector_db.add_with_langchain(docs)

    def insert_to_vector_db_with_llamaindex(self, docs):
        """
        使用Llamaindex将文档插入到 VectorDB
        """
        # 调用新增的转换方法，将LangChain文档转为llama_index格式
        llama_docs = self._convert_langchain_to_llamaindex_docs(docs)
        # 直接传入转换后的文档（此时llama_docs包含id_属性）
        self.vector_db.add_with_llamaindex(llama_docs)
        #logger.info(f'✅ 使用Llamaindex将 {len(llama_docs)} 个文档插入到VectorDB完成')

    def insert_to_elasticsearch(self, docs):
        """
        将文档插入到 Elasticsearch
        """
        self.es_client.add_documents(docs)

    def process_pdfs_group(self, pdf_files_group):
        # 读取PDF文件内容
        pdf_contents = []

        for pdf_path in pdf_files_group:
            # 读取PDF文件内容
            documents = self.load_pdf_content(pdf_path)

            # 将documents 逐一添加到pdf_contents
            pdf_contents.extend(documents)

        # 将文本切分成小段
        docs = self.split_text(pdf_contents)

        if self.db_type == 'vector':
            # 将文档插入到 VectorDB
            self.insert_docs(docs, self.insert_to_vector_db_with_llamaindex)
        elif self.db_type == 'es':
            # 将文档插入到 Elasticsearch
            self.insert_docs(docs, self.insert_to_elasticsearch)
        else:
            raise ValueError("db_type must be either 'vector' or 'es'.")
        

    def process_pdfs_table_group(self, pdf_files_group):
        # 读取PDF文件内容
        pdf_contents = []

        for pdf_path in pdf_files_group:
            # 读取PDF文件内容
            documents = self.load_pdf_table_content(pdf_path)

            # 将documents 逐一添加到pdf_contents
            pdf_contents.extend(documents)

        # 将文本切分成小段
        docs = self.split_text(pdf_contents)

        if self.db_type == 'vector':
            # 将文档插入到 VectorDB
            self.insert_docs(docs, self.insert_to_vector_db_with_llamaindex)
            #self.insert_docs(pdf_contents, self.insert_to_vector_db_with_llamaindex)
        elif self.db_type == 'es':
            # 将文档插入到 Elasticsearch
            self.insert_docs(docs, self.insert_to_elasticsearch)
            #self.insert_docs(pdf_contents, self.insert_to_elasticsearch)
        else:
            raise ValueError("db_type must be either 'vector' or 'es'.")

    def process_pdfs(self):
        # 获取目录下所有的PDF文件
        pdf_files = self.load_pdf_files()
        pdf_table_files = self.load_pdf_table_files()

        print(pdf_table_files)

        group_num = self.file_group_num

        # group_num 个PDF文件为一组，分批处理
        for i in range(0, len(pdf_files), group_num):
            pdf_files_group = pdf_files[i:i + group_num]
            self.process_pdfs_group(pdf_files_group)

        # group_num 个PDF文件为一组，分批处理
        # 对于表格类的数据不切分，以便检索时整个检索出来
        group_num = 1
        for i in range(0, len(pdf_table_files), group_num):
            pdf_files_group = pdf_table_files[i:i + group_num]
            self.process_pdfs_table_group(pdf_files_group)

        print("PDFs processed successfully!")
 