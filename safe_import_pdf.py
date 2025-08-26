import os
import sys
sys.path.append('app')

from rag.pdf_processor import PDFProcessor
from rag.vector_db import ChromaDB
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_model():
    """获取嵌入模型，避免循环导入"""
    try:
        # 尝试使用新版本的 HuggingFace 嵌入模型
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-roberta-large-v1")
    except ImportError:
        try:
            # 备用方案：使用旧版本的 HuggingFace 嵌入模型
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-roberta-large-v1")
        except Exception as e:
            logger.error(f"无法加载任何嵌入模型: {e}")
            return None

def safe_import_pdfs(pdf_dir="dataset/pdf", batch_size=5):
    """安全地导入PDF文件，遇到错误时跳过"""
    
    # 初始化向量数据库和PDF处理器
    embed = get_embedding_model()
    if embed is None:
        logger.error("无法初始化嵌入模型，退出")
        return
        
    vector_db = ChromaDB(embed=embed)
    
    # 修复：正确初始化PDFProcessor
    pdf_processor = PDFProcessor(
                                        directory="dataset/pdf",
                                        db_type='vector',
                                        vector_db=vector_db,
                                        chunksize=1024,  # 修改切分大小为1024字符
                                        overlap=200      # 修改重叠大小为200字符
                                )
    
    # 获取所有PDF文件
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF目录不存在: {pdf_dir}")
        return
    
    # 修复：支持大小写PDF扩展名
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    success_count = 0
    error_count = 0
    error_files = []
    
    # 分批处理文件
    for i in range(0, len(pdf_files), batch_size):
        batch_files = pdf_files[i:i+batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(pdf_files)-1)//batch_size + 1}")
        
        for filename in batch_files:
            file_path = os.path.join(pdf_dir, filename)
            try:
                logger.info(f"处理文件: {filename}")
                
                # 使用PDF处理器处理单个文件
                # 将第65行改为：
                documents = pdf_processor.load_pdf_content(file_path)
                
                if documents:
                    # 修复：使用正确的方法名
                    chunks = pdf_processor.split_text(documents)
                    if chunks:
                        # 修复：使用正确的插入方法
                        pdf_processor.insert_docs(chunks, pdf_processor.insert_to_vector_db)
                        success_count += 1
                        logger.info(f"成功处理: {filename}")
                    else:
                        logger.warning(f"文件切分失败: {filename}")
                        error_count += 1
                        error_files.append(filename)
                else:
                    logger.warning(f"文件加载失败: {filename}")
                    error_count += 1
                    error_files.append(filename)
                    
            except Exception as e:
                logger.error(f"处理文件失败: {filename}, 错误: {e}")
                error_count += 1
                error_files.append(filename)
                continue
    
    # 输出处理结果
    logger.info(f"\n处理完成!")
    logger.info(f"成功处理: {success_count} 个文件")
    logger.info(f"失败文件: {error_count} 个")
    
    if error_files:
        logger.info(f"失败的文件列表: {error_files}")

if __name__ == "__main__":
    safe_import_pdfs()