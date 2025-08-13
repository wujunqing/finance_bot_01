import sys
import argparse
import subprocess
from rag.pdf_processor import PDFProcessor
from test.question_answer import TestQuestion
import settings
import sqlite3
import mysql.connector
from mysql.connector import Error
from urllib.parse import urlparse
import os
from rag.vector_db import ChromaDB
from rag.vector_db import MilvusDB

def import_pdf_to_Chroma(directory):

    # 创建 ChromaDB 实例
    chroma_db = ChromaDB(chroma_server_type=settings.CHROMA_SERVER_TYPE_IMPORT, 
                         persist_path=settings.CHROMA_PERSIST_DB_PATH, 
                         embed=settings.EMBED)


    # 创建 PDFProcessor 实例
    pdf_processor = PDFProcessor(directory=directory,
                                 vector_db=chroma_db)

    # 处理 PDF 文件
    pdf_processor.process_pdfs()

def import_pdf_to_Milvus(directory):

    # 创建 MilvusDB 实例
    milvus_db = MilvusDB(embed=settings.EMBED)

    # 创建 PDFProcessor 实例
    pdf_processor = PDFProcessor(directory=directory,
                                 vector_db=milvus_db)

    # 处理 PDF 文件
    pdf_processor.process_pdfs()


def start_chroma(path, port, host):
    # 启动 chroma 的命令
    command = f"chroma run --path {path} --port {port} --host {host}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动 Chroma 时出错: {e}")

def create_index_if_not_exists(cursor, index_name, table_name, column_name):
    # 检查索引是否存在
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.statistics
        WHERE table_schema = DATABASE() 
        AND table_name = '{table_name}' 
        AND index_name = '{index_name}';
    """)
    index_exists = cursor.fetchone()[0]

    if not index_exists:
        try:
            print(f"Creating index: {index_name} on column: {column_name}")
            # 注意：这里不再使用 IF NOT EXISTS
            cursor.execute(f"CREATE INDEX `{index_name}` ON `{table_name}` (`{column_name}`);")
        except Error as e:
            print(f"Error creating index for {column_name} in table {table_name}: {e}")


def add_indexes_to_all_tables(db_uri):
    """在所有表的每个列上添加索引，支持 SQLite 和 MySQL"""
    # 判断db_config是SQLite还是MySQL
    db_type = db_uri.split(':')[0].lower()
    db_config = db_uri.split('://')[1]

    if db_type == 'sqlite':
        conn = sqlite3.connect(db_config)
        cursor = conn.cursor()

        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            print(f"Processing table: {table_name}")

            # 获取表的所有字段
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            for column in columns:
                column_name = column[1]
                index_name = f"idx_{table_name}_{column_name}".replace(' ', '_')

                try:
                    print(f"Creating index: {index_name} on column: {column_name}")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS \"{index_name}\" ON \"{table_name}\" (\"{column_name}\");")
                except sqlite3.OperationalError as e:
                    print(f"Error creating index for {column_name} in table {table_name}: {e}")

        conn.commit()
        conn.close()
        print("All indexes created successfully in SQLite.")

    elif db_type == 'mysql':
        parsed_url = urlparse(db_uri)
        config = {
            'user': parsed_url.username,
            'password': parsed_url.password,
            'host': parsed_url.hostname,
            'database': parsed_url.path[1:],  # 去掉开头的 /
            'port': parsed_url.port or 3306,  # 默认端口为 3306
            'charset': 'utf8mb4'  # 使用 utf8mb4 支持更广泛的 Unicode 字符
        }

        try:
            conn = mysql.connector.connect(**config)
            cursor = conn.cursor()

            # 获取所有表名
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                print(f"Processing table: {table_name}")

                # 获取表的所有字段
                cursor.execute(f"SHOW COLUMNS FROM `{table_name}`;")
                columns = cursor.fetchall()

                for column in columns:
                    column_name = column[0]
                    index_name = f"idx_{table_name}_{column_name}".replace(' ', '_')

                    create_index_if_not_exists(cursor, index_name, table_name, column_name)

            conn.commit()
            print("All indexes created successfully in MySQL.")

        except Error as e:
            print(f"Error: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

def sanitize_name(name):
    """替换不安全字符为安全字符"""
    replacements = {
        '(': '_',
        ')': '_',
        ' ': '_',
        '-': '_',
    }
    
    for unsafe_char, safe_char in replacements.items():
        name = name.replace(unsafe_char, safe_char)
    
    return name

def rename_tables_and_columns(db_uri):
    """重命名表和列，支持 SQLite 和 MySQL"""
    # 判断db_config是SQLite还是MySQL
    db_type = db_uri.split(':')[0].lower()
    db_config = db_uri.split('://')[1]

    if db_type == 'sqlite':
        conn = sqlite3.connect(db_config)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            sanitized_table_name = sanitize_name(table_name)

            if sanitized_table_name != table_name:
                print(f"Renaming table: {table_name} to {sanitized_table_name}")
                cursor.execute(f"ALTER TABLE \"{table_name}\" RENAME TO \"{sanitized_table_name}\";")
                table_name = sanitized_table_name

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            for column in columns:
                column_name = column[1]
                sanitized_column_name = sanitize_name(column_name)

                if sanitized_column_name != column_name:
                    print(f"Renaming column: {column_name} to {sanitized_column_name} in table {table_name}")
                    cursor.execute(f"ALTER TABLE \"{table_name}\" RENAME COLUMN \"{column_name}\" TO \"{sanitized_column_name}\";")

        conn.commit()
        conn.close()
        print("All names sanitized successfully in SQLite.")

    elif db_type == 'mysql':
        parsed_url = urlparse(db_uri)
        config = {
            'user': parsed_url.username,
            'password': parsed_url.password,
            'host': parsed_url.hostname,
            'database': parsed_url.path[1:],  # 去掉开头的 /
            'port': parsed_url.port or 3306,  # 默认端口为 3306
            'charset': 'utf8mb4'  # 使用 utf8mb4 支持更广泛的 Unicode 字符
        }


        try:
            conn = mysql.connector.connect(**config)
            cursor = conn.cursor()

            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                sanitized_table_name = sanitize_name(table_name)

                if sanitized_table_name != table_name:
                    print(f"Renaming table: {table_name} to {sanitized_table_name}")
                    cursor.execute(f"RENAME TABLE `{table_name}` TO `{sanitized_table_name}`;")
                    table_name = sanitized_table_name

                cursor.execute(f"SHOW COLUMNS FROM `{table_name}`;")
                columns = cursor.fetchall()

                for column in columns:
                    column_name = column[0]
                    sanitized_column_name = sanitize_name(column_name)

                    if sanitized_column_name != column_name:
                        print(f"Renaming column: {column_name} to {sanitized_column_name} in table {table_name}")
                        cursor.execute(f"ALTER TABLE `{table_name}` CHANGE `{column_name}` `{sanitized_column_name}` {column[1]};")

            conn.commit()
            print("All names sanitized successfully in MySQL.")

        except Error as e:
            print(f"Error: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()


def run_test_answer_question(start, end):
    current_path = os.getcwd()
    input_file_path = os.path.join(current_path, "dataset/question.json")

    test_question = TestQuestion(input_file_path, test_case_start=start, test_case_end=end)
    test_question.run_cases()


def main():
    parser = argparse.ArgumentParser(description="Entroy script for executing jobs.")
    parser.add_argument('--job', type=str, required=True, help='Job to execute: importpdf, startchroma, testapik, addindexes, renametables, test_question')

    # 添加 importpdf 任务的参数
    parser.add_argument('--dir', type=str, help='Directory for PDF files')
    parser.add_argument('--db_type', type=str, choices=['chroma', 'milvus'], help='Type of vector database to import PDFs into')
    
    # 添加 startchroma 任务的参数
    parser.add_argument('--path', type=str, default='chroma_db', help='Path for Chroma DB')
    parser.add_argument('--port', type=int, default=8000, help='Port to run Chroma on')
    parser.add_argument('--host', type=str, default='localhost', help='Host address for Chroma server')

    # 添加 test_question 任务的参数
    parser.add_argument('--start', type=int, default=0, help='Test case start id')
    parser.add_argument('--end', type=int, default=5, help='Test case end id')

    # 添加数据库类型参数
    parser.add_argument('--db_uri', type=str, help='Database URI for SQLite or MySQL')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.job == 'importpdf':
        if not args.dir:
            print("请提供 --dir 参数以指定 PDF 文件目录")
        else:
            if args.db_type == 'chroma':
                import_pdf_to_Chroma(args.dir)
            elif args.db_type == 'milvus':
                import_pdf_to_Milvus(args.dir)
            else:
                print("未知的数据库类型。请使用: chroma 或 milvus")
    elif args.job == 'startchroma':
        start_chroma(args.path, args.port, args.host)
    elif args.job == 'addindexes':
        add_indexes_to_all_tables(args.db_uri)  
    elif args.job == 'renametables':
        rename_tables_and_columns(args.db_uri)  
    elif args.job == 'test_question':
        run_test_answer_question(args.start, args.end) 
    else:
        print("未知的任务类型。请使用: importpdf, startchroma, 或 testapik")


if __name__ == "__main__":
    main()
