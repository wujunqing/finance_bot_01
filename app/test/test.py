# # from sqlalchemy import create_engine
# from sqlalchemy.exc import OperationalError

# try:
#     engine = create_engine('sqlite:///E:\ai\jrai\smart-finance-bot\dataset\dataset\博金杯比赛数据.db')
#     with engine.connect() as conn:
#         print("数据库连接成功！")
# except OperationalError as e:
#     print(f"数据库连接失败：{str(e)}")
#     # 进一步打印路径、权限信息辅助排查
#     import os
#     print(f"文件路径存在？{os.path.exists('E:\ai\jrai\smart-finance-bot\dataset\dataset\博金杯比赛数据.db')}")
#     print(f"文件权限：{os.access('E:\ai\jrai\smart-finance-bot\dataset\dataset\博金杯比赛数据.db', os.R_OK | os.W_OK)}")


import os

# 方法1：用原始字符串（r前缀）定义路径，再传入f-string
db_path = r'E:\ai\jrai\smart-finance-bot\dataset\dataset\博金杯比赛数据.db'
print(f"文件路径存在？{os.path.exists(db_path)}")

# 方法2：将反斜杠替换为正斜杠（Windows支持正斜杠路径）
db_path = 'E:/ai/jrai/smart-finance-bot/dataset/dataset/博金杯比赛数据.db'
print(f"文件路径存在？{os.path.exists(db_path)}")