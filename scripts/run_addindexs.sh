#!/bin/bash

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 设置 Python 脚本的路径
PYTHON_SCRIPT="$SCRIPT_DIR/../app/entrypoint.py"

# 数据库URI
# DB_URI="sqlite://./dataset/dataset/博金杯比赛数据.db"
DB_URI="mysql://smart_admin:123abc@localhost/smart_bot"

# 调用 entrypoint.py，传入参数
echo "Running entrypoint.py to renametables to DB..."
python "$PYTHON_SCRIPT" --job renametables --db_uri "$DB_URI"

echo "Running entrypoint.py to addindexes to DB..."
python "$PYTHON_SCRIPT" --job addindexes --db_uri "$DB_URI"
