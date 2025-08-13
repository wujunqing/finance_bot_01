#!/bin/bash

# 设置变量
SQLITE_DB_PATH="./dataset/dataset/博金杯比赛数据.db"  # SQLite 数据库文件路径
MYSQL_CONTAINER="mysql_container"                      # MySQL Docker 容器名称
MYSQL_USER="smart_admin"                               # MySQL 用户名
MYSQL_PASSWORD="123abc"                                # MySQL 密码
MYSQL_DATABASE="smart_bot"                             # MySQL 数据库名称

# 检查数据库是否存在
if [ ! -f "$SQLITE_DB_PATH" ]; then
    echo "SQLite 数据库文件不存在，请检查路径。"
    exit 1
fi

# 创建 MySQL 数据库
echo "正在创建 MySQL 数据库 ..."
docker exec -i "$MYSQL_CONTAINER" mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "CREATE DATABASE IF NOT EXISTS $MYSQL_DATABASE;"

# 检查当前目录是否有 .sql 文件
SQL_FILES=(*.sql)

if [ -e "${SQL_FILES[0]}" ]; then
    echo "找到现有的 .sql 文件，正在导入到 MySQL ..."
    for SQL_FILE in "${SQL_FILES[@]}"; do
        echo "正在导入 $SQL_FILE 到 MySQL ..."
        if docker exec -i "$MYSQL_CONTAINER" mysql --default-character-set=utf8mb4 -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" < "$SQL_FILE"; then
            echo "文件 $SQL_FILE 数据已成功导入到 MySQL。"
        else
            echo "文件 $SQL_FILE 数据导入失败。请检查错误信息。"
            docker exec "$MYSQL_CONTAINER" mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" -e "SHOW WARNINGS;"
        fi
    done
else
    # 获取数据库中的所有表名
    TABLES=$(sqlite3 "$SQLITE_DB_PATH" ".tables")

    # 遍历每个表并导出
    for TABLE in $TABLES; do
        DUMP_FILE="${TABLE}.sql"  # 每个表的导出文件名

        # 导出 SQLite 表为 SQL 文件
        echo "导出 SQLite 表 $TABLE 到 $DUMP_FILE ..."
        sqlite3 "$SQLITE_DB_PATH" ".dump $TABLE" > "$DUMP_FILE"

        # 处理兼容性：移除 PRAGMA 和其他 SQLite 特有的命令
        echo "正在处理 SQL 文件 $DUMP_FILE ..."
        sed -i '' -e '/^PRAGMA/d' \
                   -e '/^BEGIN TRANSACTION;/d' \
                   -e '/^COMMIT;/d' \
                   -e '/^ROLLBACK;/d' "$DUMP_FILE"

        # 转换 SQL 文件为 MySQL 格式
        sed -i '' 's/INTEGER/INT/g; s/REAL/DOUBLE/g; s/TEXT/VARCHAR(255)/g; s/BLOB/VARBINARY/g; s/DEFAULT CURRENT_TIMESTAMP/DEFAULT NOW()/g' "$DUMP_FILE"

        # 替换双引号为反引号
        echo "正在替换双引号为反引号 ..."
        sed -i '' 's/"\([^"]*\)"/`\1`/g' "$DUMP_FILE"

        # 将 SQL 文件导入 MySQL Docker 容器
        echo "正在导入 $DUMP_FILE 到 MySQL ..."
        if docker exec -i "$MYSQL_CONTAINER" mysql --default-character-set=utf8mb4 -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" < "$DUMP_FILE"; then
            echo "表 $TABLE 数据已成功导入到 MySQL。"
            # 清理临时文件
            rm "$DUMP_FILE"
        else
            echo "表 $TABLE 数据导入失败。请检查错误信息。"
            docker exec "$MYSQL_CONTAINER" mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" -e "SHOW WARNINGS;"
        fi
    done
fi

echo "所有表的导入完成。"
