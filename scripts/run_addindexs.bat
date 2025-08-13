@echo off
REM 获取当前脚本所在目录
SET SCRIPT_DIR=%~dp0

REM 设置 Python 脚本的路径
SET PYTHON_SCRIPT=%SCRIPT_DIR%..\app\entrypoint.py

REM 数据库类型和 URI
SET DB_TYPE=sqlite  REM 或者 mysql
SET DB_URI=sqlite://.\dataset\dataset\博金杯比赛数据.db  REM 对于 SQLite，或者 mysql://user:password@localhost/dbname 对于 MySQL

REM 调用 entrypoint.py，传入参数
echo Running entrypoint.py to renametables to DB...
python "%PYTHON_SCRIPT%" --job renametables --db_uri %DB_URI%

echo Running entrypoint.py to addindexes to DB...
python "%PYTHON_SCRIPT%" --job addindexes --db_uri %DB_URI%
