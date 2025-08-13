from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langserve import add_routes
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from finance_bot import FinanceBot
from finance_bot_ex import FinanceBotEx

# Define request/response models for better Swagger docs
class QueryRequest(BaseModel):
    input: str
    
    class Config:
        schema_extra = {
            "example": {
                "input": "请问某公司的财务状况如何？"
            }
        }

class QueryResponse(BaseModel):
    output: str
    
    class Config:
        schema_extra = {
            "example": {
                "output": "根据财务数据分析..."
            }
        }

# Enhanced FastAPI app with better documentation
app = FastAPI(
    title="Smart Finance Bot API",
    version="1.0.0",
    description="智能金融机器人API，提供财务数据查询和分析服务",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enhanced API endpoints with better documentation
@app.post(
    "/query", 
    response_model=QueryResponse,
    summary="基础财务查询",
    description="使用基础财务机器人处理用户查询",
    tags=["Finance Query"]
)
async def query(request: QueryRequest):
    try:
        result = finance_bot.handle_query(request.input)
        return QueryResponse(output=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/queryex", 
    response_model=QueryResponse,
    summary="扩展财务查询",
    description="使用扩展财务机器人处理复杂查询",
    tags=["Finance Query"]
)
async def query_extended(request: QueryRequest):
    try:
        result = finance_bot_ex.handle_query(request.input)
        return QueryResponse(output=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


finance_bot = FinanceBot()
finance_bot_ex = FinanceBotEx()


# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen API",
    version="0.1",
    description="Qwen API",
)
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许的HTTP方法
    allow_headers=["*"],  # 允许的请求头
)


# 创建API路由
@app.post("/query", response_model=dict)
async def query(query: dict):  # 使用字典类型代替Query模型
    try:
        # 从字典中获取input
        input_data = query.get("input")
        result = finance_bot.handle_query(input_data)

        # 返回字典格式的响应
        return {"output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queryex", response_model=dict)
async def query(query: dict):  # 使用字典类型代替Query模型
    try:
        # 从字典中获取input
        input_data = query.get("input")
        result = finance_bot_ex.handle_query(input_data)

        # 返回字典格式的响应
        return {"output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 运行Uvicorn服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
