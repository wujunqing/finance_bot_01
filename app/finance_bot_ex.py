import sys
import os
import json
from typing import Union, TypedDict, Annotated, List
import operator
from datetime import datetime

from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 添加输入编码处理
# 更安全的输入编码处理
# if sys.platform.startswith('win'):
#     import codecs
#     import io
#     
#     # 检查是否在交互式环境中
#     if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
#         try:
#             # 只在交互式终端中重新配置编码
#             sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
#             sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
#         except (AttributeError, io.UnsupportedOperation, ValueError):
#             # 如果无法重新配置，则跳过
#             pass
#     else:
#         # 在非交互式环境（如服务器）中，设置环境变量
#         import os
#         os.environ['PYTHONIOENCODING'] = 'utf-8'

from stock_analyzer import StockAnalyzer
import settings
from utils.logger_config import LoggerManager
from rag.vector_db import ChromaDB
from rag.vector_db import MilvusDB
from rag.rag import RagManager

logger = LoggerManager().logger


# 定义调用函数
def get_datetime() -> str:
    """
    获取当前时间
    """
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_date

# 定义股票涨幅计算函数
# 股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。
def calculate_stock_change_rate(closing_price: float, previous_closing_price: float) -> float:
    """
    计算股票涨跌幅
    """
    change_rate = ((closing_price - previous_closing_price) / previous_closing_price) * 100
    return change_rate

# 定义股票日收益率计算函数
# 日收益率 = （当日收盘价-昨收盘价）/昨收盘价
def calculate_stock_daily_return(closing_price: float, previous_closing_price: float) -> float:
    """
    计算股票日收益率
    """
    daily_return = (closing_price - previous_closing_price) / previous_closing_price
    return daily_return

# 定义股票年化收益率计算函数
# 年化收益率定义为：（（有记录的一年的最终收盘价-有记录的一年的年初当天开盘价）/有记录的一年的当天开盘价）* 100%。
def calculate_stock_annualized_return(final_closing_price: float, initial_opening_price: float) -> float:
    """
    计算股票年化收益率
    """
    annualized_return = ((final_closing_price - initial_opening_price) / initial_opening_price) * 100
    return annualized_return

# 定义计算股票是否涨停的计算函数
# （收盘价/昨日收盘价-1）>=9.8% 视作涨停
def calculate_stock_limit_up(closing_price: float, previous_closing_price: float) -> bool:
    """
    计算股票是否涨停
    """
    return (closing_price / previous_closing_price - 1) >= 0.098


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


class FinanceBotEx:
    def __init__(self, llm=settings.LLM, chat=settings.CHAT, embed=settings.EMBED, vector_db_type='chroma'):
        # 设置输入编码
        if hasattr(sys.stdin, 'reconfigure'):
            sys.stdin.reconfigure(encoding='utf-8')
        
        self.llm = llm
        self.chat = chat
        self.embed = embed
        self.tools = []

        if vector_db_type.lower() == 'chroma':
            # 使用示例
            db_config = {
                "chroma_server_type": settings.CHROMA_SERVER_TYPE,
                "host": settings.CHROMA_HOST,
                "port": settings.CHROMA_PORT,
                "persist_path": settings.CHROMA_PERSIST_DB_PATH,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
            }

            self.rag = RagManager(vector_db_class=ChromaDB, db_config=db_config, llm=self.llm, embed=self.embed)
        else:
            # 使用示例
            db_config = {
                "milvus_server_type": settings.MILVUS_SERVER_TYPE,
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT,
                "collection_name": settings.MILVUS_COLLECTION_NAME,
            }

            self.rag = RagManager(vector_db_class=MilvusDB, db_config=db_config, llm=self.llm, embed=self.embed)

        self.agent_executor = self.init_agent()
        logger.info(f'初始化程序框架：FinanceEx')

    def init_rag_tools(self):
        # 给大模型 RAG 检索器工具
        retriever = self.rag.retriever_instance.create_retriever()

        retriever_tool = create_retriever_tool(
            retriever=retriever,
            name="rag_search",
            description="按照用户的问题搜索相关的资料，对于招股书类的问题，you must use this tool!",
        )
        return retriever_tool

    def init_sql_tool(self, path):
        # 连接数据库
        logger.info(f"FinanceBotEx连接数据库: {path}")

        db = SQLDatabase.from_uri(path)

        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        sql_tools = toolkit.get_tools()  # 工具

        return sql_tools

    @staticmethod
    def create_sys_prompt():
        system_prompt = """你是一位金融助手，可以帮助用户查询数据库中的信息和预测股价。
            你要尽可能的回答用户提出的问题，为了更好的回答问题，你可以使用工具进行多轮的尝试。

            # 关于用户提出的问题:
            1、如果用户的问题中包含多个问题，请将问题分解为单个问题并逐个回答。
            
            # 关于股价预测工具的使用：
            1、当用户询问股票价格预测时，使用predict_stock_price工具。
            2、该工具可以预测下一交易日的收盘价、最高价和最低价。
            3、股票代码格式：中国股票使用'000001.SZ'或'000001.SS'，美股使用'AAPL'等。
            4、预测结果包含置信度，请向用户说明预测的不确定性。
                                                
            # 关于retriever_tool工具的使用：
            1、你需要结合对检索出来的上下文进行回答问题。
            2、你可以使用检索工具来查找相关的资料，以便回答用户的问题。
            3、检索的词语最好使用命名实体，例如：公司名称、人名、产品名称等。
            
            # 关于sql类工具的使用： 
            ## 工具使用规则                                     
            1、你需要根据用户的问题，创建一个语法正确的SQLite查询来运行，然后查看查询的结果并返回答案。
            2、除非用户指定了他们希望获得的特定数量的示例，否则总是将查询限制为最多5个结果。
            3、您可以按相关列对结果进行排序，以返回数据库中最有趣的示例。
            4、永远不要查询指定表的所有列以避免查询性能问题，你只查询给定问题的相关列即可。
            5、你必须在执行查询之前仔细检查查询。如果执行查询时出现错误，请重新编写查询并重试。
            6、请勿对数据库进行任何DML语句（INSERT，UPDATE，DELETE，DROP等）。
            7、SQL查询的表名和字段名，请务必双引号包裹起来，例如：收盘价(元) 双引号包裹为 "收盘价(元)"。
            
            ## 工具使用过程
            1、首先，你可以从记忆中寻找数据库表的信息，看看可以查询什么；如果记忆中没有，你可以查看数据库中的表；这一步骤很重要，注意不要跳过。
            2、然后，你应该查询最相关表的schema。
            3、之后，请把有哪些表以及相关表的schema记住，以便下次查询时优先从记忆中获取。
            
            ## 工具使用注意事项：
            1、如果查询过程中SQL语句有语法错误，减少查询量,总体查询次数应控制在15次以内。 
            2、请注意SQL语句的查询性能，SQL语句中如果有`SUM`、`COUNT(*)`的情况，务必使用`WITH FilteredIndustry`先筛选出符合条件的数据，然后再进行计算。
            3、对于复杂查询，请在生成 SQL 语句后使用 EXPLAIN 来评估查询计划，避免使用全表扫描或其他低效操作。
            4、如果你在查询数据库时，已经尽力了但是没有找到答案，你可以尝试使用RAG检索工具来查找相关的资料。

                                                
            # 关于你的思考和行动过程，请按照如下格式：
            问题：你必须回答的输入问题
            思考：你应该总是考虑该怎么做
            行动：你应该采取的行动，应该是以下工具之一：{tool_names}
            行动输入：行动的输入
            观察：行动的结果
            ... (这个思考/行动/行动输入/观察可以重复N次)
            最终答案：原始输入问题的最终答案

            # 关于最终答案：
            1、如果你不知道答案，就说你不知道。
            2、请对最终答案总结，给出不超过三句话的简洁回答。
            
            Begin!
                            
            """
        return system_prompt

    def init_agent(self):
        # 初始化 RAG 工具
        retriever_tool = self.init_rag_tools()
    
        # 初始化 SQL 工具
        sql_tools = self.init_sql_tool(settings.SQLDATABASE_URI)
    
        # 创建系统Prompt提示语
        system_prompt = self.create_sys_prompt()
    
        # 创建Agent - 添加股价相关工具
        agent_executor = create_react_agent(
            self.chat,
            tools=[
                get_datetime,
                calculate_stock_change_rate,
                calculate_stock_daily_return,
                calculate_stock_annualized_return, 
                calculate_stock_limit_up,
                predict_stock_price,  # 股价预测工具（包含当前价格）
                get_current_stock_price,  # 新增：独立的当前股价工具
                retriever_tool] + sql_tools,
            checkpointer=MemorySaver()
        )
        return agent_executor

    def handle_query(self, example_query):
        # 流式处理事件
        config = {"configurable": {"thread_id": "thread-1"}}

        try:
            events = self.agent_executor.stream(
                {"messages": [("user", example_query)]},
                config=config,
                stream_mode="values",
            )
            result_list = []
            # 打印流式事件的消息
            for event in events:
                logger.info(event["messages"][-1].pretty_print())

                result_list.append(event["messages"][-1].content)

            final_result = event["messages"][-1].content if result_list else None

            logger.info(f'查询过程：')
            for presult in result_list:
                logger.info(f'【agent】: {presult}')
            logger.info(f"最终结果: {final_result}")
            
            return final_result
        except Exception as e:
            logger.error(f"处理查询时出错: {e}")
            raise e

    def create_agent(self):
        """
        使用langgraph创建Agent，该函数还未启用
        """
        from langchain.agents.format_scratchpad import format_to_openai_functions
        from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
        from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
        from langchain.tools.render import format_tool_to_openai_function
        from langchain_core.output_parsers import StrOutputParser

        sys_msg = SystemMessagePromptTemplate.from_template(template=self.create_sys_prompt())
        user_msg = HumanMessagePromptTemplate.from_template(template="""
            问题：{input}
        """)
        messages = [sys_msg, user_msg]
        prompt = ChatPromptTemplate.from_messages(messages=messages)

        retriever_tool = self.init_rag_tools()
        sql_tools = self.init_sql_tool(settings.SQLDATABASE_URI)

        tools = [retriever_tool] + sql_tools

        llm_with_tools = self.chat.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )

        # agent = {
        #     "input": lambda x: x["input"],
        #     "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
        #     # "chat_history": lambda x: x["chat_history"]
        # } | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

        # 使用代理
        from langchain.agents import AgentExecutor
        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        # agent_executor.invoke({"input": "关于color有多少个字母?"})

        # 创建Agent
        agent_executor = create_react_agent(
            self.chat,
            tools=[get_datetime, retriever_tool] + sql_tools,
            state_modifier=prompt
        )

        # return agent_executor
        # return prompt | agent_executor | StrOutputParser()

        # return prompt | self.chat | agent_executor | StrOutputParser()


# 定义股价预测工具函数
def convert_to_json_serializable(obj):
    """将numpy类型转换为JSON可序列化的Python原生类型"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def predict_stock_price(symbol: str) -> str:
    """预测股票价格
    
    Args:
        symbol: 股票代码，例如 '000001.SZ' 或 'AAPL'
        
    Returns:
        包含预测结果和当前价格的JSON字符串
    """
    try:
        # 初始化股票分析器
        analyzer = StockAnalyzer(symbol)
        
        # 获取当前股价信息
        current_price_info = analyzer.get_current_price()
        
        # 获取历史数据
        df = analyzer.get_historical_data(period="1y")
        
        if df is None or len(df) == 0:
            return json.dumps({
                "error": "无法获取股票历史数据",
                "symbol": symbol,
                "current_price": convert_to_json_serializable(current_price_info)
            }, ensure_ascii=False)
        
        # 调用预测方法，传递df参数
        result = analyzer.predict_next_day(df)
        
        # 处理model_performance中的numpy类型
        model_performance = convert_to_json_serializable(result.get('model_performance', {}))
        
        # 处理当前价格信息中的numpy类型
        current_price_data = convert_to_json_serializable(current_price_info)
        
        return json.dumps({
            "symbol": symbol,
            "current_price": current_price_data,
            "prediction": {
                "next_day_close": float(result.get('next_day_close', result.get('close', 0))),
                "next_day_high": float(result.get('next_day_high', result.get('high', 0))),
                "next_day_low": float(result.get('next_day_low', result.get('low', 0))),
                "date": result.get('date', '').strftime('%Y-%m-%d') if result.get('date') else '',
                "confidence": float(result.get('confidence', 0)),
                "model_performance": model_performance
            }
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"股价预测失败: {str(e)}",
            "symbol": symbol
        }, ensure_ascii=False)


def get_current_stock_price(symbol: str) -> str:
    """获取股票当前价格信息
    
    Args:
        symbol: 股票代码，例如 '000001.SZ' 或 'AAPL'
        
    Returns:
        包含当前股价信息的JSON字符串
    """
    try:
        # 初始化股票分析器
        analyzer = StockAnalyzer(symbol)
        
        # 获取当前股价信息
        result = analyzer.get_current_price()
        
        # 处理可能的numpy类型
        result = convert_to_json_serializable(result)
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"获取当前股价失败: {str(e)}",
            "symbol": symbol
        }, ensure_ascii=False)
