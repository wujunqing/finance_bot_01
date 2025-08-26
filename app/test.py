#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融助手测试脚本
用于测试 FinanceBotEx 的基本功能
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance_bot_ex import FinanceBotEx
from utils.logger_config import LoggerManager

# 初始化日志
logger = LoggerManager().logger

def main():
    """
    主测试函数
    """
    try:
        # 1. 初始化金融助手（使用Chroma向量数据库）
        logger.info("正在初始化金融助手...")
        finance_bot = FinanceBotEx(vector_db_type='chroma')
        logger.info("金融助手初始化完成")

        # 2. 发起查询（如"计算伟思医疗（688580）今日涨跌幅，已知收盘价57.55元，昨日收盘价54.72元"）
        query = "计算伟思医疗（688580）预测下一交易日的股票涨幅。"
        logger.info(f"发起查询: {query}")
        
        result = finance_bot.handle_query(query)

        # 3. 输出结果
        print("\n=== 查询结果 ===")
        print(result)  # 最终会返回计算结果（如"伟思医疗今日涨跌幅为5.17%"）
        print("================\n")
        
        logger.info(f"查询完成，结果: {result}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        print(f"错误: {str(e)}")
        return False
    
    return True

def test_multiple_queries():
    """
    测试多个查询示例
    """
    try:
        finance_bot = FinanceBotEx(vector_db_type='chroma')
        
        # 测试查询列表
        test_queries = [
            "计算伟思医疗（688580）今日涨跌幅，已知今日收盘价57.55元，昨日收盘价54.72元",
            "什么是股票涨跌幅？",
            "如何计算股票的日收益率？",
            "获取当前时间"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n=== 测试查询 {i} ===")
            print(f"问题: {query}")
            
            try:
                result = finance_bot.handle_query(query)
                print(f"回答: {result}")
            except Exception as e:
                print(f"查询失败: {str(e)}")
                logger.error(f"查询失败 - {query}: {str(e)}")
            
            print("=" * 50)
            
    except Exception as e:
        logger.error(f"多查询测试失败: {str(e)}")
        print(f"多查询测试失败: {str(e)}")

if __name__ == "__main__":
    print("开始金融助手测试...")
    
    # 运行基本测试
    success = main()
    
    if success:
        print("\n基本测试完成，开始多查询测试...")
        test_multiple_queries()
    
    print("\n测试结束")