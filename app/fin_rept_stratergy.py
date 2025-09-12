import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime,timedelta

# 1. 定义待分析的股票池 
stock_pool = [
    '600519',  # 贵州茅台
    '603043',  # 广州酒家
    '600835',  # 上海机电
    '600332',  # 白云山
    '600518'   # 康美药业
]

def check_single_stock_finance(stock_code):
    """
    检查单只股票是否符合近5年财务筛选条件
    :param stock_code: 股票代码，带后缀，例如 '000001.SZ'
    :return: (是否符合财务条件, 详细信息字符串)
    """
    try:
        # 获取年度财务报表-盈利能力指标
        #print(f"开始获取财报盈利能力指标")
        df_fina = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        #print(f"开始打印财报盈利能力指标")

        
        # 取最新的5条年度数据（假设最后5行是最近5年）
        df_fina = df_fina.head(5)
        #print(df_fina)
        #print(f"结束打印财报盈利能力指标")
      
        # 检查数据是否足够5年
        if len(df_fina) < 5:
            return False, f"数据不足5年, 仅有{len(df_fina)}年数据"
        
        roe_series = df_fina['净资产收益率']
        gross_profit_series = df_fina['销售毛利率']   
        net_profit_series = df_fina['销售净利率']        
        # 检查每一年的数据是否都满足条件
        if roe_series.any():
            roe_series = pd.to_numeric(roe_series.str.replace('%', ''), errors='coerce')
            roe_check = (roe_series > 20).all()
        else:
            roe_check = False
            print(f"ROE缺失数据")
        gross_profit_series = pd.to_numeric(gross_profit_series.str.replace('%', ''), errors='coerce')
        gross_profit_check = (gross_profit_series > 40).all()
        net_profit_series = pd.to_numeric(net_profit_series.str.replace('%', ''), errors='coerce')
        net_profit_check = (net_profit_series > 5).all()   
        # 生成详细的检查结果信息
        detail_msg = f"""
        ROE(近5年): {roe_series.tolist()} >20? {roe_check}
        毛利率(近5年): {gross_profit_series.tolist()} >40? {gross_profit_check}
        净利率(近5年): {net_profit_series.tolist()} >5? {net_profit_check}
        """
        
        # 如果所有条件都满足，则返回True
        if roe_check and gross_profit_check and net_profit_check:
            return True, detail_msg
        else:
            return False, detail_msg
            
    except Exception as e:
        error_msg = f"获取或处理数据时出错: {e}"
        return False, error_msg

def get_pe_signal(stock_code):
    """
    获取单只股票的实时市盈率并生成交易信号
    :param stock_code: 股票代码，带后缀
    :return: (市盈率, 交易信号)
    """
    try:
        #print(f"开始PE信号")
        # 获取实时数据
        '''
        stock_zh_a_spot_df = ak.stock_zh_a_spot()
        # 去掉后缀比较代码
        #code_without_suffix = stock_code[:-3] 
        print(stock_code)
        stock_data = stock_zh_a_spot_df[stock_zh_a_spot_df['代码'] == stock_code]#code_without_suffix]
        '''
        
        stock_data = ak.stock_zh_valuation_baidu(symbol=stock_code, indicator="市盈率(TTM)", period="近一年")
        #print(stock_data)
        #current_date = datetime.now().date()
        #print("当前日期:", current_date)
        #previous_day = current_date - timedelta(days=1)
        #print("前一天日期:", previous_day)
        if not stock_data.empty:
            # !!! 请核对字段名 !!!
            pe = stock_data.iloc[-1]['value']
         #   print(f"市盈率{pe}")
            
            if pd.isna(pe):
                return pe, '无数据'
            elif pe < 22:
                return pe, '买入'
            elif pe > 22:
                return pe, '卖出'
            else:
                return pe, '持有'
        else:
            return None, '未找到股票'
    except Exception as e:
        return None, f'Error: {e}'
        
    # 2. 主程序
if __name__ == '__main__':
    print("开始分析指定股票池的巴菲特量化策略...")
    print("="*50)
    
    results = []
    
    for stock in stock_pool:
        print(f"\n正在分析 {stock} ...")
        
        # 检查财务条件
        finance_qualified, finance_detail = check_single_stock_finance(stock)
        # 获取PE和交易信号
        if finance_qualified:
            pe_value, signal = get_pe_signal(stock)
        else:
            pe_value="财务未达标不买入"
            signal="不买入"
        # 收集结果
        result = {
            '股票代码': stock,
            '财务达标': '是' if finance_qualified else '否',
            '当前市盈率(PE)': pe_value,
            '交易信号': signal,
            '财务详情': finance_detail
        }
        results.append(result)
        
        # 打印简要结果
        print(f"  财务达标: {'是' if finance_qualified else '否'}")
        print(f"  当前PE: {pe_value}")
        print(f"  交易信号: {signal}")
    
    # 3. 打印最终汇总表格
    print("\n" + "="*50)
    print("分析结果汇总:")
    print("="*50)
    
    # 创建一个简洁的DataFrame用于显示主要结果
    summary_df = pd.DataFrame([
        {
            '股票代码': r['股票代码'], 
            '财务达标': r['财务达标'], 
            '市盈率(PE)': r['当前市盈率(PE)'], 
            '交易信号': r['交易信号']
        } for r in results
    ])
    print(summary_df.to_string(index=False))
    
    # 4. 打印详细的财务检查信息
    print("\n" + "="*50)
    print("详细财务分析:")
    print("="*50)
    for r in results:
        print(f"\n{r['股票代码']} 的财务详情: {r['财务详情']}")