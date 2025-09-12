import akshare as ak
import pandas as pd
import datetime as dt
import time

def get_market_type(code):
    """根据股票代码判断市场类型"""
    if code.endswith('.SH') or code.endswith('.SS'):
        return '上海'
    elif code.endswith('.SZ'):
        return '深圳'
    else:
        # 对于没有后缀的代码，根据代码开头判断
        if code.startswith('6'):
            return '上海'
        elif code.startswith(('0', '3')):
            return '深圳'
        else:
            return '未知'

def get_a_stock_data_with_retry(max_retries=3, delay=5):
    """带重试机制的A股数据获取"""
    for attempt in range(max_retries):
        try:
            print(f"正在尝试获取A股数据... (第{attempt + 1}次尝试)")
            df = ak.stock_info_a_code_name()
            print(f"成功获取到 {len(df)} 只股票数据")
            return df
        except Exception as e:
            print(f"第{attempt + 1}次尝试失败: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("所有尝试都失败，使用备用数据")
                return get_backup_stock_data()

def get_backup_stock_data():
    """备用股票数据（当网络请求失败时使用）"""
    backup_data = {
        'code': [
            '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ', '300059.SZ',
            '600000.SS', '600036.SS', '600519.SS', '600887.SS', '601318.SS',
            '601398.SS', '601857.SS', '601988.SS', '603259.SS', '688111.SS'
        ],
        'name': [
            '平安银行', '万科A', '五粮液', '海康威视', '东方财富',
            '浦发银行', '招商银行', '贵州茅台', '伊利股份', '中国平安',
            '工商银行', '中国石油', '中国银行', '药明康德', '金山办公'
        ]
    }
    print("使用备用数据集（包含15只主要A股）")
    return pd.DataFrame(backup_data)

# 主程序
try:
    # 尝试获取完整的A股数据
    df = get_a_stock_data_with_retry()
    
    # 增加市场类型列
    df['市场类型'] = df['code'].apply(get_market_type)
    
    # 重新排列列的顺序
    df = df[['code', 'name', '市场类型']]
    
    # 保存到CSV文件
    filename = f'A股代码名称_{dt.date.today():%Y%m%d}.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存到: {filename}")
    
    # 打印统计信息
    print(f"\n总共获取到 {len(df)} 只股票")
    print("\n市场分布：")
    market_counts = df['市场类型'].value_counts()
    for market, count in market_counts.items():
        print(f"  {market}: {count} 只")
    
    # 显示前几行数据作为示例
    print("\n前10行数据示例：")
    print(df.head(10).to_string(index=False))
    
except Exception as e:
    print(f"程序执行出错: {str(e)}")
    print("请检查网络连接或稍后重试")