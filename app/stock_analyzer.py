import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from datetime import datetime

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # 对收盘价预测给予更高权重
        weights = torch.tensor([2.0, 1.0, 1.0])  # close, high, low
        weighted_loss = weights * (pred - target) ** 2
        return weighted_loss.mean()

class StockDataset(Dataset):
    def __init__(self, X, y_close, y_high, y_low, seq_length=10):
        self.X = torch.FloatTensor(X)
        self.y_close = torch.FloatTensor(y_close)  # 收盘价目标
        self.y_high = torch.FloatTensor(y_high)    # 最高价目标
        self.y_low = torch.FloatTensor(y_low)      # 最低价目标
        self.seq_length = seq_length

    def __len__(self):
        # 返回可用的序列数量
        return max(0, len(self.X) - self.seq_length)

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]
        y = torch.stack([self.y_close[idx+self.seq_length], 
                        self.y_high[idx+self.seq_length], 
                        self.y_low[idx+self.seq_length]])  # 堆叠三个目标值
        return X_seq, y

class HybridModel(nn.Module):
    def __init__(self, input_size):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=3, dropout=0.2, batch_first=True)
        self.attention = nn.MultiheadAttention(128, 8, dropout=0.1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # 预测close, high, low

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
        
    def forward(self, x):
        # LSTM层处理
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        # MultiheadAttention需要(seq_len, batch, embed_dim)格式
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)  # 转回(batch, seq_len, embed_dim)

        # 残差连接和层归一化
        attn_out = attn_out + lstm_out
        attn_out = F.layer_norm(attn_out, [attn_out.size(-1)])
        
        # 使用平均池化而非仅最后时间步
        pooled_output = torch.mean(attn_out, dim=1)
        
        # 全连接层
        out = F.leaky_relu(self.fc1(pooled_output))
        out = self.dropout1(out)
        out = F.leaky_relu(self.fc2(out))
        out = self.dropout2(out)
        out = F.leaky_relu(self.fc3(out))
        out = self.fc4(out)
        
        # 确保价格逻辑关系：high >= close >= low
        high = torch.max(out[:, 1:2], out[:, 0:1])
        low = torch.min(out[:, 2:3], out[:, 0:1])
        out = torch.cat([out[:, 0:1], high, low], dim=1)
        return out
        
    def calculate_feature_importance(self, X, y):
        """计算特征重要性"""
        if len(X) == 0 or len(y) == 0:
            return []
            
        # 确保X和y长度一致
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        if len(X) == 0:
            return []
            
        importances = []
        try:
            base_pred = self(X).detach().numpy()
            
            for i in range(X.shape[2]):  # 遍历每个特征
                X_temp = X.clone()
                X_temp[:,:,i] = 0  # 将特征i置零
                new_pred = self(X_temp).detach().numpy()
                
                # 使用收盘价计算特征重要性
                min_pred_len = min(len(base_pred), len(new_pred))
                importance = np.mean(np.abs(base_pred[:min_pred_len, 0] - new_pred[:min_pred_len, 0]))
                importances.append(importance)
                
            # 归一化特征重要性
            total = sum(importances)
            if total > 0:
                self.feature_importances_ = [imp/total for imp in importances]
            else:
                self.feature_importances_ = [0] * len(importances)
                
        except Exception as e:
            print(f"计算特征重要性时出错: {e}")
            self.feature_importances_ = [0] * X.shape[2]
            
        return self.feature_importances_

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        
    def get_stock_info(self):
        """获取股票基本信息"""
        try:
            info = self.stock.info
            # 处理中国股票代码
            if self.symbol.endswith('.SS') or self.symbol.endswith('.SZ'):
                chinese_name = info.get('longName', '')
                if not chinese_name:
                    chinese_name = info.get('shortName', self.symbol)
            else:
                chinese_name = info.get('longName', self.symbol)
                
            return {
                'name': chinese_name,
                'currency': info.get('currency', 'CNY' if self.symbol.endswith(('.SS', '.SZ')) else 'USD'),
                'sector': info.get('sector', '未知'),
                'industry': info.get('industry', '未知'),
                'market': '上海' if self.symbol.endswith('.SS') else '深圳' if self.symbol.endswith('.SZ') else '海外'
            }
        except:
            return {
                'name': self.symbol,
                'currency': 'CNY' if self.symbol.endswith(('.SS', '.SZ')) else 'USD',
                'sector': '未知',
                'industry': '未知',
                'market': '上海' if self.symbol.endswith('.SS') else '深圳' if self.symbol.endswith('.SZ') else '海外'
            }
    
    def get_historical_data(self, period="1y"):
        """
        获取股票历史数据
        
        参数:
        period: 数据周期，默认为"1y"（一年）
        
        返回:
        DataFrame包含以下字段:
        - Date: 交易日期（索引）
        - Open: 开盘价
        - High: 最高价  
        - Low: 最低价
        - Close: 收盘价
        - Volume: 成交量
        - Dividends: 股息
        - Stock Splits: 股票分割
        - y_Close: 目标收盘价 - 下一交易日的收盘价格（用于模型训练和预测）
        - y_High: 目标最高价 - 下一交易日的最高价格（用于模型训练和预测）
        - y_Low: 目标最低价 - 下一交易日的最低价格（用于模型训练和预测）
        """
        df = self.stock.history(period=period)
        
        # 删除成交量为0的日期（非交易日）
        df = df[df['Volume'] > 0]
        
        # 重置索引以确保连续性
        df = df.reset_index().set_index('Date')
        print("获取股票历史数据start！")
        df.info()
        df.head(100)
        print("获取股票历史数据end！")
        # 创建目标变量 - 下一交易日的价格，最新行用当前日数据填充
        df['y_Close'] = df['Close'].shift(-1).fillna(df['Close'])
        df['y_High'] = df['High'].shift(-1).fillna(df['High'])
        df['y_Low'] = df['Low'].shift(-1).fillna(df['Low'])
        
        return df
    
    def calculate_technical_indicators(self, df):
        """
        计算技术指标
        
        为DataFrame添加以下技术指标字段:
        
        移动平均线指标:
        - MA20: 20日移动平均线
        - MA50: 50日移动平均线
        
        相对强弱指标:
        - RSI: 相对强弱指数（14日）
        
        布林带指标:
        - BOLL_MB: 布林带中轨（20日均线）
        - BOLL_STD: 布林带标准差
        - BOLL_UP: 布林带上轨
        - BOLL_DOWN: 布林带下轨
        
        成交量指标:
        - Volume_MA20: 20日成交量均线
        - Volume_Ratio: 量比（当日成交量/20日均量）
        - Volume_Change: 成交量变化率
        
        换手率指标:
        - Turnover_Rate: 日换手率（%）
        - Turnover_Rate_5d: 5日平均换手率
        - Turnover_Rate_Ratio: 换手率相对值
        - Turnover_Rate_Change: 换手率变化率
        
        主力资金指标:
        - Big_Order: 大单标识（0/1）
        - Big_Order_Value: 大单净额
        - Big_Order_Net_5d: 5日大单净流入
        - Volume_Value_5d: 5日总成交额
        - Big_Order_Ratio_5d: 5日大单净流入比例
        - Money_Flow: 日主力资金净流入
        - Money_Flow_Acc_5d: 5日累计资金流入
        - Money_Flow_Acc_10d: 10日累计资金流入
        - Money_Flow_Strength: 主力资金强度
        - Money_Flow_Strength_5d: 5日平均资金强度
        - Money_Flow_Direction: 资金流向（+1流入/-1流出）
        - Money_Flow_Direction_5d: 5日平均资金流向
        
        交易信号:
        - Buy_Signal: 买入信号（0/1）
        - Sell_Signal: 卖出信号（0/1）
        - MA20_Cross: 均线多头排列标识
        - MA_Cross_Signal: 均线交叉信号
        """
        # 计算移动平均线
        df['MA20'] = df['Close'].rolling(window=20).mean()  # 20日移动平均线
        df['MA50'] = df['Close'].rolling(window=50).mean()  # 50日移动平均线
        
        # 计算相对强弱指标 (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))  # 相对强弱指数
        
        # 计算BOLL指标（布林带）
        df['BOLL_MB'] = df['Close'].rolling(window=20).mean()  # 布林带中轨
        df['BOLL_STD'] = df['Close'].rolling(window=20).std()  # 布林带标准差
        df['BOLL_UP'] = df['BOLL_MB'] + 2 * df['BOLL_STD']    # 布林带上轨
        df['BOLL_DOWN'] = df['BOLL_MB'] - 2 * df['BOLL_STD']  # 布林带下轨
        
        # 添加成交量指标
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()  # 20日成交量均线
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']       # 量比
        df['Volume_Change'] = df['Volume'].pct_change()             # 成交量变化率
        
        # 添加换手率指标
        # 获取股票信息中的总股本
        try:
            info = self.stock.info
            shares_outstanding = info.get('sharesOutstanding', None)
            
            if shares_outstanding and shares_outstanding > 0:
                # 计算日换手率 = 成交量 / 总股本
                df['Turnover_Rate'] = df['Volume'] / shares_outstanding * 100  # 转为百分比
            else:
                # 如果无法获取总股本，使用成交量相对值作为替代
                df['Turnover_Rate'] = df['Volume'] / df['Volume'].rolling(window=20).mean() * 100
                
            # 计算5日平均换手率
            df['Turnover_Rate_5d'] = df['Turnover_Rate'].rolling(window=5).mean()
            
            # 计算换手率相对值 - 当日换手率与5日平均的比值
            df['Turnover_Rate_Ratio'] = df['Turnover_Rate'] / df['Turnover_Rate_5d']
            
            # 计算换手率变化率
            df['Turnover_Rate_Change'] = df['Turnover_Rate'].pct_change()
            
        except Exception as e:
            print(f"计算换手率时出错: {e}")
            # 使用成交量相对值作为替代
            df['Turnover_Rate'] = df['Volume'] / df['Volume'].rolling(window=20).mean() * 100
            df['Turnover_Rate_5d'] = df['Turnover_Rate'].rolling(window=5).mean()
            df['Turnover_Rate_Ratio'] = df['Turnover_Rate'] / df['Turnover_Rate_5d']
            df['Turnover_Rate_Change'] = df['Turnover_Rate'].pct_change()
            
            # 添加主力大单特征
            # 1. 大单判断标准：成交量超过20日均量的2倍
            df['Big_Order'] = (df['Volume'] > df['Volume_MA20'] * 2).astype(float)
            
            # 2. 计算大单净额：根据价格变动方向判断大单性质
            df['Big_Order_Value'] = 0.0
            # 当日有大单且价格上涨 - 买入大单
            df.loc[(df['Big_Order'] > 0) & (df['Close'] > df['Open']), 'Big_Order_Value'] = \
                df['Volume'] * df['Close'] * (df['Close'] - df['Open']) / df['Open']
            # 当日有大单且价格下跌 - 卖出大单
            df.loc[(df['Big_Order'] > 0) & (df['Close'] < df['Open']), 'Big_Order_Value'] = \
                -df['Volume'] * df['Close'] * (df['Open'] - df['Close']) / df['Open']
            
            # 3. 计算大单净流入比例：近5日大单净额占总成交额的比例
            df['Big_Order_Net_5d'] = df['Big_Order_Value'].rolling(window=5).sum()
            df['Volume_Value_5d'] = (df['Volume'] * df['Close']).rolling(window=5).sum()
            # 避免除零错误
            df['Big_Order_Ratio_5d'] = df['Big_Order_Net_5d'] / df['Volume_Value_5d'].replace(0, 1e-8)
            
            # 4. 添加主力资金流入流出特征
            # 计算日主力资金净流入 - 基于价格变动和成交量估算
            price_range = df['High'] - df['Low']
            # 当价格区间为0时，使用一个很小的值避免除零
            price_range = price_range.replace(0, 1e-8)
            df['Money_Flow'] = df['Volume'] * df['Close'] * (2 * (df['Close'] - df['Low']) - (df['High'] - df['Close'])) / price_range
            
            # 5. 计算主力资金累计流入流出
            df['Money_Flow_Acc_5d'] = df['Money_Flow'].rolling(window=5).sum()
            df['Money_Flow_Acc_10d'] = df['Money_Flow'].rolling(window=10).sum()
            
            # 6. 计算主力资金强度指标 - 资金流量与成交额的比值
            volume_value = df['Volume'] * df['Close']
            # 避免除零错误
            volume_value = volume_value.replace(0, 1e-8)
            df['Money_Flow_Strength'] = df['Money_Flow'] / volume_value
            df['Money_Flow_Strength_5d'] = df['Money_Flow_Strength'].rolling(window=5).mean()
            
            # 7. 计算主力资金流向指标 - 正值表示流入，负值表示流出
            df['Money_Flow_Direction'] = np.sign(df['Money_Flow'])
            df['Money_Flow_Direction_5d'] = df['Money_Flow_Direction'].rolling(window=5).mean()
            
        # 添加买卖点判断
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0
        
        # BOLL突破策略
        df.loc[(df['Close'] < df['BOLL_DOWN']) & (df['RSI'] < 30), 'Buy_Signal'] = 1
        df.loc[(df['Close'] > df['BOLL_UP']) & (df['RSI'] > 70), 'Sell_Signal'] = 1
        
        # 均线金叉死叉策略
        df['MA20_Cross'] = (df['MA20'] > df['MA50']).astype(int)  # 转换为0/1
        df['MA_Cross_Signal'] = df['MA20_Cross'].diff()  # 计算差值
        
        # 金叉买入信号
        df.loc[df['MA_Cross_Signal'] == 1, 'Buy_Signal'] = 1
        # 死叉卖出信号
        df.loc[df['MA_Cross_Signal'] == -1, 'Sell_Signal'] = 1
        
        return df

    def get_trading_signals(self, df):
        """获取最近的买卖点信息"""
        last_buy = df[df['Buy_Signal'] == 1].iloc[-3:] if len(df[df['Buy_Signal'] == 1]) > 0 else pd.DataFrame()
        last_sell = df[df['Sell_Signal'] == 1].iloc[-3:] if len(df[df['Sell_Signal'] == 1]) > 0 else pd.DataFrame()
        
        signals = {
            'buy_points': [],
            'sell_points': []
        }
        
        for idx, row in last_buy.iterrows():
            signals['buy_points'].append({
                'date': idx.strftime('%Y-%m-%d'),
                'price': row['Close'],
                'reason': self._get_signal_reason(row)
            })
            
        for idx, row in last_sell.iterrows():
            signals['sell_points'].append({
                'date': idx.strftime('%Y-%m-%d'),
                'price': row['Close'],
                'reason': self._get_signal_reason(row)
            })
            
        return signals
        
    def _get_signal_reason(self, row):
        """生成买卖信号的原因说明"""
        reasons = []
        
        if row['Close'] < row['BOLL_DOWN'] and row['RSI'] < 30:
            reasons.append('布林带下轨突破+超卖')
        elif row['Close'] > row['BOLL_UP'] and row['RSI'] > 70:
            reasons.append('布林带上轨突破+超买')
            
        # 修改均线交叉判断逻辑
        if row['MA_Cross_Signal'] == 1:
            reasons.append('均线金叉')
        elif row['MA_Cross_Signal'] == -1:
            reasons.append('均线死叉')
                
        # 添加主力大单信号
        if 'Big_Order' in row and row['Big_Order'] > 0:
            if row['Close'] > row['Open']:
                reasons.append('大单买入')
            elif row['Close'] < row['Open']:
                reasons.append('大单卖出')
        
        # 添加大单净流入信号
        if 'Big_Order_Ratio_5d' in row and pd.notna(row['Big_Order_Ratio_5d']):
            if row['Big_Order_Ratio_5d'] > 0.1:  # 大单净流入超过10%
                reasons.append('大单持续净流入')
            elif row['Big_Order_Ratio_5d'] < -0.1:  # 大单净流出超过10%
                reasons.append('大单持续净流出')
        
        # 添加主力资金流向信号
        if 'Money_Flow_Strength' in row and pd.notna(row['Money_Flow_Strength']):
            if row['Money_Flow_Strength'] > 0.1 and row['Close'] > row['Open']:
                reasons.append('主力资金大幅流入')
            elif row['Money_Flow_Strength'] < -0.1 and row['Close'] < row['Open']:
                reasons.append('主力资金大幅流出')
        
        return '、'.join(reasons)
    
    def predict_next_day(self, df):
        # 首先计算技术指标
        self.calculate_technical_indicators(df)
        
        # 获取最后交易日期
        last_trade_date = df.index[-1]
        next_trade_date = last_trade_date + pd.Timedelta(days=1)
        
        # 如果是周末，调整到下一个周一
        while next_trade_date.weekday() >= 5:  # 5是周六，6是周日
            next_trade_date += pd.Timedelta(days=1)
            
        # 创建目标变量 - 下一交易日的价格，最新行用当前日数据填充
        df['y_Close'] = df['Close'].shift(-1).fillna(df['Close'])
        df['y_High'] = df['High'].shift(-1).fillna(df['High'])
        df['y_Low'] = df['Low'].shift(-1).fillna(df['Low'])
        
        # 原有的预测字段（保持兼容性）
        df['Prediction'] = df['Close'].shift(-1)
        df['Return'] = df['Close'].pct_change()
        
        # 创建特征
        df['MA20_Ratio'] = df['Close'] / df['MA20']
        df['MA50_Ratio'] = df['Close'] / df['MA50']
        
        # 添加BOLL指标特征
        df['BOLL_Position'] = (df['Close'] - df['BOLL_MB']) / (df['BOLL_UP'] - df['BOLL_DOWN'])
        df['BOLL_Width'] = (df['BOLL_UP'] - df['BOLL_DOWN']) / df['BOLL_MB']
        
        # 技术指标组合特征
        df['MA_Divergence'] = (df['MA20'] - df['MA50']) / df['MA50']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Volume_Price_Trend'] = df['Volume'] * (df['Close'] - df['Open']) / df['Open']

        # 时间特征
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)

        # 波动率特征
        df['Volatility_10d'] = df['Return'].rolling(window=10).std()
        df['Volatility_Ratio'] = df['Volatility_10d'] / df['Volatility_10d'].rolling(window=20).mean()
        # 删除所有包含 NaN 的行
        df_clean = df.dropna()
        
        # 检查数据是否足够
        if len(df_clean) < 20:
            return {
                'close': df['Close'].iloc[-1],
                'high': df['High'].iloc[-1], 
                'low': df['Low'].iloc[-1],
                'date': next_trade_date,
                'model': None,
                'metrics': None
            }
        
        # 现在可以安全地使用 df_clean
        y_close = df_clean['y_Close'].values
        y_high = df_clean['y_High'].values
        y_low = df_clean['y_Low'].values

        
        # 原有的预测字段（保持兼容性）
        df['Prediction'] = df['Close'].shift(-1)
        df['Return'] = df['Close'].pct_change()
        
        # 创建特征
        df['MA20_Ratio'] = df['Close'] / df['MA20']
        df['MA50_Ratio'] = df['Close'] / df['MA50']
        
        # 添加BOLL指标特征
        df['BOLL_Position'] = (df['Close'] - df['BOLL_MB']) / (df['BOLL_UP'] - df['BOLL_DOWN'])
        df['BOLL_Width'] = (df['BOLL_UP'] - df['BOLL_DOWN']) / df['BOLL_MB']

        




        # 在 calculate_technical_indicators 中添加
        # 技术指标组合特征
        df['MA_Divergence'] = (df['MA20'] - df['MA50']) / df['MA50']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Volume_Price_Trend'] = df['Volume'] * (df['Close'] - df['Open']) / df['Open']

        # 时间特征
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)

        # 波动率特征
        df['Volatility_10d'] = df['Return'].rolling(window=10).std()
        df['Volatility_Ratio'] = df['Volatility_10d'] / df['Volatility_10d'].rolling(window=20).mean()
        # 检查数据是否足够
        if len(df_clean) < 20:
            return {
                'close': df['Close'].iloc[-1],
                'high': df['High'].iloc[-1], 
                'low': df['Low'].iloc[-1],
                'date': next_trade_date,
                'model': None,
                'metrics': None
            }

   
        
            
        # 准备训练数据
        features = ['MA20_Ratio', 'MA50_Ratio', 'RSI', 'Return', 'BOLL_Position', 'BOLL_Width',
                   'Volume_Ratio', 'Volume_Change']
        
        # 添加振幅特征（如果存在）
        if 'Amplitude' in df_clean.columns:
            features.extend(['Amplitude', 'Amplitude_5d', 'Amplitude_Ratio', 'Amplitude_Change'])
        
        # 添加流通市值特征（如果存在）
        if 'Market_Cap_Ratio' in df_clean.columns:
            features.extend(['Market_Cap_Change', 'Market_Cap_Ratio', 'Market_Cap_Volatility'])
        
        # 添加换手率特征（如果存在）
        if 'Turnover_Rate' in df_clean.columns:
            features.extend(['Turnover_Rate', 'Turnover_Rate_5d', 'Turnover_Rate_Ratio', 'Turnover_Rate_Change'])
        
        # 添加主力大单特征（如果存在）
        if 'Big_Order_Ratio_5d' in df_clean.columns:
            features.extend(['Big_Order', 'Big_Order_Ratio_5d', 'Money_Flow_Strength', 'Money_Flow_Direction_5d'])
        
        X = df_clean[features].values


        # 根据数据量动态调整
        data_length = len(df_clean)
        if data_length > 200:
            seq_length = 20
        elif data_length > 100:
            seq_length = 15
        else:
            seq_length = 10
        print(f"特征列表: {features}")
        print(f"特征数据形状: {X.shape}")
        
        # 数据标准化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        # 避免除零错误
        X_std = np.where(X_std == 0, 1e-8, X_std)
        X_normalized = (X - X_mean) / X_std
        
        # 获取三个目标值 - 从已创建的目标变量列中获取
        y_close = df_clean['y_Close'].values
        y_high = df_clean['y_High'].values
        y_low = df_clean['y_Low'].values
        
        # 标准化三个目标值
        y_close_mean, y_close_std = y_close.mean(), y_close.std()
        y_high_mean, y_high_std = y_high.mean(), y_high.std()
        y_low_mean, y_low_std = y_low.mean(), y_low.std()
        
        # 避免除零错误
        y_close_std = max(y_close_std, 1e-8)
        y_high_std = max(y_high_std, 1e-8)
        y_low_std = max(y_low_std, 1e-8)
        
        y_close_normalized = (y_close - y_close_mean) / y_close_std
        y_high_normalized = (y_high - y_high_mean) / y_high_std
        y_low_normalized = (y_low - y_low_mean) / y_low_std
        
        # 创建数据集
        dataset = StockDataset(X_normalized, y_close_normalized, y_high_normalized, y_low_normalized, seq_length)
        
        # 确保数据集有足够的样本
        if len(dataset) < 10:
            return {
                'close': df['Close'].iloc[-1],
                'high': df['High'].iloc[-1],
                'low': df['Low'].iloc[-1],
                'date': next_trade_date,
                'model': None,
                'metrics': None
            }
        
        train_size = max(1, int(0.8 * len(dataset)))
        test_size = len(dataset) - train_size
        
        if test_size <= 0:
            test_size = 1
            train_size = len(dataset) - 1
            
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # 创建数据加载器
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 初始化模型和优化器
        model = HybridModel(len(features))
        # criterion = nn.MSELoss()
        criterion = WeightedMSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', patience=10, factor=0.5
        # )
        
        # 训练模型
        #  epochs = 2000

        # 在 predict_next_day 方法中
        epochs = 500  # 减少训练轮数，避免过拟合
        batch_size = 16  # 减小批次大小
        learning_rate = 0.0005  # 降低学习率

        # 使用更好的优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # 改进学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        model.train()
        print("\n开始训练模型...")
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            
            # 添加这一行来更新学习率
            scheduler.step()
            
            # 每10个epoch打印一次平均损失
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / batch_count
                current_lr = scheduler.get_last_lr()[0]  # 可选：打印当前学习率
                print(f"Epoch [{epoch+1}/{epochs}], 平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}")
        
        print("模型训练完成！\n")
        
        # 评估模型
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                output = model(batch_X)
                predictions.extend(output.numpy())
                actuals.extend(batch_y.numpy())
        
        # 反标准化预测结果 - 只使用收盘价计算指标
        predictions_close = np.array([p[0] for p in predictions]) * y_close_std + y_close_mean
        actuals_close = np.array([a[0] for a in actuals]) * y_close_std + y_close_mean
        
        # 计算评分
        mse = mean_squared_error(actuals_close, predictions_close)
        r2 = r2_score(actuals_close, predictions_close)
        
        # 预测下一天
        last_sequence = torch.FloatTensor(
            (X_normalized[-seq_length:]).reshape(1, seq_length, -1)
        )
        
        # 计算特征重要性
        if len(X_normalized) > seq_length:
            test_X = torch.FloatTensor(X_normalized).unfold(0, seq_length, 1).transpose(1, 2)
            y_for_importance = torch.FloatTensor(y_close_normalized[seq_length-1:])
            
            # 确保长度匹配
            min_len = min(len(test_X), len(y_for_importance))
            test_X = test_X[:min_len]
            y_for_importance = y_for_importance[:min_len]
            
            if len(test_X) > 0 and len(y_for_importance) > 0:
                feature_importance = model.calculate_feature_importance(test_X, y_for_importance)
                print("\n=== 特征重要性分析 ===")
                for i, (feature, importance) in enumerate(zip(features, feature_importance)):
                    print(f"{feature}: {importance:.4f}")
                print("========================\n")
        
        model.eval()
        with torch.no_grad():
            predictions = model(last_sequence)[0].numpy()
            next_day_close = predictions[0] * y_close_std + y_close_mean
            next_day_high = predictions[1] * y_high_std + y_high_mean
            next_day_low = predictions[2] * y_low_std + y_low_mean
        
        # 保存训练好的模型
        import os
        model_dir = './models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存模型状态字典和相关参数
        model_save_path = os.path.join(model_dir, f'{self.symbol}_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': len(features),
            'features': features,
            'scaler_params': {
                'X_mean': X_mean if 'X_mean' in locals() else None,
                'X_std': X_std if 'X_std' in locals() else None,
                'y_close_mean': y_close_mean,
                'y_close_std': y_close_std,
                'y_high_mean': y_high_mean,
                'y_high_std': y_high_std,
                'y_low_mean': y_low_mean,
                'y_low_std': y_low_std
            },
            'seq_length': seq_length,
            'model_metrics': {
                'mse': mse,
                'r2': r2
            }
        }, model_save_path)
        
        print(f"模型已保存到: {model_save_path}")
        
        # 在predict_next_day方法的返回部分（大约第620-630行）
        return {
            'next_day_close': next_day_close,
            'next_day_high': next_day_high,
            'next_day_low': next_day_low,
            'date': next_trade_date,
            'confidence': float(r2) if r2 > 0 else 0.0,
            'model_performance': {
                'mse': float(mse),
                'r2': float(r2),
                'features_count': len(features),
                'features': features,
                'feature_importances': feature_importance if 'feature_importance' in locals() else None
            }
        }
        
        # 添加振幅特征
        # 1. 日振幅 = (最高价 - 最低价) / 昨日收盘价 * 100
        df['Amplitude'] = (df['High'] - df['Low']) / df['Close'].shift(1) * 100
        
        # 2. 5日平均振幅
        df['Amplitude_5d'] = df['Amplitude'].rolling(window=5).mean()
        
        # 3. 振幅相对值 = 当日振幅 / 5日平均振幅
        df['Amplitude_Ratio'] = df['Amplitude'] / df['Amplitude_5d']
        
        # 4. 振幅变化率
        df['Amplitude_Change'] = df['Amplitude'].pct_change()
        
        # 添加流通市值特征
        try:
            info = self.stock.info
            shares_outstanding = info.get('sharesOutstanding', None)
            market_cap = info.get('marketCap', None)
            
            if shares_outstanding and shares_outstanding > 0:
                # 1. 流通市值 = 股价 * 流通股本
                df['Market_Cap'] = df['Close'] * shares_outstanding
                
                # 2. 流通市值变化率
                df['Market_Cap_Change'] = df['Market_Cap'].pct_change()
                
                # 3. 流通市值相对值 = 当前市值 / 20日平均市值
                df['Market_Cap_20d'] = df['Market_Cap'].rolling(window=20).mean()
                df['Market_Cap_Ratio'] = df['Market_Cap'] / df['Market_Cap_20d']
                
                # 4. 市值波动率 = 20日市值标准差 / 20日平均市值
                df['Market_Cap_Volatility'] = df['Market_Cap'].rolling(window=20).std() / df['Market_Cap_20d']
                
            elif market_cap:
                # 如果无法获取流通股本，使用总市值作为替代
                df['Market_Cap'] = market_cap
                df['Market_Cap_Change'] = 0  # 静态值，变化率为0
                df['Market_Cap_Ratio'] = 1   # 相对值为1
                df['Market_Cap_Volatility'] = 0  # 波动率为0
            else:
                # 使用成交额作为市值的替代指标
                df['Market_Cap'] = df['Volume'] * df['Close']
                df['Market_Cap_Change'] = df['Market_Cap'].pct_change()
                df['Market_Cap_20d'] = df['Market_Cap'].rolling(window=20).mean()
                df['Market_Cap_Ratio'] = df['Market_Cap'] / df['Market_Cap_20d']
                df['Market_Cap_Volatility'] = df['Market_Cap'].rolling(window=20).std() / df['Market_Cap_20d']
                
        except Exception as e:
            print(f"计算流通市值时出错: {e}")
            # 使用成交额作为替代
            df['Market_Cap'] = df['Volume'] * df['Close']
            df['Market_Cap_Change'] = df['Market_Cap'].pct_change()
            df['Market_Cap_20d'] = df['Market_Cap'].rolling(window=20).mean()
            df['Market_Cap_Ratio'] = df['Market_Cap'] / df['Market_Cap_20d']
            df['Market_Cap_Volatility'] = df['Market_Cap'].rolling(window=20).std() / df['Market_Cap_20d']
        
        model.eval()
        with torch.no_grad():
            predictions = model(last_sequence)[0].numpy()
            next_day_close = predictions[0] * y_close_std + y_close_mean
            next_day_high = predictions[1] * y_high_std + y_high_mean
            next_day_low = predictions[2] * y_low_std + y_low_mean
        
        return {
            'close': next_day_close,
            'high': next_day_high,
            'low': next_day_low,
            'date': next_trade_date,
            'model': model,
            'metrics': {
                'mse': mse, 
                'r2': r2,
                'features': features
            }
        }

    def load_model(self, model_path=None):
        """加载已保存的模型"""
        if model_path is None:
            model_path = f'./models/{self.symbol}_model.pth'
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        try:
            checkpoint = torch.load(model_path)
            model = HybridModel(checkpoint['input_size'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"模型加载成功: {model_path}")
            print(f"模型性能 - MSE: {checkpoint['model_metrics']['mse']:.4f}, R²: {checkpoint['model_metrics']['r2']:.4f}")
            
            return {
                'model': model,
                'features': checkpoint['features'],
                'scaler_params': checkpoint['scaler_params'],
                'seq_length': checkpoint['seq_length'],
                'metrics': checkpoint['model_metrics']
            }
        except Exception as e:
            print(f"加载模型时出错: {e}")

    def get_current_price(self):
        """获取股票当前价格信息"""
        try:
            # 获取最近1天的数据来获取当前价格
            current_data = self.stock.history(period="1d")
            
            if current_data.empty:
                # 如果当天没有数据，尝试获取最近5天的数据
                current_data = self.stock.history(period="5d")
                
            if current_data.empty:
                return {
                    "error": "无法获取当前股价数据",
                    "symbol": self.symbol
                }
            
            # 获取最新的交易数据
            latest = current_data.iloc[-1]
            latest_date = current_data.index[-1]
            
            # 计算涨跌幅
            if len(current_data) >= 2:
                previous_close = current_data.iloc[-2]['Close']
                change = latest['Close'] - previous_close
                change_percent = (change / previous_close) * 100
            else:
                # 如果只有一天数据，尝试从股票信息中获取前收盘价
                info = self.stock.info
                previous_close = info.get('previousClose', latest['Open'])
                change = latest['Close'] - previous_close
                change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
            
            # 获取股票基本信息
            stock_info = self.get_stock_info()
            
            return {
                "symbol": self.symbol,
                "name": stock_info.get('name', self.symbol),
                "current_price": float(latest['Close']),
                "open_price": float(latest['Open']),
                "high_price": float(latest['High']),
                "low_price": float(latest['Low']),
                "volume": int(latest['Volume']),
                "change": float(change),
                "change_percent": float(change_percent),
                "previous_close": float(previous_close),
                "date": latest_date.strftime('%Y-%m-%d'),
                "currency": stock_info.get('currency', 'CNY'),
                "market": stock_info.get('market', '未知'),
                "sector": stock_info.get('sector', '未知'),
                "industry": stock_info.get('industry', '未知')
            }
            
        except Exception as e:
            return {
                "error": f"获取当前股价失败: {str(e)}",
                "symbol": self.symbol
            }
    
    def _get_signal_reason(self, row):
        """生成买卖信号的原因说明"""
        reasons = []
        
        if row['Close'] < row['BOLL_DOWN'] and row['RSI'] < 30:
            reasons.append('布林带下轨突破+超卖')
        elif row['Close'] > row['BOLL_UP'] and row['RSI'] > 70:
            reasons.append('布林带上轨突破+超买')
            
        # 修改均线交叉判断逻辑
        if row['MA_Cross_Signal'] == 1:
            reasons.append('均线金叉')
        elif row['MA_Cross_Signal'] == -1:
            reasons.append('均线死叉')
                
        # 添加主力大单信号
        if 'Big_Order' in row and row['Big_Order'] > 0:
            if row['Close'] > row['Open']:
                reasons.append('大单买入')
            elif row['Close'] < row['Open']:
                reasons.append('大单卖出')
        
        # 添加大单净流入信号
        if 'Big_Order_Ratio_5d' in row and pd.notna(row['Big_Order_Ratio_5d']):
            if row['Big_Order_Ratio_5d'] > 0.1:  # 大单净流入超过10%
                reasons.append('大单持续净流入')
            elif row['Big_Order_Ratio_5d'] < -0.1:  # 大单净流出超过10%
                reasons.append('大单持续净流出')
        
        # 添加主力资金流向信号
        if 'Money_Flow_Strength' in row and pd.notna(row['Money_Flow_Strength']):
            if row['Money_Flow_Strength'] > 0.1 and row['Close'] > row['Open']:
                reasons.append('主力资金大幅流入')
            elif row['Money_Flow_Strength'] < -0.1 and row['Close'] < row['Open']:
                reasons.append('主力资金大幅流出')
        
        return '、'.join(reasons)
    
    def predict_next_day(self, df):
        # 首先计算技术指标
        self.calculate_technical_indicators(df)
        
        # 获取最后交易日期
        last_trade_date = df.index[-1]
        next_trade_date = last_trade_date + pd.Timedelta(days=1)
        
        # 如果是周末，调整到下一个周一
        while next_trade_date.weekday() >= 5:  # 5是周六，6是周日
            next_trade_date += pd.Timedelta(days=1)
            
        # 创建目标变量 - 下一交易日的价格，最新行用当前日数据填充
        df['y_Close'] = df['Close'].shift(-1).fillna(df['Close'])
        df['y_High'] = df['High'].shift(-1).fillna(df['High'])
        df['y_Low'] = df['Low'].shift(-1).fillna(df['Low'])
        
        # 原有的预测字段（保持兼容性）
        df['Prediction'] = df['Close'].shift(-1)
        df['Return'] = df['Close'].pct_change()
        
        # 创建特征
        df['MA20_Ratio'] = df['Close'] / df['MA20']
        df['MA50_Ratio'] = df['Close'] / df['MA50']
        
        # 添加BOLL指标特征
        df['BOLL_Position'] = (df['Close'] - df['BOLL_MB']) / (df['BOLL_UP'] - df['BOLL_DOWN'])
        df['BOLL_Width'] = (df['BOLL_UP'] - df['BOLL_DOWN']) / df['BOLL_MB']
        
        # 技术指标组合特征
        df['MA_Divergence'] = (df['MA20'] - df['MA50']) / df['MA50']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Volume_Price_Trend'] = df['Volume'] * (df['Close'] - df['Open']) / df['Open']

        # 时间特征
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)

        # 波动率特征
        df['Volatility_10d'] = df['Return'].rolling(window=10).std()
        df['Volatility_Ratio'] = df['Volatility_10d'] / df['Volatility_10d'].rolling(window=20).mean()
        # 删除所有包含 NaN 的行
        df_clean = df.dropna()
        
        # 检查数据是否足够
        if len(df_clean) < 20:
            return {
                'close': df['Close'].iloc[-1],
                'high': df['High'].iloc[-1], 
                'low': df['Low'].iloc[-1],
                'date': next_trade_date,
                'model': None,
                'metrics': None
            }
        
        # 现在可以安全地使用 df_clean
        y_close = df_clean['y_Close'].values
        y_high = df_clean['y_High'].values
        y_low = df_clean['y_Low'].values

        
        # 原有的预测字段（保持兼容性）
        df['Prediction'] = df['Close'].shift(-1)
        df['Return'] = df['Close'].pct_change()
        
        # 创建特征
        df['MA20_Ratio'] = df['Close'] / df['MA20']
        df['MA50_Ratio'] = df['Close'] / df['MA50']
        
        # 添加BOLL指标特征
        df['BOLL_Position'] = (df['Close'] - df['BOLL_MB']) / (df['BOLL_UP'] - df['BOLL_DOWN'])
        df['BOLL_Width'] = (df['BOLL_UP'] - df['BOLL_DOWN']) / df['BOLL_MB']

        




        # 在 calculate_technical_indicators 中添加
        # 技术指标组合特征
        df['MA_Divergence'] = (df['MA20'] - df['MA50']) / df['MA50']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Volume_Price_Trend'] = df['Volume'] * (df['Close'] - df['Open']) / df['Open']

        # 时间特征
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)

        # 波动率特征
        df['Volatility_10d'] = df['Return'].rolling(window=10).std()
        df['Volatility_Ratio'] = df['Volatility_10d'] / df['Volatility_10d'].rolling(window=20).mean()
        # 检查数据是否足够
        if len(df_clean) < 20:
            return {
                'close': df['Close'].iloc[-1],
                'high': df['High'].iloc[-1], 
                'low': df['Low'].iloc[-1],
                'date': next_trade_date,
                'model': None,
                'metrics': None
            }

   
        
            
        # 准备训练数据
        features = ['MA20_Ratio', 'MA50_Ratio', 'RSI', 'Return', 'BOLL_Position', 'BOLL_Width',
                   'Volume_Ratio', 'Volume_Change']
        
        # 添加振幅特征（如果存在）
        if 'Amplitude' in df_clean.columns:
            features.extend(['Amplitude', 'Amplitude_5d', 'Amplitude_Ratio', 'Amplitude_Change'])
        
        # 添加流通市值特征（如果存在）
        if 'Market_Cap_Ratio' in df_clean.columns:
            features.extend(['Market_Cap_Change', 'Market_Cap_Ratio', 'Market_Cap_Volatility'])
        
        # 添加换手率特征（如果存在）
        if 'Turnover_Rate' in df_clean.columns:
            features.extend(['Turnover_Rate', 'Turnover_Rate_5d', 'Turnover_Rate_Ratio', 'Turnover_Rate_Change'])
        
        # 添加主力大单特征（如果存在）
        if 'Big_Order_Ratio_5d' in df_clean.columns:
            features.extend(['Big_Order', 'Big_Order_Ratio_5d', 'Money_Flow_Strength', 'Money_Flow_Direction_5d'])
        
        X = df_clean[features].values


        # 根据数据量动态调整
        data_length = len(df_clean)
        if data_length > 200:
            seq_length = 20
        elif data_length > 100:
            seq_length = 15
        else:
            seq_length = 10
        print(f"特征列表: {features}")
        print(f"特征数据形状: {X.shape}")
        
        # 数据标准化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        # 避免除零错误
        X_std = np.where(X_std == 0, 1e-8, X_std)
        X_normalized = (X - X_mean) / X_std
        
        # 获取三个目标值 - 从已创建的目标变量列中获取
        y_close = df_clean['y_Close'].values
        y_high = df_clean['y_High'].values
        y_low = df_clean['y_Low'].values
        
        # 标准化三个目标值
        y_close_mean, y_close_std = y_close.mean(), y_close.std()
        y_high_mean, y_high_std = y_high.mean(), y_high.std()
        y_low_mean, y_low_std = y_low.mean(), y_low.std()
        
        # 避免除零错误
        y_close_std = max(y_close_std, 1e-8)
        y_high_std = max(y_high_std, 1e-8)
        y_low_std = max(y_low_std, 1e-8)
        
        y_close_normalized = (y_close - y_close_mean) / y_close_std
        y_high_normalized = (y_high - y_high_mean) / y_high_std
        y_low_normalized = (y_low - y_low_mean) / y_low_std
        
        # 创建数据集
        dataset = StockDataset(X_normalized, y_close_normalized, y_high_normalized, y_low_normalized, seq_length)
        
        # 确保数据集有足够的样本
        if len(dataset) < 10:
            return {
                'close': df['Close'].iloc[-1],
                'high': df['High'].iloc[-1],
                'low': df['Low'].iloc[-1],
                'date': next_trade_date,
                'model': None,
                'metrics': None
            }
        
        train_size = max(1, int(0.8 * len(dataset)))
        test_size = len(dataset) - train_size
        
        if test_size <= 0:
            test_size = 1
            train_size = len(dataset) - 1
            
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # 创建数据加载器
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 初始化模型和优化器
        model = HybridModel(len(features))
        # criterion = nn.MSELoss()
        criterion = WeightedMSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', patience=10, factor=0.5
        # )
        
        # 训练模型
        #  epochs = 2000

        # 在 predict_next_day 方法中
        epochs = 500  # 减少训练轮数，避免过拟合
        batch_size = 16  # 减小批次大小
        learning_rate = 0.0005  # 降低学习率

        # 使用更好的优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # 改进学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        model.train()
        print("\n开始训练模型...")
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            
            # 添加这一行来更新学习率
            scheduler.step()
            
            # 每10个epoch打印一次平均损失
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / batch_count
                current_lr = scheduler.get_last_lr()[0]  # 可选：打印当前学习率
                print(f"Epoch [{epoch+1}/{epochs}], 平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}")
        
        print("模型训练完成！\n")
        
        # 评估模型
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                output = model(batch_X)
                predictions.extend(output.numpy())
                actuals.extend(batch_y.numpy())
        
        # 反标准化预测结果 - 只使用收盘价计算指标
        predictions_close = np.array([p[0] for p in predictions]) * y_close_std + y_close_mean
        actuals_close = np.array([a[0] for a in actuals]) * y_close_std + y_close_mean
        
        # 计算评分
        mse = mean_squared_error(actuals_close, predictions_close)
        r2 = r2_score(actuals_close, predictions_close)
        
        # 预测下一天
        last_sequence = torch.FloatTensor(
            (X_normalized[-seq_length:]).reshape(1, seq_length, -1)
        )
        
        # 计算特征重要性
        if len(X_normalized) > seq_length:
            test_X = torch.FloatTensor(X_normalized).unfold(0, seq_length, 1).transpose(1, 2)
            y_for_importance = torch.FloatTensor(y_close_normalized[seq_length-1:])
            
            # 确保长度匹配
            min_len = min(len(test_X), len(y_for_importance))
            test_X = test_X[:min_len]
            y_for_importance = y_for_importance[:min_len]
            
            if len(test_X) > 0 and len(y_for_importance) > 0:
                feature_importance = model.calculate_feature_importance(test_X, y_for_importance)
                print("\n=== 特征重要性分析 ===")
                for i, (feature, importance) in enumerate(zip(features, feature_importance)):
                    print(f"{feature}: {importance:.4f}")
                print("========================\n")
        
        model.eval()
        with torch.no_grad():
            predictions = model(last_sequence)[0].numpy()
            next_day_close = predictions[0] * y_close_std + y_close_mean
            next_day_high = predictions[1] * y_high_std + y_high_mean
            next_day_low = predictions[2] * y_low_std + y_low_mean
        
        # 保存训练好的模型
        import os
        model_dir = './models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存模型状态字典和相关参数
        model_save_path = os.path.join(model_dir, f'{self.symbol}_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': len(features),
            'features': features,
            'scaler_params': {
                'X_mean': X_mean if 'X_mean' in locals() else None,
                'X_std': X_std if 'X_std' in locals() else None,
                'y_close_mean': y_close_mean,
                'y_close_std': y_close_std,
                'y_high_mean': y_high_mean,
                'y_high_std': y_high_std,
                'y_low_mean': y_low_mean,
                'y_low_std': y_low_std
            },
            'seq_length': seq_length,
            'model_metrics': {
                'mse': mse,
                'r2': r2
            }
        }, model_save_path)
        
        print(f"模型已保存到: {model_save_path}")
        
        # 在predict_next_day方法的返回部分（大约第620-630行）
        return {
            'next_day_close': next_day_close,
            'next_day_high': next_day_high,
            'next_day_low': next_day_low,
            'date': next_trade_date,
            'confidence': float(r2) if r2 > 0 else 0.0,
            'model_performance': {
                'mse': float(mse),
                'r2': float(r2),
                'features_count': len(features),
                'features': features,
                'feature_importances': feature_importance if 'feature_importance' in locals() else None
            }
        }
        
        # 添加振幅特征
        # 1. 日振幅 = (最高价 - 最低价) / 昨日收盘价 * 100
        df['Amplitude'] = (df['High'] - df['Low']) / df['Close'].shift(1) * 100
        
        # 2. 5日平均振幅
        df['Amplitude_5d'] = df['Amplitude'].rolling(window=5).mean()
        
        # 3. 振幅相对值 = 当日振幅 / 5日平均振幅
        df['Amplitude_Ratio'] = df['Amplitude'] / df['Amplitude_5d']
        
        # 4. 振幅变化率
        df['Amplitude_Change'] = df['Amplitude'].pct_change()
        
        # 添加流通市值特征
        try:
            info = self.stock.info
            shares_outstanding = info.get('sharesOutstanding', None)
            market_cap = info.get('marketCap', None)
            
            if shares_outstanding and shares_outstanding > 0:
                # 1. 流通市值 = 股价 * 流通股本
                df['Market_Cap'] = df['Close'] * shares_outstanding
                
                # 2. 流通市值变化率
                df['Market_Cap_Change'] = df['Market_Cap'].pct_change()
                
                # 3. 流通市值相对值 = 当前市值 / 20日平均市值
                df['Market_Cap_20d'] = df['Market_Cap'].rolling(window=20).mean()
                df['Market_Cap_Ratio'] = df['Market_Cap'] / df['Market_Cap_20d']
                
                # 4. 市值波动率 = 20日市值标准差 / 20日平均市值
                df['Market_Cap_Volatility'] = df['Market_Cap'].rolling(window=20).std() / df['Market_Cap_20d']
                
            elif market_cap:
                # 如果无法获取流通股本，使用总市值作为替代
                df['Market_Cap'] = market_cap
                df['Market_Cap_Change'] = 0  # 静态值，变化率为0
                df['Market_Cap_Ratio'] = 1   # 相对值为1
                df['Market_Cap_Volatility'] = 0  # 波动率为0
            else:
                # 使用成交额作为市值的替代指标
                df['Market_Cap'] = df['Volume'] * df['Close']
                df['Market_Cap_Change'] = df['Market_Cap'].pct_change()
                df['Market_Cap_20d'] = df['Market_Cap'].rolling(window=20).mean()
                df['Market_Cap_Ratio'] = df['Market_Cap'] / df['Market_Cap_20d']
                df['Market_Cap_Volatility'] = df['Market_Cap'].rolling(window=20).std() / df['Market_Cap_20d']
                
        except Exception as e:
            print(f"计算流通市值时出错: {e}")
            # 使用成交额作为替代
            df['Market_Cap'] = df['Volume'] * df['Close']
            df['Market_Cap_Change'] = df['Market_Cap'].pct_change()
            df['Market_Cap_20d'] = df['Market_Cap'].rolling(window=20).mean()
            df['Market_Cap_Ratio'] = df['Market_Cap'] / df['Market_Cap_20d']
            df['Market_Cap_Volatility'] = df['Market_Cap'].rolling(window=20).std() / df['Market_Cap_20d']
        
        model.eval()
        with torch.no_grad():
            predictions = model(last_sequence)[0].numpy()
            next_day_close = predictions[0] * y_close_std + y_close_mean
            next_day_high = predictions[1] * y_high_std + y_high_mean
            next_day_low = predictions[2] * y_low_std + y_low_mean
        
        return {
            'close': next_day_close,
            'high': next_day_high,
            'low': next_day_low,
            'date': next_trade_date,
            'model': model,
            'metrics': {
                'mse': mse, 
                'r2': r2,
                'features': features
            }
        }

    def load_model(self, model_path=None):
        """加载已保存的模型"""
        if model_path is None:
            model_path = f'./models/{self.symbol}_model.pth'
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        try:
            # 加载模型
            checkpoint = torch.load(model_path)
            model = HybridModel(checkpoint['input_size'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"模型加载成功: {model_path}")
            print(f"模型性能 - MSE: {checkpoint['model_metrics']['mse']:.4f}, R²: {checkpoint['model_metrics']['r2']:.4f}")
            
            return {
                'model': model,
                'features': checkpoint['features'],
                'scaler_params': checkpoint['scaler_params'],
                'seq_length': checkpoint['seq_length'],
                'metrics': checkpoint['model_metrics']
            }
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None

    @staticmethod
    def get_hot_stocks_recommendation(market='CN', top_n=10, use_ai_pool=False, category='热点'):
        """
        获取热点股票推荐，基于多维度热度评分
        
        Args:
            market: 市场类型 ('CN'为中国市场，'US'为美国市场)
            top_n: 返回推荐股票数量
            use_ai_pool: 是否使用AI生成的股票池（已弃用，现在从本地CSV文件读取）
            category: 股票类别
        """
        try:
            # 从本地CSV文件获取股票池
            stock_pool = StockAnalyzer._get_stock_pool_from_csv(
                csv_path='dataset/A股代码名称_20250908.csv',
                market=market,
                category=category
            )
            
            if not stock_pool:
                # 如果CSV读取失败，使用默认股票池作为备用
                stock_pool = StockAnalyzer._get_default_stock_pool(market)
                print("警告：CSV文件读取失败，使用默认股票池")
            else:
                print(f"从CSV文件成功加载 {len(stock_pool)} 只股票")
            
            # 分析股票并计算热度评分
            recommendations = []
            
            for symbol in stock_pool:
                try:
                    # 获取股票历史数据（最近3个月）
                    stock_data = StockAnalyzer._get_stock_historical_data(symbol, period='3mo')
                    
                    if stock_data is None or stock_data.empty:
                        print(f"无法获取股票 {symbol} 的历史数据，跳过")
                        continue
                    
                    # 计算热度评分
                    heat_score = StockAnalyzer._calculate_heat_score(stock_data, symbol)
                    
                    # 生成推荐理由
                    recommendation_reason = StockAnalyzer._generate_recommendation_reason(heat_score, stock_data)
                    
                    recommendation = {
                        'symbol': symbol,
                        'name': StockAnalyzer._get_stock_name_from_csv(symbol),
                        'market_type': StockAnalyzer._get_market_type_from_symbol(symbol),
                        'category': category,
                        'heat_score': heat_score['total_score'],
                        'heat_details': heat_score,
                        'recommendation_reason': recommendation_reason,
                        'current_price': float(stock_data['Close'].iloc[-1]) if not stock_data.empty else 0,
                        'price_change_3m': StockAnalyzer._calculate_price_change(stock_data, '3mo')
                    }
                    recommendations.append(recommendation)
                    
                except Exception as e:
                    print(f"分析股票 {symbol} 时出错: {e}")
                    continue
            
            # 按热度评分排序并返回top_n
            recommendations.sort(key=lambda x: x['heat_score'], reverse=True)
            
            return {
                'success': True,
                'market': market,
                'category': category,
                'data_source': 'local_csv',
                'csv_file': 'A股代码名称_20250908.csv',
                'total_analyzed': len(recommendations),
                'recommendations': recommendations[:top_n],
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'scoring_method': '多维度热度评分（成交量30分+价格趋势25分+RSI20分+其他25分）'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"获取热点股票推荐失败: {str(e)}",
                'market': market,
                'recommendations': []
            }
    
    @staticmethod
    def _get_stock_historical_data(symbol, period='3mo'):
        """
        获取股票历史数据
        
        Args:
            symbol: 股票代码（如 000001.SZ）
            period: 时间周期（3mo=3个月）
            
        Returns:
            DataFrame: 股票历史数据
        """
        try:
            import yfinance as yf
            import pandas as pd
            
            # 获取股票数据
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                print(f"警告：无法获取 {symbol} 的历史数据")
                return None
            
            return hist_data
            
        except Exception as e:
            print(f"获取 {symbol} 历史数据失败: {e}")
            return None
    
    @staticmethod
    def _calculate_heat_score(stock_data, symbol):
        """
        计算股票热度评分（总分100分）
        
        Args:
            stock_data: 股票历史数据DataFrame
            symbol: 股票代码
            
        Returns:
            dict: 热度评分详情
        """
        try:
            import numpy as np
            import pandas as pd
            
            # 1. 成交量热度（30分）
            volume_score = StockAnalyzer._calculate_volume_heat(stock_data)
            
            # 2. 价格趋势热度（25分）
            trend_score = StockAnalyzer._calculate_trend_heat(stock_data)
            
            # 3. RSI热度（20分）
            rsi_score = StockAnalyzer._calculate_rsi_heat(stock_data)
            
            # 4. 技术指标热度（15分）
            technical_score = StockAnalyzer._calculate_technical_heat(stock_data)
            
            # 5. 波动率热度（10分）
            volatility_score = StockAnalyzer._calculate_volatility_heat(stock_data)
            
            # 计算总分
            total_score = volume_score + trend_score + rsi_score + technical_score + volatility_score
            
            return {
                'total_score': round(total_score, 2),
                'volume_heat': round(volume_score, 2),
                'trend_heat': round(trend_score, 2),
                'rsi_heat': round(rsi_score, 2),
                'technical_heat': round(technical_score, 2),
                'volatility_heat': round(volatility_score, 2),
                'symbol': symbol
            }
            
        except Exception as e:
            print(f"计算 {symbol} 热度评分失败: {e}")
            return {
                'total_score': 0,
                'volume_heat': 0,
                'trend_heat': 0,
                'rsi_heat': 0,
                'technical_heat': 0,
                'volatility_heat': 0,
                'symbol': symbol
            }
    
    @staticmethod
    def _calculate_volume_heat(stock_data):
        """
        计算成交量热度（30分）
        
        Args:
            stock_data: 股票历史数据
            
        Returns:
            float: 成交量热度评分
        """
        try:
            if 'Volume' not in stock_data.columns or stock_data['Volume'].empty:
                return 0
            
            # 计算近期成交量与历史平均成交量的比值
            recent_volume = stock_data['Volume'].tail(5).mean()  # 最近5天平均成交量
            historical_volume = stock_data['Volume'].mean()  # 历史平均成交量
            
            if historical_volume == 0:
                return 0
            
            volume_ratio = recent_volume / historical_volume
            
            # 成交量放大评分逻辑
            if volume_ratio >= 2.0:  # 成交量放大2倍以上
                score = 30
            elif volume_ratio >= 1.5:  # 成交量放大1.5倍以上
                score = 25
            elif volume_ratio >= 1.2:  # 成交量放大1.2倍以上
                score = 20
            elif volume_ratio >= 1.0:  # 成交量正常
                score = 15
            elif volume_ratio >= 0.8:  # 成交量略低
                score = 10
            else:  # 成交量过低
                score = 5
            
            return score
            
        except Exception as e:
            print(f"计算成交量热度失败: {e}")
            return 0
    
    @staticmethod
    def _calculate_trend_heat(stock_data):
        """
        计算价格趋势热度（25分）
        基于多头排列和中期上涨趋势
        
        Args:
            stock_data: 股票历史数据
            
        Returns:
            float: 趋势热度评分
        """
        try:
            if stock_data.empty or 'Close' not in stock_data.columns:
                return 0
            
            # 计算移动平均线
            stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
            stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
            stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['MA60'] = stock_data['Close'].rolling(window=60).mean()
            
            # 获取最新数据
            latest = stock_data.iloc[-1]
            
            score = 0
            
            # 1. 多头排列检查（15分）
            if (latest['Close'] > latest['MA5'] > latest['MA10'] > 
                latest['MA20'] > latest['MA60']):
                score += 15  # 完美多头排列
            elif (latest['Close'] > latest['MA5'] > latest['MA10'] > latest['MA20']):
                score += 12  # 短期多头排列
            elif (latest['Close'] > latest['MA5'] > latest['MA10']):
                score += 8   # 极短期多头排列
            elif latest['Close'] > latest['MA5']:
                score += 5   # 价格在5日线上方
            
            # 2. 中期上涨趋势检查（10分）
            if len(stock_data) >= 20:
                price_20_days_ago = stock_data['Close'].iloc[-20]
                current_price = latest['Close']
                
                price_change_pct = (current_price - price_20_days_ago) / price_20_days_ago * 100
                
                if price_change_pct >= 20:  # 20天涨幅超过20%
                    score += 10
                elif price_change_pct >= 10:  # 20天涨幅超过10%
                    score += 8
                elif price_change_pct >= 5:   # 20天涨幅超过5%
                    score += 6
                elif price_change_pct >= 0:   # 20天涨幅为正
                    score += 4
                else:  # 20天涨幅为负
                    score += 0
            
            return min(score, 25)  # 最高25分
            
        except Exception as e:
            print(f"计算趋势热度失败: {e}")
            return 0
    
    @staticmethod
    def _calculate_rsi_heat(stock_data):
        """
        计算RSI热度（20分）
        
        Args:
            stock_data: 股票历史数据
            
        Returns:
            float: RSI热度评分
        """
        try:
            if stock_data.empty or 'Close' not in stock_data.columns:
                return 0
            
            # 计算RSI
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            rsi = calculate_rsi(stock_data['Close'])
            
            if rsi.empty:
                return 0
            
            current_rsi = rsi.iloc[-1]
            
            # RSI评分逻辑
            if 50 <= current_rsi <= 70:  # RSI在健康上升区间
                score = 20
            elif 40 <= current_rsi < 50:  # RSI在中性偏强区间
                score = 15
            elif 30 <= current_rsi < 40:  # RSI在超卖反弹区间
                score = 18
            elif 70 < current_rsi <= 80:  # RSI在超买但仍有空间
                score = 12
            elif current_rsi > 80:  # RSI过度超买
                score = 5
            elif current_rsi < 30:  # RSI过度超卖
                score = 8
            else:
                score = 10
            
            return score
            
        except Exception as e:
            print(f"计算RSI热度失败: {e}")
            return 0
    
    @staticmethod
    def _calculate_technical_heat(stock_data):
        """
        计算技术指标热度（15分）
        包括MACD、布林带等
        
        Args:
            stock_data: 股票历史数据
            
        Returns:
            float: 技术指标热度评分
        """
        try:
            if stock_data.empty or 'Close' not in stock_data.columns:
                return 0
            
            score = 0
            
            # 1. MACD指标（8分）
            try:
                # 计算MACD
                exp1 = stock_data['Close'].ewm(span=12).mean()
                exp2 = stock_data['Close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                histogram = macd - signal
                
                if not histogram.empty:
                    current_hist = histogram.iloc[-1]
                    prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0
                    
                    if current_hist > 0 and current_hist > prev_hist:  # MACD金叉且向上
                        score += 8
                    elif current_hist > 0:  # MACD在零轴上方
                        score += 5
                    elif current_hist > prev_hist:  # MACD向上但在零轴下方
                        score += 3
            except:
                pass
            
            # 2. 布林带指标（7分）
            try:
                # 计算布林带
                ma20 = stock_data['Close'].rolling(window=20).mean()
                std20 = stock_data['Close'].rolling(window=20).std()
                upper_band = ma20 + (std20 * 2)
                lower_band = ma20 - (std20 * 2)
                
                current_price = stock_data['Close'].iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
                current_ma = ma20.iloc[-1]
                
                # 布林带位置评分
                if current_price > current_ma:  # 价格在中轨上方
                    if current_price < current_upper * 0.9:  # 未接近上轨
                        score += 7
                    else:  # 接近上轨
                        score += 4
                elif current_price > current_lower * 1.1:  # 价格在下轨上方但中轨下方
                    score += 5
                else:  # 价格接近或跌破下轨
                    score += 2
            except:
                pass
            
            return min(score, 15)  # 最高15分
            
        except Exception as e:
            print(f"计算技术指标热度失败: {e}")
            return 0
    
    @staticmethod
    def _calculate_volatility_heat(stock_data):
        """
        计算波动率热度（10分）
        
        Args:
            stock_data: 股票历史数据
            
        Returns:
            float: 波动率热度评分
        """
        try:
            if stock_data.empty or 'Close' not in stock_data.columns:
                return 0
            
            # 计算日收益率
            returns = stock_data['Close'].pct_change().dropna()
            
            if returns.empty:
                return 0
            
            # 计算波动率（标准差）
            volatility = returns.std() * 100  # 转换为百分比
            
            # 波动率评分逻辑（适度波动最佳）
            if 1.5 <= volatility <= 3.0:  # 适度波动
                score = 10
            elif 1.0 <= volatility < 1.5:  # 波动偏小
                score = 8
            elif 3.0 < volatility <= 4.5:  # 波动偏大但可接受
                score = 7
            elif 0.5 <= volatility < 1.0:  # 波动很小
                score = 6
            elif 4.5 < volatility <= 6.0:  # 波动较大
                score = 5
            else:  # 波动过大或过小
                score = 3
            
            return score
            
        except Exception as e:
            print(f"计算波动率热度失败: {e}")
            return 0
    
    @staticmethod
    def _generate_recommendation_reason(heat_score, stock_data):
        """
        生成推荐理由
        
        Args:
            heat_score: 热度评分详情
            stock_data: 股票历史数据
            
        Returns:
            str: 推荐理由
        """
        try:
            reasons = []
            
            # 根据各项评分生成理由
            if heat_score['volume_heat'] >= 20:
                reasons.append("成交量显著放大，市场关注度高")
            elif heat_score['volume_heat'] >= 15:
                reasons.append("成交量活跃，资金参与积极")
            
            if heat_score['trend_heat'] >= 20:
                reasons.append("多头排列明显，中期上涨趋势强劲")
            elif heat_score['trend_heat'] >= 15:
                reasons.append("技术形态良好，短期趋势向上")
            
            if heat_score['rsi_heat'] >= 18:
                reasons.append("RSI指标健康，买入时机较好")
            elif heat_score['rsi_heat'] >= 15:
                reasons.append("RSI指标稳定，技术面支撑")
            
            if heat_score['technical_heat'] >= 12:
                reasons.append("技术指标配合良好，多项指标共振")
            
            if heat_score['volatility_heat'] >= 8:
                reasons.append("波动率适中，风险收益比合理")
            
            # 综合评分理由
            total_score = heat_score['total_score']
            if total_score >= 80:
                reasons.append("综合热度评分优秀，强烈推荐")
            elif total_score >= 70:
                reasons.append("综合热度评分良好，值得关注")
            elif total_score >= 60:
                reasons.append("综合热度评分中等，可适量配置")
            else:
                reasons.append("综合热度评分一般，建议谨慎")
            
            return "；".join(reasons) if reasons else "技术面分析显示该股票具有一定投资价值"
            
        except Exception as e:
            print(f"生成推荐理由失败: {e}")
            return "基于多维度技术分析的推荐"
    
    @staticmethod
    def _calculate_price_change(stock_data, period):
        """
        计算价格变化百分比
        
        Args:
            stock_data: 股票历史数据
            period: 时间周期
            
        Returns:
            float: 价格变化百分比
        """
        try:
            if stock_data.empty or 'Close' not in stock_data.columns:
                return 0
            
            current_price = stock_data['Close'].iloc[-1]
            start_price = stock_data['Close'].iloc[0]
            
            if start_price == 0:
                return 0
            
            change_pct = (current_price - start_price) / start_price * 100
            return round(change_pct, 2)
            
        except Exception as e:
            print(f"计算价格变化失败: {e}")
            return 0
    
    @staticmethod
    def _get_stock_pool_from_csv(csv_path, market='CN', category='热点', max_stocks=50):
        """
        从CSV文件获取股票池
        
        Args:
            csv_path: CSV文件路径
            market: 市场类型
            category: 股票类别
            max_stocks: 最大股票数量
            
        Returns:
            list: 股票代码列表
        """
        try:
            import pandas as pd
            import random
            
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 确保必要的列存在
            if 'code' not in df.columns or 'name' not in df.columns:
                print("错误：CSV文件缺少必要的列（code, name）")
                return []
            
            # 根据市场类型筛选
            if market == 'CN':
                if '市场类型' in df.columns:
                    # 如果有市场类型列，可以进一步筛选
                    # 这里可以根据category进行更精细的筛选
                    pass
                
                # 转换为标准格式（添加后缀）
                stock_codes = []
                for _, row in df.iterrows():
                    code = str(row['code']).zfill(6)  # 确保6位数字
                    market_type = row.get('市场类型', '')
                    
                    # 根据市场类型添加后缀
                    if market_type == '深圳' or code.startswith(('000', '002', '003', '300')):
                        stock_codes.append(f"{code}.SZ")
                    elif market_type == '上海' or code.startswith(('600', '601', '603', '605', '688')):
                        stock_codes.append(f"{code}.SS")
                    else:
                        # 默认根据代码前缀判断
                        if code.startswith(('000', '002', '003', '300')):
                            stock_codes.append(f"{code}.SZ")
                        elif code.startswith(('600', '601', '603', '605', '688')):
                            stock_codes.append(f"{code}.SS")
                
                # 随机选择股票（模拟热点股票筛选）
                if len(stock_codes) > max_stocks:
                    stock_codes = random.sample(stock_codes, max_stocks)
                
                return stock_codes
            
            else:
                # 美国市场暂不支持
                print("暂不支持美国市场的CSV数据")
                return []
                
        except Exception as e:
            print(f"从CSV文件读取股票池失败: {e}")
            return []
    
    @staticmethod
    def _get_stock_name_from_csv(symbol, csv_path='f:\\ai_pj\\AI\\finance\\finance_bot\\dataset\\A股代码名称_20250908.csv'):
        """
        从CSV文件获取股票名称
        
        Args:
            symbol: 股票代码（如 000001.SZ）
            csv_path: CSV文件路径
            
        Returns:
            str: 股票名称
        """
        try:
            import pandas as pd
            
            # 提取纯数字代码
            code = symbol.split('.')[0]
            
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 查找对应的股票名称
            matching_row = df[df['code'].astype(str).str.zfill(6) == code]
            
            if not matching_row.empty:
                return matching_row.iloc[0]['name']
            else:
                return f"股票{code}"
                
        except Exception as e:
            print(f"获取股票名称失败: {e}")
            return f"股票{symbol}"
    
    @staticmethod
    def _get_market_type_from_symbol(symbol):
        """
        根据股票代码判断市场类型
        
        Args:
            symbol: 股票代码（如 000001.SZ）
            
        Returns:
            str: 市场类型
        """
        if symbol.endswith('.SZ'):
            return '深圳'
        elif symbol.endswith('.SS'):
            return '上海'
        else:
            return '未知'
    
    @staticmethod
    def get_ai_generated_stock_pool(market='CN', category='热点', llm_client=None):
        """
        使用大模型生成股票池
        
        Args:
            market: 市场类型，'CN'为中国市场，'US'为美国市场
            category: 股票类别，如'热点'、'成长'、'价值'、'科技'等
            llm_client: 大模型客户端实例
            
        Returns:
            list: 股票代码列表
        """
        try:
            if market == 'CN':
                prompt = f"""
                请根据当前A股市场情况，推荐15-20只{category}股票。
                
                要求：
                1. 股票代码格式：深圳股票用.SZ后缀，上海股票用.SS后缀
                2. 涵盖不同行业和板块
                3. 考虑以下因素：
                   - 近期市场热点和政策导向
                   - 公司基本面和成长性
                   - 技术面表现和资金关注度
                   - 行业景气度和发展前景
                
                请直接返回股票代码列表，每行一个，格式如：000001.SZ
                不要包含其他解释文字。
                """
            else:
                prompt = f"""
                Please recommend 15-20 {category} stocks from US market based on current market conditions.
                
                Requirements:
                1. Stock symbols in standard format (e.g., AAPL, MSFT)
                2. Cover different sectors and industries
                3. Consider factors:
                   - Recent market trends and policy directions
                   - Company fundamentals and growth potential
                   - Technical performance and institutional interest
                   - Industry outlook and development prospects
                
                Please return only the stock symbols, one per line.
                No additional explanations needed.
                """
            
            if llm_client:
                response = llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",  # 或使用其他模型
                    messages=[
                        {"role": "system", "content": "你是一位专业的股票分析师，具有丰富的市场经验。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                
                stock_list = response.choices[0].message.content.strip().split('\n')
                # 清理和验证股票代码
                cleaned_stocks = []
                for stock in stock_list:
                    stock = stock.strip()
                    if stock and (stock.endswith('.SZ') or stock.endswith('.SS') or 
                                (market == 'US' and len(stock) <= 5 and stock.isalpha())):
                        cleaned_stocks.append(stock)
                
                return cleaned_stocks[:20]  # 限制最多20只
            else:
                # 如果没有LLM客户端，返回默认股票池
                return StockAnalyzer._get_default_stock_pool(market)
                
        except Exception as e:
            print(f"大模型生成股票池失败: {e}")
            # 降级到默认股票池
            return StockAnalyzer._get_default_stock_pool(market)
    
    @staticmethod
    def _get_default_stock_pool(market='CN'):
        """默认股票池，作为大模型失败时的备选方案"""
        if market == 'CN':
            return [
                '000001.SZ', '000002.SZ', '000858.SZ', '000876.SZ',
                '002415.SZ', '002594.SZ', '300059.SZ', '300750.SZ',
                '600036.SS', '600519.SS', '600887.SS', '000725.SZ',
                '002230.SZ', '300014.SZ', '600276.SS'
            ]
        else:
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'META', 'NVDA', 'NFLX', 'AMD', 'BABA',
                'CRM', 'ORCL', 'ADBE', 'PYPL', 'INTC'
            ]

    @staticmethod
    def get_all_a_share_stocks():
        """获取A股所有股票代码和公司名称
        
        Returns:
            dict: 包含股票代码、公司名称、市场等信息的字典
                {
                    'success': bool,
                    'data': [
                        {
                            'code': str,      # 股票代码 (如: '000001.SZ')
                            'name': str,      # 公司名称 (如: '平安银行')
                            'market': str,    # 市场 ('SZ'/'SS')
                            'industry': str   # 行业分类
                        }
                    ],
                    'total_count': int,
                    'message': str
                }
        """
        try:
            import pandas as pd
            import os
            
            # 本地CSV文件路径
            csv_file_path = r'f:\ai_pj\AI\finance\finance_bot\dataset\A股代码名称_20250908.csv'
            
            # 检查文件是否存在
            if not os.path.exists(csv_file_path):
                return StockAnalyzer._get_fallback_a_share_list()
            
            # 读取CSV文件
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            
            if df.empty:
                return {
                    'success': False,
                    'data': [],
                    'total_count': 0,
                    'message': 'CSV文件为空'
                }
            
            # 处理数据格式
            stocks_data = []
            for _, row in df.iterrows():
                code = str(row.get('code', '')).strip()
                name = str(row.get('name', '')).strip()
                market_type = str(row.get('市场类型', '')).strip()
                
                # 跳过无效数据
                if not code or not name:
                    continue
                
                # 根据市场类型添加后缀
                if market_type == '深圳':
                    market_code = f"{code}.SZ"
                    market = 'SZ'
                elif market_type == '上海':
                    market_code = f"{code}.SS"
                    market = 'SS'
                else:
                    # 根据代码开头判断市场
                    if code.startswith(('0', '3')):
                        market_code = f"{code}.SZ"
                        market = 'SZ'
                    elif code.startswith('6'):
                        market_code = f"{code}.SS"
                        market = 'SS'
                    else:
                        market_code = code
                        market = 'Unknown'
                
                stocks_data.append({
                    'code': market_code,
                    'name': name,
                    'market': market,
                    'market_type': market_type,
                    'industry': '待获取'  # 可以后续通过其他接口获取行业信息
                })
            
            return {
                'success': True,
                'data': stocks_data,
                'total_count': len(stocks_data),
                'message': f'成功从本地CSV文件获取{len(stocks_data)}只A股股票信息'
            }
            
        except Exception as e:
            # 发生错误时使用备用方案
            return {
                'success': False,
                'data': [],
                'total_count': 0,
                'message': f'从CSV文件获取A股数据时发生错误: {str(e)}，请检查文件路径和格式'
            }
    
    @staticmethod
    def _get_fallback_a_share_list():
        """备用A股股票列表（主要的大盘股和知名股票）"""
        fallback_stocks = [
            {'code': '000001.SZ', 'name': '平安银行', 'market': 'SZ', 'industry': '银行'},
            {'code': '000002.SZ', 'name': '万科A', 'market': 'SZ', 'industry': '房地产'},
            {'code': '000858.SZ', 'name': '五粮液', 'market': 'SZ', 'industry': '食品饮料'},
            {'code': '000876.SZ', 'name': '新希望', 'market': 'SZ', 'industry': '农林牧渔'},
            {'code': '002415.SZ', 'name': '海康威视', 'market': 'SZ', 'industry': '电子'},
            {'code': '002594.SZ', 'name': '比亚迪', 'market': 'SZ', 'industry': '汽车'},
            {'code': '300059.SZ', 'name': '东方财富', 'market': 'SZ', 'industry': '非银金融'},
            {'code': '300750.SZ', 'name': '宁德时代', 'market': 'SZ', 'industry': '电气设备'},
            {'code': '600036.SS', 'name': '招商银行', 'market': 'SS', 'industry': '银行'},
            {'code': '600519.SS', 'name': '贵州茅台', 'market': 'SS', 'industry': '食品饮料'},
            {'code': '600887.SS', 'name': '伊利股份', 'market': 'SS', 'industry': '食品饮料'},
            {'code': '000725.SZ', 'name': '京东方A', 'market': 'SZ', 'industry': '电子'},
            {'code': '002230.SZ', 'name': '科大讯飞', 'market': 'SZ', 'industry': '计算机'},
            {'code': '300014.SZ', 'name': '亿纬锂能', 'market': 'SZ', 'industry': '电气设备'},
            {'code': '600276.SS', 'name': '恒瑞医药', 'market': 'SS', 'industry': '医药生物'},
            {'code': '000063.SZ', 'name': '中兴通讯', 'market': 'SZ', 'industry': '通信'},
            {'code': '002142.SZ', 'name': '宁波银行', 'market': 'SZ', 'industry': '银行'},
            {'code': '600030.SS', 'name': '中信证券', 'market': 'SS', 'industry': '非银金融'},
            {'code': '000858.SZ', 'name': '五粮液', 'market': 'SZ', 'industry': '食品饮料'},
            {'code': '601318.SS', 'name': '中国平安', 'market': 'SS', 'industry': '非银金融'}
        ]
        
        return {
            'success': True,
            'data': fallback_stocks,
            'total_count': len(fallback_stocks),
            'message': f'使用备用股票列表，包含{len(fallback_stocks)}只主要A股股票'
        }
    
    @staticmethod
    def save_stocks_to_file(stocks_data, file_path=None, file_format='csv'):
        """将股票数据保存到文件
        
        Args:
            stocks_data (dict): 股票数据字典
            file_path (str): 文件保存路径，如果为None则使用默认路径
            file_format (str): 文件格式，支持 'csv', 'json', 'excel'
            
        Returns:
            dict: 保存结果
        """
        try:
            import pandas as pd
            import json
            import os
            
            if not stocks_data.get('success', False):
                return {
                    'success': False,
                    'message': '股票数据无效，无法保存'
                }
            
            # 设置默认文件路径
            if file_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if file_format == 'csv':
                    file_path = f'a_share_stocks_{timestamp}.csv'
                elif file_format == 'json':
                    file_path = f'a_share_stocks_{timestamp}.json'
                elif file_format == 'excel':
                    file_path = f'a_share_stocks_{timestamp}.xlsx'
            
            # 创建DataFrame
            df = pd.DataFrame(stocks_data['data'])
            
            # 根据格式保存文件
            if file_format.lower() == 'csv':
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif file_format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(stocks_data, f, ensure_ascii=False, indent=2)
            elif file_format.lower() == 'excel':
                df.to_excel(file_path, index=False, engine='openpyxl')
            else:
                return {
                    'success': False,
                    'message': f'不支持的文件格式: {file_format}'
                }
            
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            
            return {
                'success': True,
                'file_path': os.path.abspath(file_path),
                'file_size': file_size,
                'total_records': len(stocks_data['data']),
                'message': f'成功保存{len(stocks_data["data"])}条股票数据到 {file_path}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'保存文件时发生错误: {str(e)}'
            }
    
    @staticmethod
    def import_and_save_a_share_stocks(file_format='csv', file_path=None):
        """导入A股股票数据并保存到文件的一体化方法
        
        Args:
            file_format (str): 文件格式，支持 'csv', 'json', 'excel'
            file_path (str): 文件保存路径，如果为None则使用默认路径
            
        Returns:
            dict: 导入和保存的结果
        """
        print("正在获取A股股票数据...")
        
        # 获取股票数据
        stocks_data = StockAnalyzer.get_all_a_share_stocks()
        
        if not stocks_data['success']:
            return stocks_data
        
        print(f"成功获取 {stocks_data['total_count']} 只股票数据")
        print("正在保存到文件...")
        
        # 保存到文件
        save_result = StockAnalyzer.save_stocks_to_file(
            stocks_data, file_path, file_format
        )
        
        if save_result['success']:
            print(f"文件已保存到: {save_result['file_path']}")
            print(f"文件大小: {save_result['file_size']} 字节")
            
            # 合并结果
            return {
                'success': True,
                'data_info': {
                    'total_stocks': stocks_data['total_count'],
                    'message': stocks_data['message']
                },
                'file_info': {
                    'file_path': save_result['file_path'],
                    'file_size': save_result['file_size'],
                    'format': file_format
                },
                'message': f'成功导入并保存{stocks_data["total_count"]}只A股股票数据'
            }
        else:
            return save_result
    
    @staticmethod
    def search_stocks_by_name(name_keyword):
        """根据公司名称关键词搜索股票
        
        Args:
            name_keyword (str): 公司名称关键词
            
        Returns:
            dict: 搜索结果
        """
        try:
            all_stocks = StockAnalyzer.get_all_a_share_stocks()
            if not all_stocks['success']:
                return all_stocks
            
            # 搜索匹配的股票
            matched_stocks = []
            for stock in all_stocks['data']:
                if name_keyword.lower() in stock['name'].lower():
                    matched_stocks.append(stock)
            
            return {
                'success': True,
                'data': matched_stocks,
                'total_count': len(matched_stocks),
                'message': f'找到{len(matched_stocks)}只包含"{name_keyword}"的股票'
            }
            
        except Exception as e:
            return {
                'success': False,
                'data': [],
                'total_count': 0,
                'message': f'搜索股票时发生错误: {str(e)}'
            }

