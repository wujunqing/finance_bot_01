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
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None
            checkpoint = torch.load(model_path)
            model = HybridModel(checkpoint['input_size'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

