---
license: Apache License 2.0
text:
  table-question-answering:
    language:
      - zh
  question-answering:
    language:
      - zh
tags:
  - Qianwen
  - Bosera
---
<p align="center">
  <img src="./img/1.png" alt="BOSERA BIG CHALLENGE" width="50%">
</p>

## 数据集描述
赛事主办方提供三类数据。一个是10张数据表，一个是招股说明书，以及将招股说明书pdf解析后的txt文件。 

####  10张表，用sqlite存储。选手可自行替换为其他db。区间为2019年至2021年
- 基金基本信息
- 基金股票持仓明细
- 基金债券持仓明细
- 基金可转债持仓明细
- 基金日行情表
- A股票日行情表
- 港股票日行情表
- A股公司行业划分表
- 基金规模变动表
- 基金份额持有人结构
####  招股说明书
- 80份招股说明书
## 数据集的格式和结构

#### 博金杯比赛数据.db
- 大小：1.46g
- 文件格式：db文件
- 文件数量：1

#### 招股说明书 pdf源文件
- 大小：527MB
- 文件格式：pdf文件
- 文件数量：80

#### 招股说明书 pdf解析后的txt文件
- 大小：44MB
- 文件格式：txt文件
- 文件数量：80

#### 初赛问题
- 文件名：question.json


## 数据集的格式和结构

### 数据集加载方式
#### git Clone with HTTP
```bash 
# 要求安装 git lfs
git clone https://www.modelscope.cn/datasets/BJQW14B/bs_challenge_financial_14b_dataset.git
```

#### 读取问题文件
```python
import jsonlines

def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content

question = read_jsonl('./question.json')
```


## 数据集版权信息
数据集已经开源，license为Apache License 2.0，如有违反相关条款，随时联系删除。
