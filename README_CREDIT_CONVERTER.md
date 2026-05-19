# 征信报告JSON转文本转换工具

## 功能

将征信报告JSON文件转换为英文键值对格式的文本串，用于BERT预训练和微调。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 处理单个文件

```bash
python convert.py --single-file moc_data/CrisPbc.json --output output.txt
```

### 批量处理目录

```bash
python convert.py --input /path/to/json/files --output /path/to/output
```

### 参数说明

- `--input, -i`: 输入目录（包含JSON文件）
- `--output, -o`: 输出目录（存放转换后的txt文件）
- `--field-dict`: 字段字典xlsx文件路径（默认：moc_data/个人征信DB表结构字典.xlsx）
- `--code-value`: 码值表xlsx文件路径（默认：moc_data/个人征信码值表.xlsx）
- `--workers`: 并行工作进程数（默认：4）
- `--single-file`: 处理单个文件

## 输出格式

使用special tokens标记结构：

```
[CLS] [HDR] tran_date=20250124 report_time=2019-02-26 [SEP] [PERS] gender=male birthday=1981-08-15 [SEP] [ACCT] account_type=non_revolving [SEP] [PAD] ... [PAD]
```

## 排除字段

以下敏感字段会被自动排除：

- 姓名、证件号码、手机号、地址、邮箱
- 配偶信息
- 机构名称
