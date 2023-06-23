# CILLMs —— Chinese Internet Large Language Models

# Evaluated Datase

## Sentiment Detection

| 数据集 | 解释 | 来源 | 链接 |
|:-------------:|:-----:|:-----:|:-----:|
| SMP2020-EWECT | SMP2020微博情绪分类评测 | 中国中文信息学会社会媒体处理专业委员会（CIPS-SMP） | https://smp2020ewect.github.io |
| SMP2019_ECISA	| SMP2019中文隐式情感分析评测 | 中国中文信息学会社会媒体处理专委会 | http://www.cips-smp.org/smp_data/5 |
| DDmkTCCorpus	| 弹幕情感标注语料 | TinkTalks | https://github.com/TinyTalks/DDmkTCCorpus |

### NER

| 数据集 | 解释 | 来源 | 链接 |
|:-------------:|:-----:|:-----:|:-----:|
| CLUENER2020 | 清华大学开源的文本分类数据集THUCNEWS,进行筛选过滤、实体标注生成的 | 清华大学 | https://github.com/CLUEbenchmark/CLUENER2020 |


# Chinese Internet Large Language Models

## BERT

## RoBERTa

chinese_danmaku_roberta
- This model is a fine-tuned version of uer/chinese_roberta_L-8_H-512 on an Danmaku Corpus(2000k raw data) dataset.

| Mask Accuracy | Link |
|:-------------:|:-----:|
| 0.7780 | https://huggingface.co/WUJUNCHAO/chinese_danmaku_roberta |

| Dataset | Accuracy | Precision | Recall | F1 |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| DDmkTCECT | 0.89 | 0.86 | 0.88 | 0.87 |
  


## T5
