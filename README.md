EMNLP 2022 Demo paper "Automatic Comment Generation for Chinese Student Narrative Essays"

## 数据处理
下载[链接](https://cloud.tsinghua.edu.cn/d/5a8e71813fa24cab8c18/)中的train_clean.json,val_clean.json,test_clean.json,cn_stopwords.txt放t5/data目录下
然后执行以下代码：
```
cd t5/data
python pro.py
cd ../../bert_correction
python create_data.py
```

## 模型训练

### t5生成
```
cd t5
bash finetune_base.sh
```

### bert纠错
```
cd bert_correction
bash run.sh
```

## 生成
### 生成关键词
首先用t5从essay生成关键词（需要修改gen.sh中的checkpoint路径）：
```
cd t5
bash gen.sh 0 
```

### 修改关键词
首先用bert修改关键词（需要修改infer.py中的数据路径）：
```
cd bert_correction
python infer.py
```

### 生成最终评论
最后用t5从essay+关键词生成评论（需要修改gen.sh中的checkpoint路径）：
```
cd t5
bash gen.sh 1
```

