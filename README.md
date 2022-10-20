# 命名实体识别
## 1.主要环境依赖
```shell
tensorflow==1.14.0
keras==2.3.1
keras-contrib==2.0.8(直接pip安装好像跑不起来，可以参考下面说明)
```
说明：`keras-contrib`可以通过项目自带的keras-contrib文件安装，进入keras-contrib，执行`python setup.py install`

## 2.数据集
人民日报命名实体数据集

## 3.实验结果
|模型|accuracy|备注|
| :---: | :---: | :---: | 
|BiLSTM+CRF(tensorflow版)|0.9265|ChineseNER|
|BiLSTM+CRF(bert4keras版)|0.77634|bilstm_crf/bilstm_crf_bert4keras.py|
|BiLSTM+CRF(keras-contrib版)|0.8678|bilstm_crf/bilstm_crf_contrib.py|
|Bert+CRF|0.9621||
|Bert+GlobalPoint|0.9632||

疑问1：BiLSTM+CRF(keras-contrib版) 训练时，loss基本没怎么下降；  
疑问2：不同版本的BiLSTM+CRF差距有点大，没找到原因所在