### 一、医疗预测花费

#### （一）随机森林

py文件名为'RF.py'，将其与'train1.csv'、'test_sample.csv'放到同一文件运行，结果输出至'submission1.csv'。用python实现，需要用到的包：

| 名字         | 版本   |
| ------------ | ------ |
| scikit-learn | 0.23.1 |
| csv          |        |

函数有$convert$进行$one-hot$编码；$convert\_train$，读取'train1.csv'数据作为训练集；$convert\_test$读取'test_sample.csv'作为测试集；$print\_result$打印结果到'submission1.csv'。

主要函数为$build\_rf$，参数为$(n\_estimators, max\_depth, min\_samples\_split, min\_samples\_leaf)$，可以调参。$max\_depth=0$说明为默认值：无。

#### （二）AdaBoost回归器

py文件名为'Ada.py'，将其与'train1.csv'、'test_sample.csv'放到同一文件运行，结果输出至'submission1.csv'。用python实现，需要用到的包：

| 名字         | 版本   |
| ------------ | ------ |
| scikit-learn | 0.23.1 |
| csv          |        |

'Ada.py'与'RF.py'类似，只有主要函数不同，$build\_ada$，参数为$(n\_estimators, learning\_rate, loss)$，可以调参。

#### （三）GradientBoosting回归器

py文件名为'GD.py'，将其与'train1.csv'、'test_sample.csv'放到同一文件运行，结果输出至'submission1.csv'。用python实现，需要用到的包：

| 名字         | 版本   |
| ------------ | ------ |
| scikit-learn | 0.23.1 |
| csv          |        |

'GD.py'与'RF.py'类似，只有主要函数不同，$build\_gd$，参数为$(n\_estimators, learning\_rate, min\_samples\_split, min\_samples\_leaf, max\_depth)$，可以调参。

### 二、IMDB情感判断

py文件名为'LSTM.py'，将其与'train2.csv'、'test_data.csv'、'submission.csv'放到同一文件运行，结果输出至'submission2.csv'。用python实现，需要用到的包：

| 名字                | 版本   |
| ------------------- | ------ |
| keras               | 2.4.3  |
| keras-preprocessing | 1.1.1  |
| nltk                | 3.5    |
| numpy               | 1.18.5 |
| csv                 |        |

此外还要下载nltk_data语料库

主要函数$build\_lstm$，为构建$LSTM$,参数配置为（batch_size, epochs, hidden_feature, dropout），可以调参。

其他的还有$convert\_train,convert\_test$分别是从$'train2.csv'$导入训练集和$'test\_data.csv'$导入测试集，运行时需要把LSTM.py、'train2.csv'和'test_data.csv'放入同一文件夹下。

最后$print\_result$为打印结果的函数，函数按照$'submission.csv'$的格式把结果输出到$'submission2.csv'$中，运行时需将'LSTM.py'和'submission.csv'放入同一文件夹。