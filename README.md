# kaggle-Digit-Recognizer

## introduction

​	kaggle入门题目，输入数据已经处理成向量并与标签对应，据此判断新向量对应的数字。

## development

1. 数据清洗
   1. 二值化
   2. 降维（去掉多余的无效信息）

## milestone

1. 2018-03-03 knn——0.96857

## reference

1. <a href="http://blog.csdn.net/Dinosoft/article/details/50734539">[kaggle实战] Digit Recognizer -- 从KNN,LR,SVM,RF到深度学习</a>

2. <a href="http://blog.csdn.net/firethelife/article/details/51191530">初识Kaggle：手写体数字识别</a>

   ps : 数据二值化之后，采用knn算法

3. <a href="http://blog.csdn.net/buwei0239/article/details/78769985">kaggle-手写字体识别</a>

   ps : 深度学习-keras，达到0.99+

4. <a href="http://blog.csdn.net/u013691510/article/details/43195227">Kaggle项目实战1——Digit Recognizer——排名Top10%</a>

5. <a href="http://blog.csdn.net/laozhaokun/article/details/42749233">Kaggle竞赛题目之——Digit Recognizer</a>

6. <a href="http://blog.csdn.net/memoryjdch/article/details/75220498">Kaggle学习之路(二) —— Digit Recognizer之问题分析</a>

## schedule

1. 2018-03-03
   1. 完成工作：
      * 完成代码主体框架设计，包括：
        * 使用python-argparse，读取命令行参数
        * 配置logging，root log配置，未设计单独的logger
        * functions函数存储常用函数
        * pre_process预处理数据，包括将csv数据读入处理之后以npy文件存储、矩阵的二值化
        * main.py 作为项目入口，所有的代码运行都通过`python main -* *`的形式来调用
      * 完成knn的base版本
   2. 下一步工作：
      * 设计一个ML分类器的通用框架类，主要功能如下：
        * cross_validate()：交叉验证来确定分类器对应的最佳参数，得到一个最佳分类器
        * get_result():得到测试数据的预测值，用于提交kaggle
      * 熟悉、了解sklear.neighbors中的knn分类器、knn回归等