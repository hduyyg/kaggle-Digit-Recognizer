这里的内容为项目成员的日报，内容主要包括成员在项目中做的事情、学到的东西等等。

# 2018-03-04

## hduyyg

### 1. 已完成任务

1. 完成项目基础结构设计：

   * main.py作为项目入口

     已经实现：

     1. 将命令行参数args传递给python可执行文件的main入口，来运行程序

   * functions.py记录常用函数

     已经实现：

     1. 将预测结果的numpy数组转化为符合题目要求的csv文件并存储在data

   * pre_process.py记录数据预处理函数

     已经实现的预处理包括：

     1. 将csv数据读取并以npy形式存储在data
     2. 将矩阵数值二值化

   * get_args.py读取命令行参数并配置loggging

     参考资料：

     1. [argparse - 命令行选项与参数解析（译）](http://blog.xiayf.cn/2013/03/30/argparse/)

   * daily_report.py记录成员日报

   * solutions文件夹存储solution

   * data文件夹存储所有数据

2. 完成knn的base版本-solutions/knn.py

   采用二值化之后的数据，n_neighbors=3，algorithm、weights选择默认，score=0.96357

### 2.下一步任务计划

1. 设计一个ML分类器的通用框架类，主要功能如下：

   - cross_validate()：交叉验证来确定分类器对应的最佳参数，得到一个最佳分类器
   - get_result():得到测试数据的预测值，用于提交kaggle

2. 将solutions的knn实现改为上述通用框架类的类结构

3. 熟悉、了解sklear.neighbors中的knn分类器、knn回归等

   - 学习knn算法的KdTree实现原理

     参考资料：<a href="http://www.cnblogs.com/pinard/p/6061661.html">[K近邻法(KNN)原理小结](http://www.cnblogs.com/pinard/p/6061661.html)</a>

   - 学习knn算法的BallTree实现原理

     - 参考资料：<a href="http://www.cnblogs.com/pinard/p/6061661.html">[K近邻法(KNN)原理小结](http://www.cnblogs.com/pinard/p/6061661.html)</a>

### 3.随笔

1. KNN分类器使用说明：

   * [官方文档](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
   * [scikit-learn K近邻法类库使用小结](http://www.cnblogs.com/pinard/p/6065607.html)
2. [机器学习中的归一化方法](http://blog.csdn.net/dulingtingzi/article/details/51365545)