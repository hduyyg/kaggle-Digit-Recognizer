# kaggle-Digit-Recognizer

## introduction

kaggle入门题目，输入数据已经处理成向量并与标签对应，据此判断新向量对应的数字。

项目基本思路：

1. 在solutions中存储不同的解法，在所有解法单独实现完成之后，最终结果采用所有解法的结果来做交叉验证之后的答案，即对于某个向量的预测，采用所有解法中出现次数最多的作为结果。
2. main.py作为项目的入口，所有程序的调用都采用 `python main.py -* *  `的形式来调用，主要是为了统一log等的设置

## milestone

1. 2018-03-08 score=0.97342

   knn-采用0.5倍缩放之后，只设置n_neighbors=3，其余保持默认。

2. 2018-03-03 score=0.96357

   knn-采用二值化之后的数据，n_neighbors=3，algorithm、weights选择默认。

2. 开始前 score=0.96857

   knn-采用原始数据，n_neighbors=3，algorithm、weights选择默认

## schedule

1.  SVM base 版本

    deadline：2018-03-09

2.  深度学习 base 版本

    deadline：2018-03-10

## reference

1. <a href="http://blog.csdn.net/Dinosoft/article/details/50734539">[kaggle实战] Digit Recognizer -- 从KNN,LR,SVM,RF到深度学习</a>

2. <a href="http://blog.csdn.net/firethelife/article/details/51191530">初识Kaggle：手写体数字识别</a>

   ps : 数据二值化之后，采用knn算法

3. <a href="http://blog.csdn.net/buwei0239/article/details/78769985">kaggle-手写字体识别</a>

   ps : 深度学习-keras，达到0.99+

4. <a href="http://blog.csdn.net/u013691510/article/details/43195227">Kaggle项目实战1——Digit Recognizer——排名Top10%</a>

5. <a href="http://blog.csdn.net/laozhaokun/article/details/42749233">Kaggle竞赛题目之——Digit Recognizer</a>

   这篇文章中有对数据进行分析，将矩阵以图片形式显示

6. <a href="http://blog.csdn.net/memoryjdch/article/details/75220498">Kaggle学习之路(二) —— Digit Recognizer之问题分析</a>
