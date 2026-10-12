# 参考文献

作业题目给出的阅读清单，共 11 篇。这里只给**引用与链接**，不转载原文文件——各篇版权归原作者所有。

按对本次实验的作用重新分组。编号沿用题目原文，方便对照。

## 核心对标：用 GPR 预测股票的先例

题目要求"采用国外学生们研究过的几种股票，和他们比试一下"，这两篇就是直接的对标对象。

**[5] M. T. Farrell, A. Correa.** *Gaussian Process Regression Models for Predicting Stock Trends.* MIT 6.867 课程项目, 2007.
→ <http://web.media.mit.edu/~mtf/6867/6867Final/6867finalproject.pdf>
用协方差（核 / Gram）矩阵把股价建模为时间的函数，预测个股次日涨跌及幅度。实验取 8 支股票、10 年数据，同时跟踪**价格**与**涨跌方向**两个目标。本次实验的 DA（方向准确率）指标直接承接这一思路。

**[6]** *Long-term Stock Market Forecasting using Gaussian Processes.* UBC CPSC 540 课程项目.
→ <https://www.cs.ubc.ca/~nando/540-2013/projects/p5.pdf>
沿用 Chapados & Bengio (2007) 的框架做长期预测。关键做法是把**每个交易年视为独立输入变量**，而不是把整段历史当作单一连续序列——避免用一条序列代表全部历史。取 NASDAQ 三支股票（HPQ、YHOO、SBUX），2002–2011 日线数据。

> 该文正文匿名（投稿评审格式），作者信息据 PDF 元数据为 Khaled Alutaibi。

## 理论根基：非平稳性

**[11] C. J. Paciorek.** *Nonstationary Gaussian Processes for Regression and Spatial Modelling.* 博士论文, Carnegie Mellon University, Department of Statistics, 2003.
→ <https://www.stat.berkeley.edu/~paciorek/diss/paciorek-thesis.pdf>
把 Higdon 等人的卷积核方法推广成一类**非平稳协方差函数**，并证明它们保留原平稳相关函数的光滑性（样本轨道可微性）。本次实验的核心出发点——"标准 GPR 的平稳性假设在结构转折期失效"——正是沿着这篇论文的方向。

## 领域基石与教材

**[1] C. K. I. Williams, C. E. Rasmussen.** "Gaussian Processes for Regression." *Advances in Neural Information Processing Systems 8* (NIPS 1995), pp. 514–520. MIT Press, 1996.
GPR 的开山之作。指出对权重的简单先验会诱导出函数空间上复杂的先验，因此转向**直接在函数空间上放置高斯过程先验**，从而在超参数固定时获得精确的贝叶斯预测。

**[2] M. Ebden.** "Gaussian Processes for Regression: A Quick Introduction." University of Oxford, 2008.
→ <http://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf>
最短的入门材料。从平方指数核 `k(x,x') = σ_f² exp(−(x−x')²/2l²)` 出发，把噪声折进核矩阵，推导预测式 `E(y*) = K*(K + σ_n²I)⁻¹y`，并点明 `O(n³)` 的矩阵求逆代价——这正是本次实验"计算效率"一节的理论依据。

**[10] C. B. Do** (updated by H. Lee). "Gaussian Processes." Stanford CS229 讲义, 2008.
从多元高斯、贝叶斯线性回归一路推到 GP 回归的预测方程，衔接自然。

## 直观材料

**[3] H. M. Wallach.** *Introduction to Gaussian Process Regression.* 2005.
→ <http://www.inference.org.uk/hmw26/papers/gp_intro.pdf>
从权重空间视角（贝叶斯线性回归、基函数、核技巧）对照函数空间视角讲 GP，两组视角的对应关系讲得很清楚。

> 作业题目中写作 "H.M.Wallace"，作者实为 Hanna M. Wallach。

**[4] Z. Ghahramani.** "A Tutorial on Gaussian Processes (or why I don't use SVMs)." MLSS 2011.
讲稿形式，覆盖 GP 定义、核函数及其可解释参数（信号方差、偏置方差、噪声方差、长度尺度、粗糙度）。

## 跨领域应用

**[7] J. M. Tomczak, J. Swiatek, K. J. Latawiec.** "Gaussian process regression as a predictive model for Quality-of-Service in Web service systems." arXiv:1207.6910, 2012.
→ <https://arxiv.org/abs/1207.6910>
用 GPR 预测 Web 服务的 QoS 属性。结论：**线性核**表现最佳，且统计显著优于 CART 决策树。

**[8] Y. Altmann, N. Dobigeon, S. McLaughlin, J.-Y. Tourneret.** "Nonlinear Spectral Unmixing of Hyperspectral Images Using Gaussian Processes." *IEEE Transactions on Signal Processing*, 61(10):2442–2453, 2013.
DOI: [10.1109/TSP.2013.2245127](https://doi.org/10.1109/TSP.2013.2245127)
高光谱图像的非线性解混。用 GP 潜变量模型估计丰度，再用 GP 回归估计端元。

**[9] H. Topa, A. Honkela.** "Gaussian process modelling of multiple short time series." *ESANN 2015*, pp. 83–88. arXiv:1210.2503.
→ <https://arxiv.org/abs/1210.2503>
针对**大量短序列**分别拟合 GP 时容易严重过拟合或欠拟合的问题，给出两条对策：约束长度尺度使协方差能量集中在采样率的奈奎斯特频率以下；以及对观测噪声使用更强的先验。对本次实验中窗口长度的选取有直接启发。

## 延伸阅读

Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006 —— 全书可在 <http://www.gaussianprocess.org/gpml/> 免费获取，是 [1] 的完整展开。
