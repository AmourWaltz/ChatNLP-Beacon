# [线性回归 Linear Regression](./ch1_linear_regression/README.md)

&emsp;&emsp;在机器学习中，我们通常根据训练集中是否包含训练样本 (sample)（或输入向量 (input) ）所对应的标签 (label)（或目标向量 (target) ），将任务划分为带有标签的有监督学习 (supervised learning) 和不带标签的无监督学习 (unsupervised learning)。在有监督学习中，我们又根据标签的类型，将任务划分标签为离散变量的分类 (classification) 问题和标签为连续变量的回归 (regression) 问题。回归问题的目标向量通常是价钱，温度等连续的物理量，广泛应用于股票及气温等模型预测。

&emsp;&emsp;线性回归模型的最简单形式是拟合输⼊变量的线性函数，其定义如下。对于给定的数据集 $\mathcal{D} =\left \{ \boldsymbol{x,t} \right \}, \boldsymbol{x}=\left \{ \boldsymbol{x}^1, \boldsymbol{x}^2, ..., \boldsymbol{x}^n \right \} $ 是一组输入变量，$\boldsymbol{t}=\left \{ t_1, t_2, ..., t_n \right \}$ 是对应的目标向量，对于 $\forall \boldsymbol{x}^i\in\boldsymbol{x}$ 都包含 $d$ 维特征的向量。线性回归就是利用一个线性方程

$$y(\boldsymbol{x,w})=w_0+w_1\boldsymbol{x}_1+w_2\boldsymbol{x}_2+···+w_d\boldsymbol{x}_d\tag1$$

&emsp;&emsp;通过 $ \mathcal{D} $ 中已知数据估计参数 $ \boldsymbol{w}=\left \{ w_0,w_1, w_2,...,w_d \right \} $，拟合 $ \boldsymbol{x} $和$ \boldsymbol{t} $ 之间的关系，从而对未知标签的输入变量 $ \boldsymbol{x}^j $ 进行预测。参数 $ w_0 $ 使得数据中可以存在任意固定的偏置，通常被称作偏置项 (bias)。仅使用线性函数对于精确的线性回归模型虽说难窥全豹，却也可见一斑，不失为一种简单实用的方法。

## [1.1 多项式拟合 Polynomial Fitting](./ch1_linear_regression/1.1_polynomial_fitting.md)
## [1.2 线性基函数模型 Linear Basis Function Model](./ch1_linear_regression/1.2_linear_basis_function_model.md)
## [1.3 最大似然估计 Maximum Likelihoood Estimation](./ch1_linear_regression/1.3_maximum_likelihoood_estimation.md)
## [1.4 最小均方差 Minimum Square Error](./ch1_linear_regression/1.4_minimum_square_error.md)
## [1.5 梯度下降法 Gradient Desent](./ch1_linear_regression/1.5_gradient_desent.md)
## [1.6 解析法 Analytic Method](./ch1_linear_regression/1.6_analytic_method.md)