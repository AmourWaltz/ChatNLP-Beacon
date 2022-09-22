# 1 线性回归 Linear Regression

&emsp;&emsp;在机器学习任务中，我们通常根据训练数据是否包含样本 (sample)（或输入向量 (input vector) ）所对应的标签 (label)（或目标向量 (target vector) ），可以将任务划分为带标签的有监督学习 (supervised learning) 和无标签的无监督学习 (unsupervised learning)。在有监督学习中，我们又根据标签的类型，将任务划分标签为离散变量的分类 (classification) 问题和标签为连续变量的回归 (regression) 问题。回归问题的目标向量通常是价钱，温度等连续的物理量，广泛应用于股票及气温等预测模型。

&emsp;&emsp;线性回归模型的最简单形式是拟合输⼊变量的线性函数，其定义如下。对于给定含有 $n$ 个训练数据的的数据集 $\mathcal{D} =\left \{ ({\boldsymbol x}^{(1)}, t^{(1)}), ({\boldsymbol x}^{(2)}, t^{(2)}), ..., ({\boldsymbol x}^{(n)}, t^{(n)}) \right \} $，其中 $\forall {\boldsymbol x}^{(i)}\in\mathbb{R}^{\mathrm d}$。线性回归就是根据如下线性方程

$$y({\boldsymbol x, \boldsymbol w})=w_0+w_1 {x}_1+w_2{x}_2+···+w_{\mathrm d}{x}_{\mathrm d}\tag1$$

利用 $ \mathcal{D} $ 中训练数据估计一组参数 $ \boldsymbol{w}=\left \{ w_0,w_1, w_2,...,w_{\mathrm d} \right \} $ 来拟合 $y({\boldsymbol x}^{(i)},{\boldsymbol w})$ 与 $t^{(i)}$ 使之尽可能接近，进而对未知标签的输入变量 $ {\boldsymbol x}^{(j)}$ 进行预测。参数 $ w_0 $ 使得数据中可以存在任意固定的偏置，通常被称作偏置项 (bias)。这组成了最基本的线性回归模型，在很多线性场景都非常简单实用。


## 1.1 线性基函数模型 Linear Basis Function Model

> 参考： PRML 1.1, 3.1 节；面试考点总结如下
> 1. 示例：多项式拟合。
> 2. 线性基函数。

#### 多项式拟合 Polynomial Fitting

&emsp;&emsp;线性回归函数对于回归模型虽说难窥全豹，却也可见一斑。然而在大部分实际应用中，目标变量很难与输入呈线性关系，在此仅讨论单变量模型即 $\mathrm d=1$ 的情况，使用线性回归方程 $ y({x,{\boldsymbol w}}) =w_0+w_1\cdot {x}$ 很难准确拟合出输入与目标的关系，因此我们考虑将上述线性回归方程推广到更一般的多项式函数形式

$$y(x, {\boldsymbol w})=w_0+w_1\cdot x + w_2\cdot x^2 + \cdot \cdot \cdot +w_k\cdot x^k=\sum_{i=0}^{k} w_i\cdot x^i\tag1$$ 
&emsp;&emsp;其中 $ k $ 是多项式的阶数 (order)，$x^i$ 表⽰ $x$ 的 $i$ 次幂，多项式系数 $ w_0, w_1, ..., w_k $ 整体记作向量 $ \boldsymbol{w} $。
多项式函数是一类典型的线性回归模型，其线性关系实际上是针对系数 $ \boldsymbol{w} $ 而并非 $ {x} $，具备基本的线性特性

$$f(a{\boldsymbol w_1})=af({\boldsymbol w_1}),\ \ \ \ f({\boldsymbol w_1}+{\boldsymbol w_2})=f({\boldsymbol w_1})+f({\boldsymbol w_2})\tag2 $$
其中 $ {\boldsymbol w_1}, \boldsymbol{w}_2 $ 为两组可能的参数向量。

&emsp;&emsp;使用多项式函数进行曲线拟合不无道理，根据泰勒公式 (Taylor's Formula)，对于任意连续函数 $ f(x) $，如果$ f(x) $在 $ x_0 $ 处具有任意阶导数，其泰勒展开公式为

$$f(x)=f(x_0)+(x-x_0){f}'(x_0)+ (x-x_0)^2 \frac{{f}''(x_0)}{2!}+···=\sum_{n=0}^{\infty }(x-x_0)^n\frac{{f}^{(n)}(x_0)}{2!} =\sum_{n=0}^{\infty} c_nx^{n}\tag3$$
可以简化为多项式函数的形式，对于自变量只在小范围区间上定义的函数，多项式可以近乎完美的拟合曲线；但是由于多项式往往拟合的是针对输⼊变量的全局函数，其形式也并不唯一，在输⼊空间一个区域引入新的训练数据往往可能影响所有其他区域的数据拟合结果。对此我们把输⼊空间切分成若⼲个区域，然后在每个区域⽤不同的多项式函数拟合，这种函数称做样条函数 (spline function)。

&emsp;&emsp;多项式的系数可以通过调整函数拟合训练数据的⽅式确定，一般通过最小化误差函数 (error function) 实现。误差函数定量描述给定参数 $\boldsymbol w$ 时，$ y(x,{\boldsymbol w}) $ 与训练数据的差别，最常见的是计算数据集 $\mathcal D$ 中每个样本 $ x^{(i)} $ 的预测值 $ y(x^{(i)},{\boldsymbol w}) $ 与⽬标值 $ t^{(i)} $ 之差的平⽅和，其形式为

$$E({\boldsymbol w})=\frac {1}{2}\sum_{i=1}^{N}[y(x^{(i)},{\boldsymbol w})-t^{(i)}]^2\tag4$$
其中 $ E({\boldsymbol w}) $ 也称作代价函数 (cost function)，引入因⼦ $ \frac {1}{2} $ 是为了简化一阶导数形式。

#### 线性基函数 Linear Basis Functions

&emsp;&emsp;由上文可知，当我们引入一些输⼊变量 $ x $ 的幂函数进⾏线性组合时，对很多非线性实际应用的拟合效果会更好，但是精确的样条函数区域划分却很困难。所以更一般地，我们尝试引入其他形式的非线性函数，假设输入向量 $ {\boldsymbol x} \in\mathbb R^d$，我们引入一系列关于 $ \boldsymbol{x} $ 的非线性函数 $ \phi(\boldsymbol{x}) $，也称为基函数 (basis function)；特别地，对于偏置项 $ w_0 $，定义⼀个额外的虚 “基函数” $ \phi_0(\boldsymbol{x})=1 $。模型仍是关于参数 $ \boldsymbol{w} $ 的线性函数，此时回归函数形式为

$$y(\boldsymbol{x,w})=\sum_{j=0}^{M-1}w_j\phi_j(\boldsymbol{x})=\boldsymbol{w}^T\bm{\phi}(\boldsymbol{x})\tag5$$
我们把下标 $ j $ 的最⼤值记作 $ M-1 $，此时模型参数数量为 $ M $。
在机器学习中，我们对原始数据变量进行线性或非线性变换，本质上是对数据进行某种固定形式的预处理 (preprocess) 或者特征提取 (feature extraction)。
所提取特征可⽤基函数 $ \phi_j(\boldsymbol{x}) $ 表示，其中 $j = 0,1,...,M-1$。
上文所讨论的多项式曲线拟合就是一个基函数为 $ x $ 幂指数形式的线性模型。基函数有很多选择，如 “⾼斯” 基函数

$$\phi_j(x)=\exp\left \{ -\frac{(x-u_j)^2}{2s^2} \right \}\tag6$$
其中基函数在输⼊空间中的位置和大小分别由 $ u_j $ 和 $ s_j $ 决定。高斯基函数未必是⼀个概率表达式，特别地，归⼀化系数不重要，因为基函数还会与⼀个参数 $ w_j $ 相乘。
此外还有 $ \mathrm{sigmoid} $ 基函数，定义为

$$\phi_j(x)=\sigma(\frac {x-u_j} {s}),\sigma(a)=\frac {1} {1+\exp(-a)}\tag7$$
同样地，我们也可以使用 $ \tanh $ 函数，它和 $ \mathrm{sigmoid} $ 函数的关系为 $ \tanh(a)=2\sigma(2a)-1$， 因此 $ \mathrm{sigmoid} $ 函数的⼀般的线性组合等价于 $ \tanh $ 函数的⼀般的线性组合。下图说明了基函数分别是多项式函数，高斯函数以及 $ \mathrm{sigmoid} $ 函数时的情况

<div align=center>
<img src="images/1_2_lbfm1.png" width="75%"/>
</div>

&emsp;&emsp;基函数还可以选择傅⾥叶基函数，⽤正弦函数展开。每个基函数表⽰⼀个具体频率，在频域中是有限的但在空间域中⽆限延伸，相反，在时域有限的基函数在频域中是无限的。与标准傅里叶变换比较，小波在时域和频域都是局部的，对于时间序列中的连续的时间点，以及图像中的像素都有广泛的应用。

&emsp;&emsp;关于基函数的形式还会在后续章节中继续讨论，选择合适的基函数，我们可以建⽴输⼊向量到⽬标值之间的任意映射，这一章，我们更关注基函数为 $ \phi(\boldsymbol{x})=\boldsymbol{x} $ 的一般情形。然而，线性基函数模型有一些重要局限，由于我们假设对于任意观测数据基函数都是固定的，随着输⼊空间维度 $ \mathrm d $ 迅速增长，基函数的数量呈指数级增长，带来维度灾难 (the curse of dimensionality)。

## 1.2 最大似然与最小平方 Maximum Likelihoood & Least Square

> 参考： PRML 3.1 节；面试考点总结如下
> 1. 示例：多项式拟合。
> 2. 线性基函数。

#### 最大似然估计 Maximum Likelihoood Estimation

&emsp;&emsp;线性回归问题都可以通过最⼩化误差函数来解决，通常选择平方和作为误差函数，接下来就从最大似然估计 (maximum likelihood estimation, MLE) 的角度解释为什么选择最小平方 (least squares) 来解决回归问题。

&emsp;&emsp;考虑这样一个线性回归模型：已知含有 $\it N$ 个输入数据点的数据集 $\boldsymbol{X}=\left \{ \boldsymbol x^{(1)}, \boldsymbol x^{(2)}, ..., \boldsymbol x^{(N)} \right \}$ 及对应的目标变量集合 $\boldsymbol{t}=\left \{ t^{(1)}, t^{(2)}, ..., t^{(N)} \right \}$，对新输入的变量 $\boldsymbol x^{(j)}$ 的目标值 $t^{(j)}$ 做出预测。正如前文所讲，我们对任意 $\boldsymbol{x}^{(i)}$ 到 ${t}^{(i)}$ 的真实映射关系并不知晓，只是人为的构造一些基函数模型来逼近真实映射，我们使用关于目标的概率分布来表示预测可信度。
通常选择高斯分布 (Gaussian distribution) 来建模，因为实际建模的很多分布是比较接近高斯分布的。中心极限定理（central limit theorem）说明很多独立随机变量的和近似服从高斯分布，意味着在实际中，很多复杂系统都可以被成功地建模成高斯分布的噪声。在⾼斯噪声模型的假设下，我们假定给定 $\boldsymbol x^{(i)}$ 的值， 对应 $t^{(i)}$ 的值服从⾼斯分布，分布均值为 $y(\boldsymbol x^{(i)},\boldsymbol{w})$ ，此时有

$$p(t^{(i)}|\boldsymbol x^{(i)}, \boldsymbol{w}, \beta) = \mathcal{N} (t^{(i)}|y(\boldsymbol x^{(i)}, \boldsymbol{w}), \beta^{-1})\tag 8 $$
其中 $\beta$ 对应高斯分布方差的导数，很显然分布均值 $y(\boldsymbol x^{(i)}, \boldsymbol{w})$ 就是我们选取的拟合函数。即我们对 $\boldsymbol x^{(i)}$ 的预测结果 $t^{(i)}$ 加了一个均值为0，方差为 $\beta$ 的高斯噪声 $\epsilon$ ，使得 $t=y(\boldsymbol x,\boldsymbol{w})+\epsilon$ ，最终预测结果符合以上假设的概率分布。而新输入的 $\boldsymbol x^{(j)}$ 的最优预测由⽬标变量的条件均值给出，条件均值可以写为

$$\mathbb{E} \left [ t|\boldsymbol x  \right ] =\int t\cdot p(t|\boldsymbol x)\mathrm dt=y(\boldsymbol x,\boldsymbol{w})\tag 9$$
为了证明假设的概率分布适用于给定数据集上的所有数据，我们需要用训练数据 $\left \{\boldsymbol{X},\boldsymbol{t} \right \}$ ，通过最⼤似然 (maximum likelihood) ⽅法，来决定参数 $\boldsymbol{w}$ 和 $\beta$ 的值。所谓最大似然估计，就是寻找能够以较高概率产生观测数据的概率模型（或似然函数），通过最⼤化数据集的真实概率分布计算固定的参数，即认为参数 $\boldsymbol{w}$ 和 $\beta$ 是定值而数据集是变量，与之相对应的最大后验估计 (maximum posterior) 则是在给定数据集的情况下最⼤化参数的概率分布，认为参数 $\boldsymbol{w}$ 和 $\beta$ 是变量而数据集是定值，这个会在后续篇幅展开讨论。此时似然函数为

$$p(\boldsymbol{t}|\boldsymbol{X}, \boldsymbol{w}, \beta) = \prod_{i=1}^{N}\mathcal{N} (t^{(i)}|y(\boldsymbol x^{(i)}, \boldsymbol{w}), \beta^{-1})\tag {10}$$
为了最大化似然函数，我们会将高斯分布取对数然后替换之，得到对数似然函数

$$\ln{p(\boldsymbol{t} |\boldsymbol{X},\boldsymbol{w},\beta) } =-\frac{\beta}{2} \sum_{i=1}^{N} \left \{ y(\boldsymbol x^{(i)},\boldsymbol{w})-t^{(i)} \right \}^2+\frac{N}{2}\ln{\beta }-\frac{N}{2}\ln{2\pi }\tag {11}$$

&emsp;&emsp;⾸先考虑确定 $\boldsymbol{w}$ 的最⼤似然解（记作 $\boldsymbol{w}^{\star}$ ），先省略与 $\boldsymbol{w}$ 无关的公式后两项，同时注意到，使⽤⼀个正的常数系数来缩放对数似然函数并不会改变 $\boldsymbol w^{\star}$ 的位置， 因此我们可以⽤ $\frac {1}{2}$ 来代替系数 $\frac {\beta}{2}$ 。最后，我们不去最⼤化似然函数，⽽是等价地最⼩化负对数似然函数。于是对于确定 $\boldsymbol w^{\star}$ 的问题来说，最⼤化似然函数等价于最⼩化第一节中的平⽅和误差函数。因此，在⾼斯噪声的假设下，平⽅和误差函数是最⼤化似然函数的⼀个⾃然结果。在确定了控制均值的参数向量 $\boldsymbol w^{\star}$ 之后，就可以求解 $\beta$ ，因为平方和误差函数的最小值为0，很容易确定最大化似然函数的 $\beta^{\star}$ ，其形式为

$$\frac{1}{\beta^{\star}} =\frac{1}{N} \sum_{i=1}^{N} \left \{ y(\boldsymbol x^{(i)},\boldsymbol{w}^{\star})-t^{(i)} \right \}^2\tag {12}$$

&emsp;&emsp;此时关于新输入的 $\boldsymbol x^{(j)}$ ，我们代入参数确定的最大似然函数，得到关于 $t^{(j)}$ 的预测分布

$$p(t^{(j)}|\boldsymbol x^{(j)},\boldsymbol w^{\star}, \beta^{\star})=\mathcal{N}(t_j|y(x_j, \boldsymbol w^{\star}),{\beta^{\star}} ^{-1})\tag {13}$$

#### 最大后验估计 Maximum A Posterior

&emsp;&emsp;这时再考虑将参数 $\boldsymbol w$ 看做服从某种概率分布的变量，同样假设为高斯噪声模型

$$p(\boldsymbol w|\alpha)=\mathcal{N}(\boldsymbol w|\boldsymbol 0, \alpha^{-1}\boldsymbol{I})=(\frac {\alpha}{2\pi})^{\frac {M}{2}}\exp \left \{ -\frac{\alpha}{2} \boldsymbol {w} ^T \boldsymbol{w} \right \}\tag{14}$$
其中 $\alpha$ 是分布方差， $M$ 是参数数量，选用0均值是因为我们通常在实际操作中将 $\boldsymbol w$ 初始化为 $\boldsymbol 0$ 。如果再将 $\boldsymbol w$ 看做给定数据集 $\left \{ \boldsymbol {X,t} \right \}$ 上的条件概率分布，可表示为 $p(\boldsymbol w| \boldsymbol{X,t})$ ，也称作 $\boldsymbol w$ 的后验概率 (posterior distribution)，与似然概率 $p(\boldsymbol t|\boldsymbol{w,X})$ 相同的是，我们总是将概率的条件项假设为定值。使⽤贝叶斯定理 (Bayes Theorem)， $\boldsymbol w$ 的后验概率正⽐于先验分布和似然函数的乘积。

$$p(\boldsymbol w| \boldsymbol{X},\boldsymbol{t},\alpha,\beta)\propto p(\boldsymbol t|\boldsymbol{w},\boldsymbol{X},\beta)p(\boldsymbol w|\alpha)\tag {15}$$
为了最大化后验概率分布确定 $\boldsymbol w$ ，同样将高斯分布公式代入并取负对数，可得最大化后验概率就是最小化下式

$$\frac{\beta}{2} \sum_{i=1}^{N} \left \{ y(\boldsymbol x^{(i)},\boldsymbol{w})-t_n \right \}^2+\frac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w}\tag{16}$$
其中第二项 $\frac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w}$ 也称作 $L_2$ 正则项 ($L_2$  regularization )，具有防止过拟合 (overfitting) 的作用，这些将在后续章节中详细讨论，下一篇文章将讲述误差函数 $E(\boldsymbol{w})$ 的求解方式，我们只考虑最大似然项，即针对 $\boldsymbol w$ 的最小平方和函数。


## 1.3 最小均方差 Minimum Square Error

> 参考： PRML 3.1 节；面试考点总结如下
> 1. 示例：多项式拟合。
> 2. 线性基函数。

#### 顺序学习 Sequential Learning

&emsp;&emsp;紧接上文，我们利用线性回归模型进行拟合，需要根据训练数据调节参数 $\boldsymbol{w}$ 的值，使得对于任意训练数据 $\boldsymbol{x}^{(i)}$ ，模型的输出 $y(\boldsymbol{x}^{(i)}$, $\boldsymbol{w})$ 更加接近目标值 $t^{(i)}$ ，同时最小化代价函数 $E(\boldsymbol{w})=\frac {1}{2}\sum_{i=1}^{N}[y(\boldsymbol x^{(i)},\boldsymbol{w})-t^{(i)}]^2$ 使之无限趋近于0。由于误差函数是系数 $\boldsymbol{w}$ 的二次函数（线性回归 $y(\boldsymbol{x}, \boldsymbol{w})$ 本身定义就是针对 $\boldsymbol{w}$ 的线性函数），所以其导数是关于 $\boldsymbol{w}$ 的线性函数，误差函数的最小值有唯一最优解 $\boldsymbol{w}^\star$ ，线性回归问题也可以看作是针对误差函数的优化问题。但由于求解系数 $\boldsymbol{w}$ 通常是在一整个大数据集上进行的，我们很难得到精确的 $\boldsymbol{w}^\star$ ，最直观的方法就是逐一考虑每一个数据点 $\boldsymbol{x}^{(i)}$ 来逼近 $\boldsymbol{w}^\star$ 。
这种求解方法也称作顺序算法 (sequential algorithm)（或在线算 (on-line algorithm)），通过迭代每次只考虑⼀个数据点，模型的参数在每观测到数据点之后进⾏更新（实际上一般不会在每个点计算后都更新一次，这样效率极低且误差函数往往震荡不收敛。一般都是在小批量操作的），在迭代过程中通常梯度下降 (gradient descent) 算法来逐步优化求解 $\boldsymbol{w}^\star$ 。

&emsp;&emsp;为了能找到使误差函数 $E(\boldsymbol{w})$ 最小的参数 $\boldsymbol{w}$ ，我们对 $\boldsymbol{w}$ 的初值先做出一定假设（比如初始化为0），然后在利用梯度下降算法不断更新 $\boldsymbol{w}$ 使之收敛至某处可以最小化 $E(\boldsymbol{w})$ ，如下

$$\boldsymbol w^{(\tau+1)}=\boldsymbol w^{(\tau)}-\eta\nabla E(\boldsymbol w)\tag {17}$$
其中 $\tau$ 是迭代次数， $\eta$ 被称作学习率，是我们需要手动调节的超参数 (hyperparameter)，会直接影响到函数能否收敛，需要反复斟酌选定。梯度下降算法通过反复迭代使权值向量 $\boldsymbol w$ 都会沿着误差函数 $E(\boldsymbol{w})$ 下降速度最快的⽅向移动，因此这种⽅法被称为梯度下降法或最速下降法 (steepest descent)。假设线性回归函数形式为 $y(\boldsymbol{x,w})=\boldsymbol{w}^T\phi(\boldsymbol{x})$ ，方便起见我们令 $\phi(\boldsymbol x)=\boldsymbol x$ ，对于 $y$ 的平⽅和误差函数，有

$$\boldsymbol w^{(\tau+1)}=\boldsymbol w^{(\tau)}-\eta(t-\boldsymbol w^{(\tau)T}\boldsymbol x)\tag{18}$$ 
这被称为最小均方 (least mean squares, LMS) 或 Widrow-Hoff 算法。这种⽅法在直觉上看很合理，预测和目标之差 $t-\boldsymbol w^{(\tau)T}\boldsymbol x$ 是参数 $\boldsymbol{w}$ 更新幅值的一项因子，包含梯度方向和幅值信息，因此，当预测值与真实目标比较接近时， $\boldsymbol{w}$ 变化幅度也会较小，反之亦然。

#### 最小平方和的几何描述 Geometric Explanation of Least Square

&emsp;&emsp;我们考虑⼀个 $N$ 维空间，它的坐标轴由 $\boldsymbol t$ 给出，即 $\boldsymbol t=\left [ t^{(1)},t^{(2)},...,t^{(N)} \right ]^T$ 是这个空间中的⼀个向量。 每个由 $N$ 个数据点构成的基函数 $\left [\phi_j(\boldsymbol x^{(1)}), \phi_j(\boldsymbol x^{(2)}),\dots,\phi_j(\boldsymbol x^{(N)})\right ]^T$ 也可以表示为这个空间中的⼀个向量，记作 $\varphi_j$。$\varphi_j$ 对应于 $\Phi$ 的第 $j$ 列，⽽ $\boldsymbol \phi(\boldsymbol x^{(n)})$ 对应于 $\Phi$ 的第 $n$ ⾏。 如果基函数的数量 $M$ ⼩于数据点的数量 $N$ (很容易联想特征数小于数据点个数，参数才有唯一最优解)，那么 $M$ 个向量 $\varphi_j$ 将会张成⼀个 $M$ 维的⼦空间 $\mathcal{S}$  。我们定义 $\boldsymbol y=\left \{ y(\boldsymbol x^{(1)},\boldsymbol w),y(\boldsymbol x^{(2)},\boldsymbol w),...,y(\boldsymbol x^{(N)},\boldsymbol w) \right \}^T$ 是⼀个 $N$ 维向量。由于 $\boldsymbol y$ 是 $M$ 个向量 $\varphi_j$ 的任意线性组合，因此它可以位于 $M$ 维⼦空间的任何位置。这样，平⽅和误差函数就等于 $\boldsymbol y$ 和 $\boldsymbol t$ 之间的欧⽒平方距离。 因此， $\boldsymbol w$ 的最⼩平⽅解对应位于⼦空间 $\mathcal S$ 的与 $\boldsymbol t$ 最近的 $\boldsymbol y$ 的选择。

&emsp;&emsp;直观来看，根据下图，我们很容易猜想这个解对应于 $\boldsymbol t$ 在⼦空间 $\mathcal S$ 上的正交投影。事实上确实是这样。 假设 $\boldsymbol y^{\mathcal {(S)}}$ （下图中的 $\boldsymbol y$，此处重新定义向量是针对原书做区分，原书中已经定义了 $\boldsymbol y$ 是一个 $N$ 维向量此处又出现在⼦空间 $\mathcal S$ 上容易产生误解）是子空间 $\mathcal S$ 上 $\varphi_1,\varphi_2$ 在参数为 $\boldsymbol w^\star$ 时的线性组合，只要证明它的表达式为 $\boldsymbol t$ 的正交投影即可。

<div align=center>
<img src="images/1_4_mse1.png" width="72%"/>
</div>

## 1.4 梯度下降法 Gradient Desent

> 参考： PRML 3.1 节；面试考点总结如下
> 1. 示例：多项式拟合。
> 2. 线性基函数。

#### 批量梯度下降 Gradient Desent

&emsp;&emsp;梯度下降算法使⽤梯度信息，权值参数 $\boldsymbol{w}$ 通过迭代的方式，每次在误差函数 $E(\boldsymbol{w})$ 关于 $\boldsymbol{w}$ 负梯度⽅向上通过⼀次⼩的移动进行更新。误差函数是关于训练集定义的，因此计算 $\nabla E(\boldsymbol{w})$，每⼀步都需要处理整个训练集，也称作批量梯度下降 (batch gradient descent)，如下

$$\boldsymbol w^{(\tau+1)}=\boldsymbol w^{(\tau)}-\eta \frac{1}{N}\sum_{i=1}^{N}(t^{(i)}-\boldsymbol w^{(\tau)T}\boldsymbol x^{(i)})\tag{19}$$
此处假设 $y(\boldsymbol w, \boldsymbol x)=\boldsymbol w^T\boldsymbol x$。需要注意的是，虽然梯度下降通常可能会受到局部极小值的影响，但我们在此处针对的优化问题 $E(\boldsymbol{w})$ 是一个关于 $\boldsymbol{w}$ 的二次凸函数 (convex quadratic function)，因此它只存在一个全局极小值 (global minimum) 且没有局部最小 (local minimum)。如下图所示，椭圆表示二次型函数轮廓的轮廓采样。蓝色轨迹表示参数通过梯度下降算法的连续过程

<div align=center>
<img src="images/1_5_gd1.png" width="75%"/>
</div>

&emsp;&emsp;对于非线性的情形，误差函数 $E(\boldsymbol{w})$ 可能存在多个局部最优，为了能找到⼀个⾜够好的极⼩值，可能有必要多次运⾏梯度下降算法，每次都随机选⽤不同的起始点，然后在⼀个独⽴的数据集上对⽐最终性能。
实际上批量梯度下降法在大数据集使用时效率很低，每一步都需要计算全部数据的梯度值，运算开销很高。然⽽，梯度下降法有另⼀个版本，与批量梯度下降算法不同，我们每一步更新都是基于⼀个独⽴观测数据的误差函数，这种算法也称作随机梯度下降 (stochastic gradient descent) 或顺序梯度下降 (sequential gradient descent, Bishop PRML) 或递增梯度下降 (incremental gradient descent, CS229)，使得权值参数的更新每次只依赖于⼀个数据点，即

$$\boldsymbol w^{(\tau+1)}=\boldsymbol w^{(\tau)}-\eta(t_n-\boldsymbol w^{(\tau)T}\boldsymbol x^{(i)})\tag{20}$$
与批量梯度下降相⽐，随机梯度下降的⼀个优点是可以更加⾼效地处理数据中的冗余性。考虑⼀种极端的情形：给定⼀个数据集，我们将每个数据点都复制⼀次，从⽽将数据集的规模翻倍，但这仅仅把误差函数乘以⼀个因⼦2，等价于使⽤原始的误差函数。批量梯度下降法必须付出两倍的计算量来计算误差函数的梯度，⽽在随机梯度下降法则不受影响。

&emsp;&emsp;随机梯度下降另⼀个特点是便于逃离局部极⼩值点，因为整个数据集误差函数的驻点通常不会是单一数据点误差函数的驻点，然而，也正因如此，我们几乎永远无法得到关于整个数据集的参数最优解 $\boldsymbol w^\star$ ，因为我们每次计算的都是单一数据点误差函数关于 $\boldsymbol w$ 的梯度。此外，随机梯度下降法的参数更新过于频繁，除了迭代次数会骤增，权值参数的更新过程也会不断震荡很难收敛。折中的⽅法是，每次更新依赖于一组独立观测的数据点，实现小批量数据上的随机梯度下降 (minibatch stochastic gradient descent)。

#### 二次函数曲率 Quadratic Function Curvature

&emsp;&emsp;有时我们需要计算输入输出均为向量的函数的所有偏导数（可以理解为多目标的回归问题或分类问题），包含所有这样的偏导数的矩阵被称为 Jacobian 矩阵。具体来说，如果我们有一个函数映射 $f(\boldsymbol x):\mathbb{R}^m\rightarrow\mathbb{R}^n$ ， $f$ 的 Jacobian 矩阵 $\boldsymbol J \in\mathbb{R}^{n\times m}$ 定义为 $J_{i,j}=\frac{\partial }{\partial x_j} {f(\boldsymbol x)}_i$ 。
有时，我们也对导数的导数感兴趣，即二阶导数 (second derivative)，一维情况下，我们将二阶导数记作 $\frac{\partial^2 }{\partial x^2} {f(x)}$ ，二阶导数表示一阶导数如何随着输入的变化而改变，即基于梯度信息的下降步骤是否会产生如我们预期的那样的改善。二阶导数是对曲率 (curvature) 的衡量。

&emsp;&emsp;延续上述篇幅的所使用的二次误差函数 $E(\boldsymbol w)$ ，真实的误差函数形式往往要复杂许多，但利用泰勒展开我们可以在小范围区间用多项式函数拟合绝大部分未知函数，而随机梯度下降的更新也是在微观步骤上更新的，因此对于连续可导的误差函数 $E(\boldsymbol w)$ ，我们同样可以把它当做一个样条函数（参见第一篇文章），每一处都当做一个多项式函数，为简便起见我们只考虑到二阶泰勒展开级数，也就是使用二次函数进行局部拟合，此时误差函数就可以看做由无数个不同的二次函数的微小区间所组成的连续且处处可导的函数。
取二次函数进行拟合也是为了更方便的观察二阶导数，我们使用沿负梯度方向大小为的 $\epsilon$ 下降步（类似上文梯度下降算法中的学习率 $\eta$ ，因为梯度下降算法目标与预测值之差也会包含一定下降幅度信息，而此处下降步包含了所有梯度下降的幅度信息）。当该梯度是 1 时，代价函数将下降 $\epsilon$ 。 如果二阶导数是负的 (negative curvature)，函数曲线向上凸出，代价函数将下降的比 $\epsilon$ 多；如果二阶导数为 0 (no curvature)，那就没有曲率，仅用梯度就可以预测它的值；如果二阶导数是正的 (positive curvature)，函数曲线是向下凸出，代价函数将下降的比 $\epsilon$ 少。下图依次反应了三种情况的曲率如何影响基于梯度的预测值与真实误差函数值的关系。

<div align=center>
<img src="images/1_5_gd2.png" width="75%"/>
</div>

#### Hessian 特征值

&emsp;&emsp;当函数具有多维输入输出时，二阶导数也有很多。我们可以将这些导数合并成一个矩阵，称为 Hessian 矩阵。Hessian 矩阵 $\boldsymbol H (f)(\boldsymbol x)$ 定义为

$$\boldsymbol H (f)(\boldsymbol x )_{i,j}=\frac{\partial^2 }{\partial x_i \partial x_j} {f(\boldsymbol x)}=\frac{\partial^2 }{\partial x_j \partial x_i} {f(\boldsymbol x)}\tag {21}$$
其中二阶导数项表示 $f$ 的一阶导数（关于 $x_j$ ）关于 $x_i$ 的导数，Hessian 等价于梯度的 Jacobian 矩阵，微分算子在任何二阶偏导连续的点处可交换，也就是它们的顺序可以互换。这意味着 $\boldsymbol H_{i,j}=\boldsymbol H_{j,i}$ ，因此 Hessian 矩阵在这些点上是对称的。因为 Hessian 矩阵是实对称的，我们可以将其分解成一组实特征值和一组特征向量的正交基。在特定方向 $\boldsymbol d$ 上的二阶导数可以写成 $\boldsymbol d^{T}\boldsymbol {Hd}$ 。当 $\boldsymbol d$ 是 $\boldsymbol H$ 的一个特征向量时，这个方向的二阶导数就是对应的特征值。对于其他的方向 $\boldsymbol d$ ，方向二阶导数是所有特征值的加权平均，权重在 0 和 1 之间， 且与 $\boldsymbol d$ 夹角越小的特征向量的权重越大。最大特征值确定最大二阶导数，最小特征值确定最小二阶导数。
我们可以通过（方向）二阶导数预期一个梯度下降步骤能表现得多好。我们在 当前点 $\boldsymbol {x}^{(0)}$ 处作函数 $f(\boldsymbol x)$ 的近似二阶泰勒级数：

$$f(\boldsymbol x)\approx f(\boldsymbol x^{(0)})+(\boldsymbol x-\boldsymbol x^{(0)})^T\boldsymbol g+\frac{1}{2}(\boldsymbol x-\boldsymbol x^{(0)})^T\boldsymbol H(\boldsymbol x-\boldsymbol x^{(0)})\tag {22}$$
其中 $\boldsymbol g$ 是梯度， $\boldsymbol H$ 是 $\boldsymbol {x}^{(0)}$ 点的 Hessian。如果我们使用学习率 $\eta$ ， 那么新的点 $\boldsymbol x$ 将会是 $\boldsymbol x^{(0)}-\eta\boldsymbol g$ 。 代入上述近似，可得

$$f(\boldsymbol x^{(0)}-\eta\boldsymbol g)\approx f(\boldsymbol x^{(0)})+-\eta\boldsymbol g^T\boldsymbol g+\frac{1}{2} \eta^2 \boldsymbol g^T\boldsymbol H\boldsymbol g\tag {23}$$
其中有 3 项：函数的原始值、函数斜率导致的预期改善、函数曲率导致的校正。当最后一项太大时，梯度下降实际上是可能向上移动的。当 $\boldsymbol g^T\boldsymbol H\boldsymbol g$ 为零或负时，近似的泰勒级数表明增加 $\eta$ 将永远使 $f$ 下降。在实践中，泰勒级数不会在 $\eta$ 大的时候也保持准确，因为我们对误差函数的近似二次拟合都是微小区间的，学习率过大很可能就到另一个二次函数的区间了，因此在这种情况下我们必须采取更启发式的选择。当 $\boldsymbol g^T\boldsymbol H\boldsymbol g$ 为正时，通过计算可得，使近似泰勒级数下降最多的最优学习率为

$$\eta^\star=\frac{\boldsymbol g^T\boldsymbol g}{\boldsymbol g^T\boldsymbol H\boldsymbol g}\tag {24}$$
最坏的情况下， $\boldsymbol g$ 与 $\boldsymbol H$ 最大特征值 $\lambda_{max}$ 对应的特征向量对齐，则最优步长是 $\frac {1}{\lambda_{max}}$ 。 我们要最小化的误差函数能用二次函数很好地近似的情况下，Hessian 的特征值决定了学习率的量级。

#### 二阶导数测试 Second Derivative Test

&emsp;&emsp;二阶导数还可以被用于确定一个临界点是否是局部极大点、局部极小点或鞍点。在临界点处 ${f}' (x)=0$ 。而 ${f}''(x) > 0$ 意味着 ${f}' (x)$ 会随着我们移向右边而增加，移向左边而减小，也就是 ${f}' (x-\delta)<0 和 {f}' (x-\delta)>0$ 对足够小的 $\delta$ 成立。因此我们得出结论，当 ${f}' (x)=0$ 且 ${f}''(x) > 0$ 时， $x$ 是一个局部极小点。同样，当 ${f}' (x)=0$ 且 ${f}''(x) <0$ 时， $x$ 是一个局部极大值点。这就是所谓的二阶导数测试 (second derivative test)。不幸的是，当 ${f}''(x) = 0$ 时测试是不确定的。在这种情况下， $x$ 可以是一个鞍点或平坦区域的一部分。

&emsp;&emsp;在多维情况下，我们需要检测函数的所有二阶导数。利用 Hessian 的特征值分解，我们可以将二阶导数测试扩展到多维情况。在临界点处 $\nabla_{\boldsymbol x}  f(\boldsymbol x)=0$ ，我们通过检测 Hessian 特征值来判断该临界点是一个局部极大点、局部极小点还是鞍点。 当 Hessian 是正定的（所有特征值都是正的），则该临界点是局部极小点。同样的，当 Hessian 是负定的（所有特征值都是负的），这个点就是局部极大点。在多维情况下，我们可以找到确定该点是否为鞍点的迹象。如果 Hessian 的特征值中至少一个是正的且至少一个是负的，那么 $x$ 是 $f$ 某个横截面的局部极大点，却是另一个横截面的局部极小点，典型例子就是鞍点。最后，多维二阶导数测试可能像单变量版本那样是不确定的。当所有非零特征值是同号的且至少有一个特征值是 0 时，这个检测就是不确定的。这是因为单变量的二阶导数测试在零特征值对应的横截面上是不确定的。

&emsp;&emsp;多维情况下单个点处每个方向上的二阶导数是不同。Hessian 的条件数 (condition number) 衡量这些二阶导数的变化范围。当 Hessian 的条件数很差时，梯度下降法也会表现得很差。这是因为一个方向上的导数增加得很快，而在另一个方向上增加得很慢。梯度下降不知道导数的这种变化，所以它不知道应该优先探索导数长期为负的方向，如下图所示。为了避免冲过最小而向具有较强正曲率的方向，防止发生震荡，我们需要将学习率设置较小，然而这也导致了更新步长太小，在其他较小曲率的方向上进展不明显。

<div align=center>
<img src="images/1_5_gd3.png" width="75%"/>
</div>

#### 牛顿法 Newton Method

&emsp;&emsp;上图中，红线表示梯度下降的路径。这个非常细长的二次函数类似一个长峡谷。梯度下降把时间浪费于在峡谷壁反复下降，因为它们是最陡峭的特征。由于步长有点大，有超过函数底部的趋势，因此需要在下一次迭代时在对面的峡谷壁下降。与指向该方向的特征向量对应的 Hessian 的大的正特征值表示该方向上的导数值仍然较大，因此基于 Hessian 的优化算法可以预测，在此情况下最陡峭方向实际上不是有前途的搜索方向。

&emsp;&emsp;我们可以使用 Hessian 矩阵的信息来指导搜索，以解决这个问题。其中最简单的方法是牛顿法 (Newton’s method)。仅使用梯度信息的优化算法被称为一阶优化算法 (ﬁrst-order optimization algorithms)，如梯度下降。使用 Hessian 矩阵的优化算法被称为二阶最优化算法 (second-order optimization algorithms)，如牛顿法。牛顿法基于一个二阶泰勒展开来近似 $\boldsymbol {x}^{(0)}$ 附近的 $f(\boldsymbol x)$ ：

$$f(\boldsymbol x)\approx f(\boldsymbol x^{(0)})+(\boldsymbol x-\boldsymbol x^{(0)})^T\nabla_{\boldsymbol x} f(\boldsymbol x^{(0)})+\frac{1}{2}(\boldsymbol x-\boldsymbol x^{(0)})^T\boldsymbol H(f)(\boldsymbol x^{(0)})(\boldsymbol x-\boldsymbol x^{(0)})\tag {25}$$
计算这个函数的临界点，令 ${f}' (\boldsymbol x)=0$ ，得

$$\boldsymbol x^\star=\boldsymbol x^{(0)}-\boldsymbol H(f)(\boldsymbol x^{(0)}) ^{-1}\nabla_{\boldsymbol x} f(\boldsymbol x^{(0)})\tag {26}$$
当 $f$ 是一个正定二次函数时，牛顿法只要应用一次式就能直接跳到函数的最小点。如果 $f$ 能在局部近似为正定二次，牛顿法则需要多次迭代。迭代地更新近似函数和跳到近似函数的最小点可以比梯度下降更快地到达临界点。这在接近局部极小点时是一个特别有用的性质，但是在鞍点附近是有害的。当附近的临界点是最小点（Hessian 的所有特征值都是正的）时牛顿法才适用，而梯度下降不会被鞍点吸引。

## 1.5 解析法 Analytic Method

> 参考： PRML 3.1 节；面试考点总结如下
> 1. 示例：多项式拟合。
> 2. 线性基函数。

&emsp;&emsp;我们可以利用梯度下降法来求解最小平方和函数，回忆第一节所定义的平方和误差函数 $E(\boldsymbol{w})=\frac {1}{2}\sum_{i=1}^{N}[y(\boldsymbol x^{(i)},\boldsymbol{w})-t^{(i)}]^2$ ，也是似然函数，我们可以用最大似然的方法确定 $\boldsymbol w$ 和 $\beta$ ，在条件⾼斯噪声分布的情况下，采用线性基函数的形式，线性函数的对数似然函数如下

$$\ln{p(\boldsymbol{t} |\boldsymbol{X},\boldsymbol{w},\beta) } =-\frac{\beta}{2} \sum_{i=1}^{N} \left \{ t^{(i)} - \boldsymbol{w}^T\phi (\boldsymbol x^{(i)}) \right \}^2+\frac{N}{2}\ln_{}{\beta }-\frac{N}{2}\ln_{}{2\pi }\tag{27}$$

&emsp;&emsp;线性模型的似然函数的最⼤化等价于平⽅和误差函数的最⼩化，我们在第二篇文章提及，对数似然函数是关于 $\boldsymbol{w}$ 的二次函数，所以似然函数有唯一最优解 $\boldsymbol{w}^\star$，前文中我们是从微观的角度将训练集中的数据依次代入计算 $\boldsymbol{w}$ ，通过梯度下降算法用迭代的形式逼近 $\boldsymbol{w}^\star$。本文将使用一种更为直观，整体的方法，首先对数似然函数的梯度为

$$\nabla \ln{p(\boldsymbol{t} |\boldsymbol{X},\boldsymbol{w},\beta) } = \beta\sum_{i=1}^{N} \left \{t^{(i)} - \boldsymbol{w}^T\phi (\boldsymbol x^{(i)})\right \}\phi (\boldsymbol x^{(i)})\tag {28}$$

&emsp;&emsp;令对数似然函数的梯度为 0 求解 $\boldsymbol{w}^\star$ ，可得

$$\boldsymbol{w}^\star = ( \Phi^T \Phi)^{-1}\Phi^T\boldsymbol t\tag {29}$$

&emsp;&emsp;这就是使用解析法 (analytic method) 对方程直接求解，也称为最小平方问题的规范方程 (normal equation)。这⾥ $\Phi$ 是⼀个 $N\times M$ 的矩阵，被称为设计矩阵 (design matrix)，它的元素为 $\Phi_{i,j}=\phi_j(\boldsymbol x^{(i)})$ ，即

$$\Phi=\begin{pmatrix}  \phi_0 (\boldsymbol x^{(1)}) & \phi_1 (\boldsymbol x^{(1)}) & \dots  & \phi_{M-1} (\boldsymbol x^{(1)}) \\  \phi_0 (\boldsymbol x^{(2)}) & \phi_1 (\boldsymbol x^{(2)}) & \dots  & \phi_{M-1} (\boldsymbol x^{(2)}) \\  \vdots  & \vdots  & \ddots  & \vdots \\  \phi_0 (\boldsymbol x^{(N)}) & \phi_1 (\boldsymbol x^{(N)}) & \dots & \phi_{M-1} (\boldsymbol x^{(N)})  \end{pmatrix}$$

&emsp;&emsp;$\Phi^{\dagger }\equiv (\Phi^T\Phi)^{-1}\Phi^T$ 被称为矩阵 $\Phi$ 的 Moore-Penrose 伪逆矩阵 (pseudo-inverse matrix)。 它可以被看成逆矩阵的概念对于⾮⽅阵的矩阵的推⼴。实际上，如果 $\Phi$ 是⽅阵且可逆，那么使⽤性质 ($\boldsymbol {AB})^{-1}=\boldsymbol B^{-1} \boldsymbol A^{-1}$ ，我们可以看到 $\Phi ^\dagger \equiv \Phi ^{-1}$ 。在实际应⽤中，当 ($\Phi^T\Phi)^{-1}$ 接近奇异矩阵时，直接求解规范⽅程会导致数值计算上的困难。特别地，当两个或者更多的基函数向量共线（矩阵 $\Phi$ 的列）或者接近共线时，最终的参数值会相当⼤。这种数值计算上的困难可以通过奇异值分解 (singular value decomposition) 的⽅法解决。注意，正则项的添加确保了矩阵是⾮奇异的。

&emsp;&emsp;注意此处关于基函数的写法，实际上 (12) 式中 $\phi (\boldsymbol x^{(i)}) = \left [ \boldsymbol \phi_1(\boldsymbol x^{(i)}), \boldsymbol \phi_2(\boldsymbol x^{(i)}),...,\boldsymbol \phi_{M-1}(\boldsymbol x^{(i)}) \right ]$ ，参数 $\boldsymbol w$ 的维度是 $M$ ，在第一篇文章节提到，线性基函数变换的本质还是特征提取，此处直接对基函数变换之后的特征进行线性变换，所以我们忽略了数据点 $\boldsymbol x^{(i)}$ 的原始维度，而 CS229 讲义中是对原始数据  $\boldsymbol x^{(i)}$ 进行线性变换，所以写法上略有不同，但原理是一样的，此处的 $\Phi$ 相当于 CS229 讲义中的 X ，是一个 $N\times M$ 的矩阵，横向是将 (12) 式中的求和步骤改为将所有训练数据点合并在矩阵内，纵向是每个数据点的特征，如果是直接进行线性变换就是原始数据点的维度，如果进行过基函数变换就是基函数的数量。
现在，我们可以更加深刻地认识偏置参数 $w_0$ 。如果我们显式地写出偏置参数，那么误差函数变为

$$E(\boldsymbol{w})=\frac {1}{2}\sum_{i=1}^{N}[t^{(i)}-w_0-\sum_{j=1}^{M-1}w_j\phi_j(\boldsymbol x^{(i)})]^2\tag {30}$$

&emsp;&emsp;令关于 $w_0$ 的导数等于零，解出 $w_0$ ，可得

$$w_0=\frac {1}{N}\sum_{i=1}^{N}t^{(i)}-\sum_{j=1}^{M-1}w_j\cdot\sum_{i=1}^{N}\phi_j(\boldsymbol x^{(i)})\tag {31}$$

&emsp;&emsp;可以看出偏置 $w_0$ 补偿了训练数据⽬标值的平均值 $\frac {1}{N}\sum_{i=1}^{N}t^{(i)}$ 与基函数的值的平均值的加权求和 $\sum_{i=1}^{N}\phi_j(\boldsymbol x^{(i)})$ 之间的差。

&emsp;&emsp;回归第一篇文章内容，我们也可以关于噪声精度参数 $\beta$ 最⼤化似然函数

$$\frac{1}{\beta^\star} =\frac{1}{N} \sum_{i=1}^{N} \left \{ t^{(i)}-{\boldsymbol{w}^{\star}}^T\phi(\boldsymbol x^{(i)}) \right \}^2\tag {32}$$

&emsp;&emsp;因此我们看到噪声精度的倒数由⽬标值在回归函数周围的残留⽅差 (residual variance) 给出。