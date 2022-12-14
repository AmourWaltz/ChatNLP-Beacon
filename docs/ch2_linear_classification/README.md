# 2 线性分类 Linear Classification

&emsp;&emsp;与线性回归相似，一种最简单的分类模型就是直接用一个线性函数进行分类，首先考虑二分类的情形，我们构造一个输入向量的线性函数 $y(\boldsymbol x)=\boldsymbol w^T\boldsymbol x+w_0$ ，对于输入向量 $\boldsymbol x$ ，如果 $y(\boldsymbol x)\geq0$ ，那么就被分到 $C_1$ 类，否则分到 $C_2$ 类，此时决策面由 $y(\boldsymbol x)=0$ 确定。
        
## 2.1 线性判别分析 Linear Discriminate Analysis

> 参考： PRML 4.1 节；面试考点总结如下
> 1. 。
> 2. 。

&emsp;&emsp;对于任意两个在决策面上的点 $\boldsymbol x_A$ 和 $\boldsymbol x_B$ ，有 $\boldsymbol w(\boldsymbol x_A-\boldsymbol x_B)=0$ ，所以向量 $\boldsymbol w$ 正交于决策面。对于任意一点 $\boldsymbol x$ ，在 $\boldsymbol w$ 方向上的投影代表了原点到决策面的垂直距离，即

$$\frac{\boldsymbol w^T\boldsymbol x}{\left \| \boldsymbol w \right \| }=-\frac{w_0}{\left \| \boldsymbol w \right \| }\tag 1$$

&emsp;&emsp;可见 $\boldsymbol w$ 和 $w_0$ 分别决定了决策面的方向和位置。对于任意一点 $\boldsymbol x$ ，我们将其投影至决策面上的一点 $\boldsymbol x_\bot$ ，这样我们就可以将其写成两个向量之和的形式

$$\boldsymbol x=\boldsymbol x_\bot+r\frac{\boldsymbol w}{\left \| \boldsymbol w \right \| }\tag 2$$
在两边同乘 $\boldsymbol w$ 并加上 $w_0$ ，由于 $\boldsymbol x_\bot$ 在决策面上，因此可以得到 $r=\frac{f(\boldsymbol x)}{\left \| \boldsymbol w \right \| }$ ，如下图所示。

<div align=center>
<img src="images/2_1_lda1.png" width="80%"/>
</div>

&emsp;&emsp;关于多分类的线性判别函数，我们可以引入一个由 $K$ 个线性判别函数组成的 $K$ 类判别函数 $y_k(\boldsymbol x)=\boldsymbol w_k^T\boldsymbol x+w_k0$ ，对于任意一点 $\boldsymbol x$ ，如果对于所有的 $i\ne j$ 都有 $y_k(\boldsymbol x)>y_j(\boldsymbol x)$ ，那么就把它分到 $C_k$ ，此时 $C_k$ 与 $C_j$ 的决策面为 $y_k(\boldsymbol x)=y_j(\boldsymbol x)$ ，对应于一个 $D-1$ 维超平面，形式为 $(\boldsymbol w_k-\boldsymbol w_j)^T\boldsymbol x+(w_{k0}-w_{j0})=0$ ，具有二分类决策面类似的性质。并且这样的决策区域都是单连通且凸的。
考虑决策区域 $\mathcal R_k$ 内任意两点 $\boldsymbol x_A,\boldsymbol x_B$ ，这两点连成的线段上的任意一点 $\boldsymbol x_C$ 都可以表示成

$$\boldsymbol x_C=\lambda\boldsymbol x_A+(1-\lambda)\boldsymbol x_B\tag 3$$
其中 $0<\lambda<1$ ，根据线性函数的基本性质，很容易得出 $\boldsymbol x_C$ 也属于 $\mathcal R_k$ 。
这种线性判别函数与最简单的线性回归形式基本类似，所以我们先考虑用最小平方训练模型，将所有偏置和参数向量聚集在一起，有 $y( {\boldsymbol X})={\boldsymbol  W}^T {\boldsymbol X}+\boldsymbol {w_0}$ ，其中 $\boldsymbol W^{D\times K}$ 代表 $K$ 个种类的参数， $\boldsymbol w_0$ 是所有偏置项， $\boldsymbol X^{D\times N}$ 是 $N$ 个训练集数据的矩阵，$\boldsymbol T^{K\times N}$ 是目标向量矩阵，平方和误差函数可以写成 $\boldsymbol w_0$

$$E({\boldsymbol X})=\frac{1}{2}\mathrm{Tr} \left \{ ({\boldsymbol W}^T {\boldsymbol X} +\boldsymbol {1w_0}-\boldsymbol T)^T ( {\boldsymbol W}^T{\boldsymbol X}  +\boldsymbol {1w_0}-\boldsymbol T) \right \}\tag 4$$

&emsp;&emsp;使用解析法可以计算出 ${\boldsymbol W}=\tilde {\boldsymbol X}^\dag \tilde {\boldsymbol T}$ ，其中 $\tilde {\boldsymbol X}^\dag$ 是 $\tilde {\boldsymbol X}$ 的伪逆矩阵， $\tilde {\boldsymbol X}=\boldsymbol X-\boldsymbol {1\bar{x}}^T$ ， $\boldsymbol {\bar x}^{D\times 1}$ 是训练数据的均值向量， $\tilde {\boldsymbol T}=\boldsymbol T-\boldsymbol {1\bar{t}}^T$ 。多⽬标变量的最⼩平⽅解有⼀个性质，如果训练集⾥的每个⽬标向量都满⾜某个线性限制

$$\boldsymbol a^T \boldsymbol t_n+b=0\tag 5$$
其中 $\boldsymbol a$ 和 $\boldsymbol b$ 为常数，那么对于任何 $\boldsymbol x$ 值，模型的预测也满⾜同样的限制，即

$$\boldsymbol a^T y(\boldsymbol x)+b=0\tag 6$$
将解析解直接代入，可得

$$y(\boldsymbol x^\star)=\boldsymbol W^T\boldsymbol x^\star+\boldsymbol w_0=\boldsymbol {\bar t}-\tilde{\boldsymbol T}^T(\boldsymbol {\tilde X}^\dag)^T(\boldsymbol x^\star-\boldsymbol {\bar x})\tag 7$$
其中 $\boldsymbol w_0$ 由公式 (6) 的一阶导置零求得。对公式 (8) 两边都乘以 $\boldsymbol a^T$ 并且根据 $\tilde {\boldsymbol T}=\boldsymbol T-\boldsymbol {1\bar{t}}^T$ 可得结果为 $-b$ ，证明公式 (7) 成立。
最⼩平⽅法对于离群点缺乏鲁棒性，而且由于最⼩平⽅法对应于⾼斯条件分布假设下的最⼤似然法，多目标向量显然不是服从一个高斯分布，线性分类模型需要使用其他方法训练参数。

## 2.2 Fisher分类器 Fisher Classifier

> 参考： PRML 4.1 节；面试考点总结如下
> 1. 。
> 2. 。

&emsp;&emsp;如果从维度降低的⾓度考察线性分类模型，对于二分类问题，可以看做针对输入向量 $\boldsymbol x$ 在一维空间的投影

$$y=\boldsymbol w^T\boldsymbol x\tag 8$$

&emsp;&emsp;线性判别函数等价于我们在 $y$ 上设置⼀个阈值，然后把 $y\geq-w_0$ 的样本分为 $C_1$ 类，把其余的样本分为 $C_2$ 类。 但是⼀维投影会造成相当多的信息丢失，因此在原始 $D$ 维空间能够完美分离的样本在⼀维空间中可能会相互重叠。Fisher 分类器提出的思想是最⼤化⼀个函数，能够让类均值的投影分开得较⼤，同时让每个类别内部的⽅差较⼩，从⽽最⼩化类别的重叠。仍然考虑上述二分类问题，包含 $N_1$ 个 $C_1$ 类的点和 $N_2$ 个 $C_2$ 类的点，两类的均值分别为 $\boldsymbol m_1=\frac{1}{N_1}\sum_{n\in C_1}\boldsymbol x_n$, $\boldsymbol m_2=\frac{1}{N_2}\sum_{n\in C_2}\boldsymbol x_n$ ，最简单的度量类间区分程度的⽅式就是类别均值投影之后的距离。取单位长度向量 $\boldsymbol w$ 并向其投影，取下式的最大值即可

$$m_2-m_1=\boldsymbol w^T(\boldsymbol m_2-\boldsymbol m_1)\tag 9$$
其中 $m_k=\boldsymbol w^T\boldsymbol m_k$ ，是类别 $C_k$ 的均值投影。但是如下图所示，当投影到一维空间时，就有了⼀定程度的重叠。如果类概率分布的协⽅差矩阵与对角化矩阵差距较⼤，即类内方差在各个方向上差异较大，那么这种问题就会出现。如下图左图所示

<div align=center>
<img src="images/2_2_fisher1.png" width="80%"/>
</div>

&emsp;&emsp;我们将投影在一维空间的类 $C_k$ 的类内方差记作

$$s_k^2=\sum_{n\in C_k}(y_n-m_k)^2\tag{10}$$ 
&emsp;&emsp;我们把整个数据集的总的类内方差定义为 $s_1^2+s_2^2$ ，Fisher 准则根据类间⽅差和类内⽅差的⽐值定义，即

$$J(\boldsymbol w)=\frac{(m_2-m_1)^2}{s_1^2+s_2^2}=\frac{\boldsymbol w^T\boldsymbol S_B\boldsymbol w}{\boldsymbol w^T\boldsymbol S_W\boldsymbol w}\tag {11}$$

&emsp;&emsp;分别用 $\boldsymbol S_B=(\boldsymbol m_2-\boldsymbol m_1)(\boldsymbol m_2-\boldsymbol m_1)^T$ 和 $\boldsymbol S_w=\sum_{n\in C_1}( \boldsymbol x_n-\boldsymbol m_1)(\boldsymbol x_n-\boldsymbol m_1)^T + \sum_{n\in C_2}( \boldsymbol x_n-\boldsymbol m_2)(\boldsymbol x_n-\boldsymbol m_2)^T$ 表示类间 (between-class) 协方差矩阵和类内 (within-class) 协方差矩阵。对公式 (12) 求导，发现 $J(\boldsymbol w)$ 取极大值的条件为

$$(\boldsymbol w^T\boldsymbol S_B\boldsymbol w)\boldsymbol S_W\boldsymbol w=(\boldsymbol w^T\boldsymbol S_W\boldsymbol w)\boldsymbol S_B\boldsymbol w\tag {12}$$
考虑只是一维投影，因此我们只关心 $\boldsymbol w$ 的方向，忽略标量因子 $\boldsymbol w^T\boldsymbol S_B\boldsymbol w$ 和 $\boldsymbol w^T\boldsymbol S_W\boldsymbol w$ ，而 $\boldsymbol S_B\boldsymbol w$ 总是在 $\boldsymbol m_2-\boldsymbol m_1$ 的方向上，对公式 (13) 两侧同乘以 $\boldsymbol S_W^{-1}$ ，可得

$$\boldsymbol w\propto \boldsymbol S_W^{-1}(\boldsymbol m_2-\boldsymbol m_1)\tag {13}$$

&emsp;&emsp;如上图右图就是最大化类间方差与类内方差比值的结果。如果类内协⽅差矩阵是各向同性的，即各个方向方差一致，协方差为正实数与单位矩阵相乘，从⽽ $\boldsymbol S_W$ 正⽐于单位矩阵，那么 $\boldsymbol w$ 正⽐于类均值的差。
最⼩平⽅⽅法确定线性判别函数的⽬标是使模型的预测尽可能地与⽬标值接近。相反，Fisher 判别准则的⽬标是使输出空间的类别有最⼤的区分度。这两种方法也并非毫无关系，我们可以通过修改目标向量建立二者的联系，对于⼆分类问题，Fisher 准则可以看成最⼩平⽅的⼀个特例。对于 $C_1$ 类，我们令其目标值为 $\frac{N}{N_1}$ ，而 $C_2$ 类为 $\frac{N}{N_2}$ ， $N_1,N_2$ 分别为类别 $C_1,C_2$ 数据点的个数，此时平方误差函数可以写成

$$E=\frac{1}{2}\sum_{n=1}^{N}(\boldsymbol w^T\boldsymbol x_n+w_0-t_n)^2\tag {14}$$

&emsp;&emsp;令 $E$ 关于 $w_0$ 和 $\boldsymbol w$ 的导数为零，得

$$\sum_{n=1}^{N}(\boldsymbol w^T\boldsymbol x_n+w_0-t_n)=0\tag {15}$$ 

$$\sum_{n=1}^{N}(\boldsymbol w^T\boldsymbol x_n+w_0-t_n)\boldsymbol x_n=0\tag {16}$$

&emsp;&emsp;先求解公式 (16)，可得偏置表达式

$$w_0=-\boldsymbol w^T\boldsymbol m\tag {17}$$ 
其中 $\sum_{n=1}^{N}t_n=N_1\frac{N}{N_1}-N_2\frac{N}{N_2}=0, \boldsymbol m=\frac {1}{N}(N_1\boldsymbol m_1+N_2\boldsymbol m_2)$ ，将这两个式子以及公式 (18) 代入 (17) 通过繁琐但并不复杂的运算，可得

$$(\boldsymbol S_W+\frac{N_1N_2}{N}\boldsymbol S_B)\boldsymbol w=N(\boldsymbol m_2-\boldsymbol m_1)\tag{18}$$
仍然可以得到与公式 (14) 类似的结果，因为最小二乘法的计算本就包含了类内方差的因子，我们只不过通过构造目标值引入类间方差从而得到 Fisher 分类器的结果。

## 2.3 感知器算法 Perceptron Algorithm

> 参考： PRML 4.1 节；面试考点总结如下
> 1. 。
> 2. 。

&emsp;&emsp;线性判别模型的另⼀个例⼦是感知器算法 (perceptron algorithm)，它对应⼀个⼆分类模型，输⼊向量 $\boldsymbol x$ ⾸先使⽤⼀个固定的⾮线性变换得到⼀个特征向量 $\phi(\boldsymbol x)$ ，然后被⽤于构造⼀个线性模型，形式为

$$y(\boldsymbol x) =f(\boldsymbol w^T\phi(\boldsymbol x))\tag 1$$
其中非线性激活函数 $f(\cdot)$ 是一个阶梯函数，形式为

$$f(a)=\left\{\begin{matrix} +1,a\ge 0 \\ -1,a<0 \end{matrix}\right\}$$

&emsp;&emsp;向量 $\phi(\boldsymbol x)$ 通常包含⼀个偏置分量 $\phi_0(\boldsymbol x)=1$ 。我们仍可以使用误差最小化来确定感知器的参数 $\boldsymbol w$ ，可以将误分类的数据作为误差函数，但这样做会使误差函数变为 $\boldsymbol w$ 的分段常函数，无法正常使用梯度下降算法。考虑其他的误差函数，根据二分类的线性判别函数，我们可以寻找一个权向量 $\boldsymbol w$ 使得对 $C_1$ 类都有 $\boldsymbol w^T\phi(\boldsymbol x)>0$ ，对于 $C_2$ 类有 $\boldsymbol w^T\phi(\boldsymbol x)<0$ ，同时由于目标值正好异号，可以利用所有误分类的模式构造如下误差函数

$$E(\boldsymbol w)=-\sum_{n\in \mathcal M}\boldsymbol w^T\phi_nt_n\tag 3$$

&emsp;&emsp;对这个误差函数使⽤随机梯度下降算法，权向量 $\boldsymbol w$ 的变化为

$$\boldsymbol w^{(\tau+1)}=\boldsymbol w^{(\tau)}-\eta\nabla E(\boldsymbol w)=\boldsymbol w^{(\tau)}+\eta\phi_n t_n\tag 4$$
其中 $\eta$ 是学习率参数， $\tau$ 是迭代次数。如果我们将 $\boldsymbol w$ 乘以⼀个常数，那么感知器函数 $y=f(\boldsymbol x)$ 不变，因此我们可令学习率参数 $\eta$ 等于 1 ⽽不失⼀般性。我们对训练模式进⾏循环处理，对于每个数据点 $\boldsymbol x_n$ 计算感知器函数，如果模式正确分类，那么权向量保持不变，⽽如果模式被错误分类，那么对于类别 $C_1$ ，我们把向量 $\phi(\boldsymbol x_n)$ 加到当前权向量 $\boldsymbol w$ 的估计值上， ⽽对于类别 $C_2$ ，我们从 $\boldsymbol w$ 中减掉向量 $\phi(\boldsymbol x_n)$ 。
在线性判别函数一节中已经讲过，参数 $\boldsymbol w$ 实际上决定了决策界面的方向，感知器算法的迭代过程可以看做是将决策面朝着错误分类数据点的移动，如下图所示

<div align=center>
<img src="images/2_3_mlp1.png" width="70%"/>
</div>

&emsp;&emsp;如果我们考虑感知器学习算法中⼀次权值更新的效果，可以看到，⼀个误分类模式对于误差函数的贡献会逐渐减⼩，如下

$$-\boldsymbol w^{(\tau+1)T}\phi_nt_n=-\boldsymbol w^{(\tau)T}\phi_nt_n-(\phi_nt_n)^{T}\phi_nt_n<-\boldsymbol w^{(\tau)T}\phi_nt_n\tag 5$$

&emsp;&emsp;当然，这并不表明其他的误分类数据点对于误差函数的贡献会减⼩。此外，权向量的改变会使得某些之前正确分类的样本变为误分类。因此感知器学习规则并不保证在每个阶段都会减⼩整体的误差函数。如果数据点确实线性可分，那么感知器算法在经过一定迭代步骤一定可以找到精确解，但是所需要的步骤往往很大，并且在达到收敛状态之前，我们不能够区分不可分问题与缓慢收敛问题。即使数据集是线性可分的，也可能有多个解，并且最终收敛的解依赖于参数的初始化以及数据点出现的顺序。而对于线性不可分的数据集，感知器算法永远不会收敛。

&emsp;&emsp;本文讲述了三种线性分类的判别函数，下篇文章将从概率的角度观察分类问题。

## 2.4 判别式 Logistic 回归 Discrminate Logistic Regression

> 参考： PRML 4.1 节；面试考点总结如下
> 1. 。
> 2. 。

&emsp;&emsp;上一篇文章讨论了几种线性判别函数，这些判别函数都是将分类的推断及决策合而为一的，如果分成两个阶段讨论，针对决策，很自然联想使用概率表示分类的可能性，所以我们可以将分类结果映射到区间 $[0,1]$ 上再决策。在原始输入空间中有很多已知类型的数据点，我们需要建立合适的模型对这些点进行区分。最简单的情况就是这些数据点是线性可分的，利用上一章的线性判别函数就能区分，但是在一些复杂空间，类条件概率密度 $p(\boldsymbol x|C_k)$ 有相当大的重叠，也就是对于某些数据点不能简单粗暴的认为属于或不属于某一类，我们需要一种更加细致定量的概率方法来表示，所以才引入对后验概率的建模。在此引入逻辑回归 (logistic regression) ，将输出映射至 $[0,1]$ 的概率空间上，logistic 函数一般形式如下，后续讨论会发现 logistic 回归一些特性很有助于我们分析分类问题。

$$\sigma(a)=\frac{1}{1+\exp(-a)}\tag 1$$

&emsp;&emsp;我们平常所熟知的利用 logistic 回归做分类一般都是判别式模型。上文已经讨论过，判别式模型就是根据直接对后验概率 $p(C_k|\boldsymbol x)$ 精确建模，这种方法很直观，准确率高，可直接学习求得参数，简化学习。

&emsp;&emsp;在线性回归一节中提到，对输入向量做非线性变换可以提取出某些高级特征，所以我们可以引入一些非线性函数做分类。⾮线性变换不会消除数据类型重叠，甚至会增加重叠的程度，但恰当地选择⾮线性变换能够让后验概率的建模过程更简单。在此选择 logistic sigmoid 函数，下图给出了这个函数的图像。“sigmoid” 的意思是 “S形”。这种函数有时被称为“挤压函数”，因为它把整个实数轴映射到了⼀个有限的区间中

<div align=center>
<img src="images/2_4_lg1.png" width="80%"/>
</div>

&emsp;&emsp;它满⾜下⾯的对称性

$$\sigma(-a)=1-\sigma(a)\tag 2$$

#### 最大似然参数估计 Maximum Likelihood Parameter Estimation

&emsp;&emsp;我们现在使⽤最⼤似然⽅法来确定 logistic 回归模型的参数。为了完成这⼀点，我们要使⽤ logistic sigmoid 函数的导数，它可以很⽅便地使⽤sigmoid函数本⾝表示如下：

$$\frac{d\sigma}{da}=\sigma(1-\sigma)\tag 3$$

&emsp;&emsp;先考虑最简单的二分类情形，我们定义输入 $\boldsymbol x$ 在类 $C_1$ 上的后验概率为

$$p(C_1|\boldsymbol x)=y(\boldsymbol x, \boldsymbol w)=\sigma (\boldsymbol w^T\boldsymbol x)\tag 4$$

&emsp;&emsp;那么 $p(C_2|\boldsymbol x)=1-p(C_1|\boldsymbol x)$ ，此时整体分布恰好是一个伯努利分布 (Bernoulli distribution)。将上式合并，取类别标签 $t\in\left \{1,0 \right \}$ 分别对应 $C_1,C_2$ ，那么 $p(C_{k\in\left \{ 1,2\right \}}|\boldsymbol x)=y^{t}(1-y)^{1-t}$ 。更一般地，对于⼀个数据集 $\mathcal{D}$ ，包含 $N$ 个相互独立的输入数据 $\boldsymbol x$ 和标签 $\boldsymbol t=\left \{ t_1, t_2,...,t_N \right \}$ ，其中 $t_n\in\left \{0,1 \right \} ， y_n=p(C_1|\boldsymbol x_n, \boldsymbol w)=\sigma(\boldsymbol w^T\boldsymbol x_n)$ ，此时似然函数可以写成

$$p(\boldsymbol t|\boldsymbol w)=\prod_{n=1}^{N}y_n^{t_n}(1-y_n^{1-t_n})\tag 5$$

&emsp;&emsp;与之前⼀样，我们可以通过取似然函数负对数的⽅式，定义⼀个误差函数，由此引入交叉熵 (cross-entropy) 误差函数，形式为

$$E(\boldsymbol w)=-\ln{p(\boldsymbol t|\boldsymbol w)}=-\sum_{n=1}^{N}\left \{ t_n\ln y_n +(1-t_n)\ln{(1-y_n)} \right \}\tag 6$$
两侧取误差函数关于 $\boldsymbol w$ 的梯度，结合公式 (3) 将涉及到 logistic sigmoid 的导数的因⼦消去，我们有

$$\nabla E(\boldsymbol w)=\sum_{n=1}^{N}(y_n-t_n)\boldsymbol x_n\tag 7$$

&emsp;&emsp;可见数据点 $n$ 对梯度的贡献为⽬标值和模型预测值之间的误差 $y_n-t_n$ 与输入向量 $\boldsymbol x_n$ 相乘，这种函数形式与线性回归模型中的平⽅和误差函数的梯度形式完全相同。 

&emsp;&emsp;最⼤似然法对于线性可分的数据集会产⽣严重的过拟合现象，这是由于最⼤似然解出现在超平⾯对应于 \sigma=0.5 一阶导数最大的情况，由于线性可分的数据集具有如下性质：

$$\boldsymbol w^T\boldsymbol x_n=\left\{\begin{matrix} \ge 0,~~~ if~~t_n=1 \\ <1, ~~~ otherwise \end{matrix}\right\}$$
等价于 $\boldsymbol w^T\boldsymbol x=\boldsymbol 0$ ，最⼤似然解把数据集分成两类，一阶导为零等价于 $y_n=\sigma(\boldsymbol w^T\phi_n)=t_n$ ， $\boldsymbol w$ 趋向于⽆穷⼤。这种情况下，logistic sigmoid 函数在特征空间中变得⾮常陡峭，对应于⼀个跳变的阶梯函数，使得每⼀个来⾃类别 $k$ 的训练数据都被赋予⼀个后验概率 $p(C_k|\boldsymbol x)=1$ 。最⼤似然法⽆法区分某个解优于另⼀个解，并且在实际应⽤中哪个解被找到将会依赖于优化算法的选择和参数的初始化（大多数模型都这样）。即使训练数据远大于参数数量，只要数据线性可分，这个问题就会出现。可以通过引⼊类先验概率，然后寻找 $\boldsymbol w$ 的最大后验解，或者给误差函数增加⼀个正则化项， 这种奇异性就可以被避免。

#### 牛顿法-迭代重加权最⼩平⽅ Newton's Method - Iteratively Reweighted Least Squares

&emsp;&emsp;在线性回归第二篇文章中曾讨论了利用误差函数二阶导更新梯度的牛顿法 (Newton-Raphson)。线性回归的最小平方误差函数是关于参数 $\boldsymbol w$ 的二次凸函数，最大似然解具有解析解，对应于在牛顿法中误差函数是一个正定二次函数，应用一次就能跳到最小值点；logistic 回归模型的误差函数中不具备解析解，但它是一个凸函数仍可以局部近似成正定二次函数，因此也存在唯一最小值。牛顿法对权值的更新形式为

$$\boldsymbol w^{\mathrm{new}}=\boldsymbol w^{\mathrm{old}}-\boldsymbol H^{-1}\nabla E(\boldsymbol w)\tag 9$$ 
其中 $\boldsymbol H$ 是一个 Hessian 矩阵，它的元素由 $E(\boldsymbol w)$ 关于 $\boldsymbol w$ 的二阶导构成。将牛顿法应用到交叉熵误差函数上，误差函数的梯度和 Hessian 矩阵为

$$\nabla E(\boldsymbol w)=\sum_{n=1}^{N}(y_n-t_n)\boldsymbol x_n=\boldsymbol X^T(\boldsymbol y-\boldsymbol t)\tag{10}$$

$$\boldsymbol H=\nabla\nabla E(\boldsymbol w)=\sum_{n=1}^{N}y_n(1-y_n)\boldsymbol x_n\boldsymbol x_n^T=\boldsymbol X^T\boldsymbol R\boldsymbol X\tag{11}$$

&emsp;&emsp;$y_n(1-y_n)$ 由公式 (3) 所得，仔细观察会发现，如果把误差函数换成最小平方误差，Hessian 矩阵就可以消掉 $\boldsymbol R$ ，代入公式 (9)，就可以得出最小平方误差的解析解形式，对应于第三篇文章解析法的基函数取 $\phi(\boldsymbol x)=\boldsymbol x$ ，所以解析法和牛顿法在二次凸函数上的应用原理是相同的。再回到交叉熵函数，我们引入一个 $N\times N$ 的对角矩阵 $\boldsymbol R$ ，元素为 $R_{nn}=y_n(1-y_n)$ ，看到 Hessian 矩阵不再是常量，⽽是通过权矩阵 $\boldsymbol R$ 仍含有 $\boldsymbol w$ ，对应于误差函数不是⼆次函数的事实。

&emsp;&emsp;由于 $0<y_n<1$ ，因此对于任意向量 $\boldsymbol u$ 都有 $\boldsymbol u^T\boldsymbol H\boldsymbol u>0$ ，因此 Hessian 矩阵 $\boldsymbol H$ 是正定的，误差函数是 $\boldsymbol w$ 的⼀个凸函数，从⽽有唯⼀的最⼩值。这样 logistic 回归模型更新公式就如下

$$\begin{align} \boldsymbol w^{\mathrm {new}}&=\boldsymbol w^{\mathrm {old}}-(\boldsymbol X^T\boldsymbol R\boldsymbol X)^{-1}\boldsymbol X^T(\boldsymbol y-\boldsymbol t)\\ &=(\boldsymbol X^T\boldsymbol R\boldsymbol X)^{-1}\left \{ \boldsymbol X^T\boldsymbol R\boldsymbol X\boldsymbol w^{\mathrm {old}}-\boldsymbol X^T(\boldsymbol y-\boldsymbol t) \right \} \\ &=(\boldsymbol X^T\boldsymbol R\boldsymbol X)^{-1}\boldsymbol X^T\boldsymbol R\boldsymbol z \end{align}$$

&emsp;&emsp;由于权矩阵 $\boldsymbol R$ 不是常量，⽽是依赖于参数向量 $\boldsymbol w$ ， 因此我们必须迭代地应⽤牛顿法的解，每次使⽤新的权向量 $\boldsymbol w$ 计算⼀个修正的权矩阵 $\boldsymbol R$ ，这个算法也被称为迭代重加权最⼩平⽅ (iterative reweighted least squares)。logistic 回归模型 t 的均值和⽅差分别为

$$\mathbb{E}\left [ t \right ] =\sigma(\boldsymbol x)=y\tag{13}$$

$$\mathrm{var}\left [ t \right ] =\mathbb{E}\left [ t^2 \right ] -\mathbb{E}\left [ t \right ] ^2=\sigma(\boldsymbol x)-\sigma(\boldsymbol x)^2=y(1-y)\tag{14}$$
$t\in\left \{ 0,1 \right \}$ 时， $t^2=t$ ，所以对角矩阵 $\boldsymbol R$ 也可以看成⽅差。我们可以把迭代重加权最⼩平⽅看成变量空间 $a_n=\boldsymbol w^T\boldsymbol x_n$ 的线性问题的解。这样， $\boldsymbol z$ 的第 $n$ 个元素 $\boldsymbol z_n$ 就可以简单看成这个空间中的有效⽬标值。这里引入 logistic sigmoid 的反函数

$$a=\ln{\frac{\sigma}{1-\sigma}}\tag{15}$$

&emsp;&emsp;$\boldsymbol z_n$ 可以通过对当前操作点 $\boldsymbol w^{\mathrm{old}}$ 附近的 logistic sigmoid 函数的局部线性近似的⽅式得到。

$$a_n({\boldsymbol w})\simeq a_n({\boldsymbol w^{\mathrm{old}}})+\frac{\mathrm{d} a_n}{\mathrm{d} y_n}|_{\boldsymbol w^{\mathrm{old}}}(t_n-y_n)=\boldsymbol x^T\boldsymbol w^{\mathrm{old}}-\frac{y_n-t_n}{y_n(1-y_n)}=z_n\tag{16}$$

## 2.5 生成式 Logistic 回归 Generative Logistic Regression

> 参考： PRML 4.1 节；面试考点总结如下
> 1. 。
> 2. 。

&emsp;&emsp;现在讨论分类问题的概率生成式模型，生成式模型对类条件概率密度 $p(\boldsymbol x|C_k)$ 和类先验概率 $p(C_k)$ 建模，然后通过贝叶斯定理计算后验概率 $p(C_k|\boldsymbol x)$ 。不难看出，生成模型只需进行统计技术来获得模型，且对噪声有更好的鲁棒性；但由于需要人为设定先验分布，因此生成模型准确率并不高。
同样先只考虑二分类情形，我们直接应用贝叶斯定理对类别 C_1 的后验概率建模

$$p(C_1|\boldsymbol x)=\frac{p(\boldsymbol x|C_1)p(C_1)}{p(\boldsymbol x|C_1)p(C_1)+p(\boldsymbol x|C_2)p(C_2)}=\frac{1}{1+\exp{(-a)}}=\sigma(a)\tag 1$$
其中定义了 $a=\ln\frac{p(\boldsymbol x|C_1)p(C_1)}{p(\boldsymbol x|C_2)p(C_2)}$ 。虽然我们只是把后验概率写成了⼀个等价的形式，引入logistic sigmoid 函数似乎没有意义，然⽽，假设 $a(\boldsymbol x)$ 的函数形式相当简单，如上一节判别式模型所讲，考虑 $a(\boldsymbol x)$ 是 $\boldsymbol x$ 的线性函数的情形，这种情况下，后验概率由⼀个通⽤的线性模型确定，下文会予以证明。对于 $K>2$ 的情形，有

$$p(C_k|\boldsymbol x)=\frac{p(\boldsymbol x|C_k)p(C_k)}{ {\textstyle \sum_{j}^{}} p(\boldsymbol x|C_j)p(C_j)}=\frac{\exp(a_k)}{ {\textstyle \sum_{j}^{}} \exp(a_j)}\tag 2$$
其中 $a_k=\ln p(\boldsymbol x|C_k)p(C_k)$ ，上式也被称为归⼀化指数 (normalized exponential)，可以被当做 logistic sigmoid 函数对于多类情况的推⼴，也就是我们很熟悉的 Softmax 函数，因为它表示 “max” 函数的⼀个平滑版本，因为如果对所有的 $j\ne k$ 都有 $a_k\gg a_j$ ，那么 $p(C_k|\boldsymbol x)\simeq 1$ 且 $p(C_j|\boldsymbol x)\simeq 0$ 。

&emsp;&emsp;假设类条件概率密度是⾼斯分布，然后求解后验概率。这种生成式算法在 CS229 中也称作高斯判别模型 (Gaussian discriminative model)。⾸先，我们假定所有类别的协⽅差矩阵相同，这样类别 $C_k$ 的类条件概率为

$$p(\boldsymbol x|C_k)=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{|\boldsymbol\Sigma|^{\frac{1}{2}}}\exp\left \{ -\frac{1}{2}(\boldsymbol x-\boldsymbol \mu_k)^T \boldsymbol\Sigma^{-1} (\boldsymbol x-\boldsymbol \mu_k) \right \}\tag 3$$

&emsp;&emsp;考虑二分类情形，我们直接将上述类条件概率密度代入 $a=\ln\frac{p(\boldsymbol x|C_1)p(C_1)}{p(\boldsymbol x|C_2)p(C_2)}$ ，假设类概率的协⽅差矩阵相同，可消除⾼斯概率密度的指数项中 $\boldsymbol x$ 的⼆次型。最后可以将公式化简为 $p(C_1|\boldsymbol x)=\sigma(\boldsymbol w^T\boldsymbol x+w_0)$ ，其中

$$\boldsymbol w=\boldsymbol \Sigma^{-1}(\boldsymbol \mu_1-\boldsymbol \mu_2)\tag 4$$

$$w_0=-\frac{1}{2}\boldsymbol \mu_1^T\boldsymbol \Sigma^{-1}\boldsymbol \mu_1+\frac{1}{2}\boldsymbol \mu_2^T\boldsymbol \Sigma^{-1}\boldsymbol \mu_2+\ln\frac{p(C_1)}{p(C_2)}\tag 5$$
从⽽得到了参数为 $\boldsymbol x$ 的线性函数的 logistic sigmoid 函数，最终求得的决策边界对应于后验概率 $p(C_k|\boldsymbol x)$ 为常数的决策⾯，因此由 $\boldsymbol x$ 的线性函数给出，从⽽决策边界在输⼊空间是线性的。先验概率密度只出现在偏置参数 $w_0$ 中，因此先验的改变的效果是平移决策边界，即平移后验概率中的常数轮廓线。

&emsp;&emsp;下图给出了⼆维输⼊空间 $\boldsymbol x$ 的结果，左图给出了两个类别的类条件概率密度，分别⽤红⾊和蓝⾊表示。右图给出了对应后验概率分布 $p(C_1|\boldsymbol x)$ ，它由 $x$ 的线性函数的 logistic sigmoid 函数给出。右图的曲⾯的颜⾊中，红⾊所占的⽐例由 $p(C_1|\boldsymbol x)$ 给出，蓝⾊所占的⽐例由 $p(C_2|\boldsymbol x)=1-p(C_1|\boldsymbol x)$ 给出。

<div align=center>
<img src="images/2_5_lg1.png" width="80%"/>
</div>

#### 最大似然参数估计 Maximum Likelihood Parameter Estimation

&emsp;&emsp;⼀旦我们具体化了类条件概率密度 $p(\boldsymbol x|C_k)$ 的参数化函数形式，我们就能够使⽤最⼤似然法确定参数的值，以及先验类概率 $p(C_k)$ 。这需要数据集由观测 $\boldsymbol x$ 以及对应的类别标签组成。
依然只考虑二分类的情形，每个类别都有⼀个⾼斯类条件概率密度，且协⽅差矩阵相同。给定包含 $N$ 个数据点的数据集 $\mathcal D=\left \{ \boldsymbol x_n,t_n \right \}$ 。取类别标签 $t\in\left \{1,0 \right \}$ 分别对应 $C_1,C_2$。我们把先验概率记作 $p(C_1)=\pi$ ，从而 $p(C_2)=1-\pi$ 。对于⼀个来⾃类别 $C_1$ 的数据点 $\boldsymbol x_n$ ，有

$$p(\boldsymbol x_n,C_1)=p(C_1)p(\boldsymbol x_n|C_1)=\pi\mathcal N(\boldsymbol x_n|\boldsymbol \mu_1, \boldsymbol \Sigma)\tag 6$$

&emsp;&emsp;于是似然函数为

$$p(\boldsymbol t,\boldsymbol X|\pi,\boldsymbol \mu_1,\boldsymbol \mu_2,\boldsymbol \Sigma )=\prod_{n=1}^{N} [ \pi\mathcal N(\boldsymbol x_n|\boldsymbol \mu_1, \boldsymbol \Sigma)]^{t_n}[(1-\pi)\mathcal N(\boldsymbol x_n|\boldsymbol \mu_2, \boldsymbol \Sigma)]^{1-t_n}\tag 7$$

&emsp;&emsp;仍然最大化负对数似然函数，依次对四个参数 $\pi,\boldsymbol \mu_1,\boldsymbol \mu_2,\boldsymbol \Sigma$ 求偏导，先考虑关于 $\pi$ 的最大化，对数似然函数与 $\pi$ 相关的项为 $\sum_{n=1}^{N}\left \{ t_n\ln \pi+(1-t_n)\ln (1-\pi) \right \}$ ，令其关于 $\pi$ 的导数等于零，整理可得

$$\pi=\frac{1}{N}\sum_{n=1}^{N}t_n=\frac{N_1}{N_1+N_2}\tag 8$$
其中 $N_1,N_2$ 分别表示类别 $C_1,C_2$ 的数据点总数，因此， $\pi$ 的最⼤似然估计就是类别 $C_1$ 的点所占的⽐例，这与我们预期的相同。

&emsp;&emsp;接着考虑关于 $\boldsymbol \mu_1$ 的最⼤化。与之前⼀样，我们把对数似然函数中与 $\boldsymbol \mu_1$ 相关的量挑出来，即 $-\frac{1}{2}\sum_{n=1}^{N}t_n(\boldsymbol x_n-\boldsymbol \mu_1)^T\boldsymbol \Sigma^{-1}(\boldsymbol x_n-\boldsymbol \mu_1)$ ，令其关于 $\boldsymbol \mu_1$ 的导数等于零，整理可得

$$\boldsymbol \mu_1 =\frac{1}{N_1}\sum_{n=1}^{N}t_n\boldsymbol x_n\tag 9$$

&emsp;&emsp;这就是属于类别 $C_1$ 的输⼊向量 $\boldsymbol x_n$ 的均值。通过类似的推导，对应 $\boldsymbol \mu_2$ 的结果为

$$\boldsymbol \mu_2 =\frac{1}{N_2}\sum_{n=1}^{N}(1-t_n)\boldsymbol x_n\tag{10}$$

&emsp;&emsp;最后，考虑协⽅差矩阵 $\boldsymbol \Sigma$ 的最⼤似然解，选出与 $\boldsymbol \Sigma$ 相关的项 $-\frac{N}{2}\ln |\boldsymbol \Sigma|-\frac{N}{2}\mathrm{Tr}\left \{ \boldsymbol \Sigma^{-1}\boldsymbol S \right \}$ ，使⽤⾼斯分布的最⼤似然解的标准结果可得 $\boldsymbol \Sigma=\boldsymbol S$ ，其中

$$\boldsymbol S=\frac{N_1}{N}·\frac{1}{N_1}\sum_{n\in C_1}(\boldsymbol x-\boldsymbol \mu_1)(\boldsymbol x-\boldsymbol \mu_1)^T+\frac{N_2}{N}·\frac{1}{N_2}\sum_{n\in C_2}(\boldsymbol x-\boldsymbol \mu_2)(\boldsymbol x-\boldsymbol \mu_2)^T\tag{11}$$
它表示对⼀个与两类都有关系的协⽅差矩阵求加权平均。拟合类⾼斯分布的⽅法对于离群点并不鲁棒，因为⾼斯的最⼤似然估计是不鲁棒的。

#### 生成式模型与判别式模型 Generative Models and Discriminative Models

&emsp;&emsp;对于一般的线性判别模型，如果输入向量 $\boldsymbol x$ 具有 $M$ 维特征，那么这个模型有 $M$ 个可调节参数。对于上述二分类的生成式模型，如果我们使⽤最⼤似然⽅法调节⾼斯类条件概率密度，那么我们有 $2M$ 个参数来描述均值， 以及 $\frac{M(M+1)}{2}$ 个参数来描述协⽅差矩阵，仍然假设协⽅差矩阵相同，算上类先验，参数总数量为 $\frac{M(M+5)}{2}+1$ ，这随着 $M$ 的增长以⼆次的⽅式增长。这和判别式模型对于参数数量 $M$ 的线性依赖不同，对于大的 $M$ 值，生成式模型参数量通常很大。可以看出，生成式模型相比判别式模型，建模更加复杂，使用也比较受限。

&emsp;&emsp;通过公式 (17) 我们已经发现，逻辑回归的判别式模型和生成式模型有着相同的形式，但对于相同的数据集两种算法会给出不同的边界，有一个结论是：如果 $p(\boldsymbol x| C_k)$ 属于多元高斯分布且共享协方差矩阵 $\Sigma$ ，那么 $p(C_k|\boldsymbol x)$ 一定是逻辑函数，反之不成立。在这些假设都正确的前提下，生成式模型效果往往更好，特别是对于较少的训练集，也可以取得更好的效果。相反，判别式模型由于假设相对简单，主要靠大量数据集训练，因此对于错误的模型假设不那么敏感。


## 2.6 广义线性模型 Generalized Linear Model

> 参考： PRML 4.1 节；面试考点总结如下
> 1. 。
> 2. 。

&emsp;&emsp;前面所介绍的线性模型实际上都是一个模型族的特例，这个模型族被称为广义线性模型 (generalized linear model) ，首先需要介绍相关的指数族分布。

#### 指数族分布 The Exponential Family

&emsp;&emsp;前文中，我们使用最大似然法估计线性回归和线性分类模型参数时，都需要表示出目标值或向量的概率分布来建模，其中线性回归符合正态分布 $p(\boldsymbol  t|\boldsymbol x,\boldsymbol w)\sim \mathcal N(\boldsymbol \mu, \boldsymbol \sigma^2)$ ， logistic 回归对应伯努利分布 $p(\boldsymbol  t|\boldsymbol x,\boldsymbol w)\sim \mathrm{Bernoulli} (\rho)$ ，大部分概率分布（⾼斯混合分布除外）都是⼀⼤类被称为指数族 (exponential family) 概率分布的具体例⼦，它们有许多共同的性质。参数为 $\boldsymbol \eta$ 的变量 $\boldsymbol x$ 指数族分布可定义为具有下⾯形式的概率分布的集合（此处采用 Bishop PRML 中的写法，与 CS229 讲义中表示方法略有不同）

$$p(\boldsymbol x|\boldsymbol \eta)=h(\boldsymbol x)g(\boldsymbol \eta)\exp \left \{\boldsymbol \eta^T\phi(\boldsymbol x )\right \}\tag 1$$
其中 $\boldsymbol x$ 是输入向量（或标量亦可）， $\boldsymbol \eta$ 被称为概率分布的⾃然参数 (natural parameters)， $\boldsymbol \phi (\boldsymbol x)$ 是关于 $\boldsymbol x$ 的某个函数，被称为充分统计量 (sufficient statistic)，一般情况下我们仍取 $\boldsymbol\phi (\boldsymbol x)=\boldsymbol x$ 。函数 $g(\boldsymbol \eta)$ 可以被看成系数，它确保了概率分布的归⼀化，满⾜

$$g(\boldsymbol \eta)\int h(\boldsymbol x)\exp \left \{\boldsymbol \eta^T\boldsymbol\phi(\boldsymbol x )\right \}d\boldsymbol x=1\tag 2$$

&emsp;&emsp;下面我们来证明一下前文涉及的概率分布属于指数族分布。

#### 伯努利分布

&emsp;&emsp;伯努利分布可以表示成

$$\begin{align} p(x|\mu)=\mu^x(1-\mu)^{1-x}&=\exp\left \{ x\ln \mu+(1-x)\ln(1-u) \right \}\\ &=(1-\mu)\exp\left \{ \ln(\frac{\mu}{1-\mu})x \right \} \end{align}$$
可以看出 $\eta=\ln(\frac{\mu}{1-\mu})$ ，反之 $\mu=\sigma(\eta)=\frac{1}{1+\exp(-\eta)}$ ，正好是 logistic sigmoid 函数。使用公式 (28) 的标准型就可将伯努利分布写成下面的形式

$$p(x|\mu)=\sigma(-\eta)\exp(\eta x)\tag 3$$
其中利用了 logistic sigmoid 的对称性 $\sigma(-\eta)=1-\sigma(\eta)$ ，并有 $\mu(x)=x ， h(x)=1 ， g(\eta)=\sigma(-\eta)$ 。

#### 多项分布

&emsp;&emsp;接下来考虑单⼀观测 $\boldsymbol x$ 的多项式分布，形式为

$$p(\boldsymbol x|\boldsymbol \mu)=\exp\left \{ \prod_{k=1}^{M}x_k\ln \mu_k \right \}\tag 4$$

&emsp;&emsp;令 $\eta_k=\ln \mu_k$ ，我们有

$$p(\boldsymbol x|\boldsymbol \mu)=\exp(\boldsymbol \mu^T\boldsymbol x)\tag 5$$
需要注意参数 $\eta_k$ 不是相互独⽴的，因为要满⾜以下限制

$$\sum_{k=1}^{M}\mu_k=1\tag 6$$

&emsp;&emsp;因此给定任意 $M-1$ 个参数 $\mu_k$ ，剩下参数就固定了。一般情况下去掉这个限制⽐较⽅便。如果考虑此限制，我们就只⽤ $M-1$ 个参数来表示这个分布。使⽤公式 (34) 的关系，把 $\mu_M$ ⽤剩余的 $\left \{\mu_k \right \}$ 表示，其中 $k=1,2,...,M-1$ ，这样就只剩下了 $M-1$ 个参数。剩余的参数仍满⾜以下限制

$$0\leq \mu_k\leq1,\sum_{k=1}^{M-1}\mu_k\leq1\tag 6$$

&emsp;&emsp;关于公式 (32)，我们进一步可以写成

$$\exp \left \{ \sum_{k=1}^{M-1}x_k\ln \mu_k+(1- \sum_{k=1}^{M-1}x_k) \ln(1-\sum_{k=1}^{M-1}x_k)\right \}=\exp \left \{ \sum_{k=1}^{M-1}x_k\ln (\frac{\mu_k}{1-\sum_{k=1}^{M-1}\mu_k})+\ln(1-\sum_{k=1}^{M-1}\mu_k)\right \}\tag 7$$

&emsp;&emsp;整理可得 $\mu_k=\frac{\exp(\eta_k)}{1+ {\textstyle \sum_{j}^{}}\exp(\eta_j) }$ ，也就是常见的 Softmax 函数。在这种表达形式下，多项式分布为

$$p(\boldsymbol x|\boldsymbol \mu)=(1+\sum_{k=1}^{M-1}\exp(\mu_k))^{-1}\exp(\boldsymbol \mu^T\boldsymbol x)\tag 8$$

#### 高斯分布

&emsp;&emsp;对于⼀元⾼斯分布，我们有

$$\begin{align} p(x|\mu,\sigma^2)&=\frac{1}{(2\pi\sigma^2)^{\frac{1}{2}}}\exp\left \{ -\frac{1}{2\sigma^2}(x-\mu)^2 \right \}\\ &=\frac{1}{(2\pi\sigma^2)^{\frac{1}{2}}}\exp\left \{ -\frac{1}{2\sigma^2}x^2+\frac{\mu}{\sigma^2}x-\frac{1}{2\sigma^2}\mu^2 \right \} \end{align} $$
令 $\boldsymbol \eta=[\frac{\mu}{\sigma^2},-\frac{1}{2\sigma^2}] ， \boldsymbol \phi(x)=[x,x^2] ， h(x)=(2\pi)^{-\frac{1}{2}} ， g(\boldsymbol \eta)=(-2\eta_2)^{\frac{1}{2}}\exp(\frac{\eta_1^2}{4\eta_2})$ ，高斯分布就可以转化为标准指数族分布的形式。

#### 最⼤似然与充分统计量

&emsp;&emsp;考虑⽤最⼤似然法估计公式 (28) 给出的⼀般形式的指数族分布的参数向量 $\boldsymbol\phi(\boldsymbol x)$ 的问题。对公式 (29) 两侧求梯度，得

$$\nabla g(\boldsymbol \eta)\int h(\boldsymbol x)\exp \left \{\boldsymbol \eta^T\boldsymbol\phi(\boldsymbol x )\right \}d\boldsymbol x+g(\boldsymbol \eta)\int h(\boldsymbol x)\exp \left \{\boldsymbol \eta^T\boldsymbol\phi(\boldsymbol x )\right \}\boldsymbol\phi(\boldsymbol x )d\boldsymbol x=0\tag 9$$

&emsp;&emsp;重新排列并再次使用公式 (29)，可得

$$-\frac{1}{g(\boldsymbol \eta)}\nabla g(\boldsymbol \eta)=g(\boldsymbol \eta)\int h(\boldsymbol x)\exp \left \{\boldsymbol \eta^T\boldsymbol\phi(\boldsymbol x )\right \}\boldsymbol\phi(\boldsymbol x )d\boldsymbol x=\mathbb{E}[\boldsymbol\phi(\boldsymbol x)]\tag{10}$$

$$-\nabla \ln g(\boldsymbol \eta)=\mathbb{E}[\boldsymbol\phi(\boldsymbol x)]\tag{11}$$
$\boldsymbol\phi (\boldsymbol x)$ 的协⽅差可以根据 $g(\boldsymbol \eta)$ 的⼆阶导表达，对于⾼阶矩的情形也类似。因此，如果我们能对⼀个来⾃指数族分布的概率分布进⾏归⼀化，那么我们总能通过简单的求微分的⽅式找到它的矩。

&emsp;&emsp;现在考虑⼀组独⽴同分布的数据 $\boldsymbol X=\left \{ \boldsymbol x_1,...,\boldsymbol x_n\right \}$ 。对于这个数据集，似然函数为

$$p(\boldsymbol X\mid \boldsymbol \eta)=(\prod_{n=1}^{N}h(\boldsymbol x_n))g(\boldsymbol \eta)^N\exp \left \{ \boldsymbol \eta^T\sum_{n=1}^{N}\boldsymbol\phi (\boldsymbol x_n) \right \}\tag{12}$$
取负对数似然函数并令关于 $\boldsymbol \eta$ 的导数等于零，我们可以得到最⼤似然估计 $\boldsymbol\phi_{ML}$ 满⾜的条件

$$-\nabla \ln g(\boldsymbol \eta_{ML})=\frac{1}{N}\sum_{n=1}^{N}\boldsymbol\phi(\boldsymbol x_n)\tag{13}$$

&emsp;&emsp;原则上可以通过解这个⽅程来得到 $\boldsymbol\eta_{ML}$ 。我们看到最⼤似然估计的解只通过 $\sum_{n=1}^{N}\boldsymbol\phi(\boldsymbol x_n)$  对数据产⽣依赖，因此这个量被称为的充分统计量，所以最大似然解只依赖于充分统计量。对伯努利分布，函数 $\boldsymbol\phi (\boldsymbol x)=\boldsymbol x$ ，因此我们只需要关注数据点 $\left \{ x_n \right \}$ 的和即可。⽽对于⾼斯分布，$\boldsymbol \phi(x)=[x,x^2]$ ，因此我们应同时关注 $\left \{ \boldsymbol x_n \right \}$ 的和以及 $\left \{ \boldsymbol x_n^2 \right \}$ 的和。
如果考虑极限 $N\rightarrow+\infty$ ，那么公式 (43) 的右侧变成了 $[\boldsymbol\phi(\boldsymbol x_n)]$ ， 通过与公式 (13) ⽐较，可以看到在这个极限的情况下， $\boldsymbol\eta_{ML}$ 与真实值 $\boldsymbol\eta$ 相等。

#### 构建广义线性模型 Constructing Generalized Linear Model

&emsp;&emsp;正如伯努利分布，多项分布，高斯分布与指数族分布的关系一样，线性回归，逻辑回归等也是一个更为广泛的广义线性模型的例子。为方便和以前公式对照，指数族分布对应的应该是目标值 $\boldsymbol y$ ，所以下面讨论需要将 $\boldsymbol x$ 替换成 $\boldsymbol y$ ，另外将 $\boldsymbol x$ 定义为线性模型的输入，并令 $\boldsymbol y=f(\boldsymbol x)$ 。

&emsp;&emsp;广义线性模型的构建需要基于以下三条假设：

1. $p(\boldsymbol y|\boldsymbol \eta,\boldsymbol x)$ 符合以 $\eta$ 为参数的指数族分布。
2. 给定输入 $\boldsymbol x$ ，我们的目标是预测 $\phi(\boldsymbol y)$ （注意这里将第一节的 $x$ 全部用目标值 $y$ 代替）的理想值，一般取 $\phi(\boldsymbol y) =\boldsymbol y$ ，正好是目标值。此时 $f$ 应满足 $h(\boldsymbol x)=E[\boldsymbol y\mid \boldsymbol x]$ 。
3. 自然参数 $\boldsymbol\eta$ 和输入 $\boldsymbol x$ 满足线性关系 $\boldsymbol \eta=\boldsymbol w^T \boldsymbol x$ 

&emsp;&emsp;在指数族分布的推导过程中，我们已经发现，如果认为上述三条假设成立，我们已经得出了相应的 logistic sigmoid 函数，softmax 函数，以及更一般的线性函数，分别对应二分类，多分类，以及线性回归的基础模型，所以他们都属于广义线性模型。

&emsp;&emsp;至此，线性分类也告一段落，虽然现在主流 deep learning 的任务都是以分类为主，但是成熟的模型体系从训练到应用似乎都不用了解这些基础的分类算法，然而那些高级算法总归还是脱胎于这些基础的理论，扎实的基本功对于提高学科上限还是很重要的。
