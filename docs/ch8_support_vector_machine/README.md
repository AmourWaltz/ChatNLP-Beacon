# 8 稀疏核机 Sparse Kernel Machine

&emsp;&emsp;前⼀章中讨论的核方法在训练阶段需要对所有训练点进行两两计算核函数，由于时间复杂度过高在数据集较大时几乎是不可⾏的，并且在预测时也会花费过多的时间。本章我们会讨论具有稀疏 (sparse) 解的核算法，一种对新数据的预测只依赖于训练数据的⼀个⼦集上计算的核函数，这通常被称为稀疏核机 (sparse kernel machine)。本章重点讨论⽀持向量机 (support vector machine, SVM)，这就是一种稀疏核机，下面会通过一个例子解释它的名称以及稀疏性来源。这是一种很流行的算法，常常被用来解决分类和回归问题。⽀持向量机的⼀个重要性质是模型参数的确定对应于⼀个凸优化问题，因此许多局部解也是全局最优解。

## 8.1 支持向量机 Support Vector Machine

&emsp;&emsp;我们首先通过一个二分类问题引出支持向量机，然后再讨论其与核函数的关系。假设线性基函数模型为

$$y(\boldsymbol x)=\boldsymbol w^T\phi(\boldsymbol x)+b\tag 1 $$
训练数据集 $\mathcal D$ 由 N 个输入向量 $\boldsymbol x_1,\dots,\boldsymbol x_N$ 组成，对应目标值为 $t_1,\dots,t_N$ ，其中 $t_n\in\left \{ -1, 1 \right \}$。假设训练数据集在特征空间中是线性可分的，即对于所有的训练数据都有 $t_ny(\boldsymbol x_n)>0$。这与感知器一节中的定义相同，在感知器算法中，我们通过在有限步骤内不断迭代寻找出一个解（即参数 $\boldsymbol w$ 和 $b$），但这种解往往不唯一，感知器算法通常选择能将所有数据正确分类的第一个解（数据线性可分时），这依赖于参数的初始化，对于可能存在的所有解，我们应该寻找泛化误差和过拟合最小的那个，在支持向量机中，引入了边缘 (margin) 的概念，即决策边界与任意样本之间的最⼩距离，如下图所示
​
<div align=center>
<img src="images/8_1_svm1.png" width="80%"/>
</div>

&emsp;&emsp;在⽀持向量机中，决策边界被选为使边缘最⼤化的那个边界。已有相关高斯核方法证明了最优超平⾯是有着最⼤边缘的超平⾯，距离超平⾯较近的点对超平⾯的影响⼤于距离较远的点，在高斯核参数取极限的情况下，超平⾯会变得与那些距离较远的点无关，也就是在核方法不再考虑这些点，这就是支持向量机稀疏性的来源，对于那些有决定意义的点，我们就称之为支持向量，因此这种方法就叫做支持向量机。

&emsp;&emsp;在第三章线性分类中，我们提过点 $\boldsymbol x$ 距离由 $y(\boldsymbol x)=0$ 定义的超平⾯的垂直距离为 $\frac{\left | y(\boldsymbol x) \right |}{\left \| \boldsymbol w \right \|}$，我们需要考虑的是那些能够被正确分类的数据点，即 $t_ny(\boldsymbol x_n)>0$，因此点 $\boldsymbol x$ 距离决策⾯的距离为

$$\frac{t_ny(\boldsymbol x)}{\left \| \boldsymbol w \right \|}=\frac{t_n(\boldsymbol w^T\phi(\boldsymbol x)+b)}{\left \| \boldsymbol w \right \|}\tag 2$$
边缘由数据集⾥垂直距离最近的点 $\boldsymbol x_n$ 给出，我们希望最优化参数 $\boldsymbol w 和 b$，使得这个距离最大化，通过下式得到

$${\arg \max_{\boldsymbol w,b}}\left \{ \frac{1}{\left \| \boldsymbol w \right \| }\min_{n}\left [ t_n\left ( \boldsymbol w^T\phi(\boldsymbol x)+b \right )  \right ]  \right \} \tag 3$$ 
优化上式的关键是找出最近的点 $\boldsymbol x_n$，这就需要对 $t_n\left ( \boldsymbol w^T\phi(\boldsymbol x)+b \right )$ 进行一些限制同时不影响到决策面的距离，由于对参数 $\boldsymbol w$ 和 $b$ 进行同比例缩放并不会改变任意点 $\boldsymbol x_n$ 到决策面的距离 $\frac{t_ny(\boldsymbol x_n)}{\left \| \boldsymbol w \right \|}$，可以利用这个性质，假设有一组参数恰好使距离决策面最近的点 $\boldsymbol x_n$ 有

$$t_n\left ( \boldsymbol w^T\phi(\boldsymbol x)+b \right ) =1\tag 4$$
这时所有的点都会满足

$$t_n\left ( \boldsymbol w^T\phi(\boldsymbol x)+b \right ) \geq1,~~~~~~n\in\left\{ 1,\dots,N  \right \} \tag 5$$

&emsp;&emsp;根据公式 (5) 定义及上图不难看出，当找到这个最大化的边缘后，至少存在两个 $\boldsymbol x_n$ 使公式 (5) 取等号。这样最优化问题就简化为最⼤化 $\frac{1}{\left \| \boldsymbol w \right \| }$ ，这里做一些小小的变换，将问题转换为最⼩化 $\left \| \boldsymbol w \right \|^2$ ，这是为了变成一个二次规划 (quadratic programming) 问题，即在⼀组线性不等式的限制条件下最⼩化⼆次函数，之所以这样做是因为二次型是一个凸函数且易于优化分析。因此我们要在限制条件 (4) 下，求解最优化问题

$${\arg \min_{\boldsymbol w,b}}\frac{1}{2}\left \| \boldsymbol w \right \|^2\tag 6$$
为了解决这个限制的最优化问题，我们引⼊拉格朗⽇乘数 $a_n\geq 0$。由于并不知道最近的点是哪个，我们考虑所有数据，令公式 (5) 中的每个限制条件都对应⼀个乘数 $a_n$，从⽽可得下⾯的拉格朗⽇函数

$$L(\boldsymbol w,b,\boldsymbol a)=\frac{1}{2}\left \| \boldsymbol w \right \|^2-\sum_{n=1}^Na_n\left \{ t_n(\boldsymbol w^T\phi(\boldsymbol x_n)+b)-1 \right \}\tag 7$$
其中 $\boldsymbol a=\left ( a_1,\dots,a_N \right )^T$，对于公式 (7)，我们要做关于 $\boldsymbol w$ 和 $b$ 的最小化，关于 $\boldsymbol a$ 的最大化。首先令 $L(\boldsymbol w,b,\boldsymbol a)$ 关于 $\boldsymbol w$ 和 $b$ 的导数为零，可得

$$\boldsymbol w=\sum_{n=1}^Na_nt_n\phi(\boldsymbol x_n)\tag8$$
$$\sum_{n=1}^Na_nt_n=0\tag 9$$

&emsp;&emsp;公式 (8) 恰好就是上一章对偶表示中参数的表示方法，使⽤这两个条件从 $L(\boldsymbol w,b,\boldsymbol a)$ 中消去 $\boldsymbol w$ 和 $b$，就得到了最⼤化边缘问题的对偶表示，于是再转换为关于 $\boldsymbol a$ 的最大化

$$\tilde L(\boldsymbol a)=\sum_{n=1}^Na_n-\frac{1}{2}\sum_{n=1}^N\sum_{m=1}^Na_na_mt_nt_mk(\boldsymbol x_n,\boldsymbol x_m)\tag {10}$$
其中 $a_n\geq 0,\sum_{n=1}^Na_nt_n=0,k(\boldsymbol x,\boldsymbol x')=\phi(\boldsymbol x)^T\phi(\boldsymbol x')$。公式 (10) 右侧第二项让核函数 $k(\boldsymbol x,\boldsymbol x')$ 正定这⼀限制条件存在的原因变得很显然，因为必须确保拉格朗⽇函数 $\tilde L(\boldsymbol a)$ 有上界，从⽽使最优化问题有良好的定义。关于公式 (10) 的求解在后文会提及，我们直接看对新输入数据 $\boldsymbol x$ 的预测，将公式 (8) 替换公式 (1) 中的参数项，可得

$$y(\boldsymbol x)=\sum_{n=1}^Na_nt_nk(\boldsymbol x,\boldsymbol x_n)+b\tag {11}$$
拉格朗日乘数法的最优化函数成立需要满足 Karush-Kuhn-Tucker 条件，即假设拉格朗日函数形式为

$$L=f(\boldsymbol x)+\lambda g(\boldsymbol x)\tag {12}$$
要利用这个函数最大化 $f(\boldsymbol x)$ ，需要同时满足 $\lambda \geq0,g(\boldsymbol x)\geq0,\lambda g(\boldsymbol x)=0$，对于公式 (7)，我们还需要满足

$$a_n\left \{ t_ny(\boldsymbol x_n)-1 \right \}=0\tag {13}$$

&emsp;&emsp;任何使得 $a_n=0$ 的数据点都不会出现在公式 (11) 的求和式中，因此对新数据点的预测没⽤，剩下的数据点才能被称为⽀持向量 (support vector)。由于这些⽀持向量满⾜ $t_ny(\boldsymbol x_n)=1$，因此它们对应于特征空间中位于最⼤边缘超平⾯上的点。这个性质是⽀持向量机在实际应⽤中的核⼼，⼀旦模型被训练完毕，相当多的数据点都可以被丢弃，只有⽀持向量被保留。
在解决二次规划并找到 $\boldsymbol a$ 之后，根据 $t_ny(\boldsymbol x_n)=1$ ，代入公式 (11)，即可确定参数 $b$ 的值

$$t_n\left ( \sum_{m\in \mathcal S}a_mt_mk(\boldsymbol x_n,\boldsymbol x_m)+b \right )=1\tag {14}$$
其中 $\mathcal S$ 是 $\mathcal D$ 中支持向量的下标集合，虽然我们能⽤任意⽀持向量 $\boldsymbol x_n$ 解这个关于 $b$ 的⽅程，但通过下⾯的⽅式可以得到⼀个在数值计算上更加稳定的解。⾸先乘以 $t_n$ ，使⽤ ${t_n}^2=1$ 的性质，然后对所有⽀持向量整理⽅程，解出 $b$ ，可得

$$b=\frac{1}{N_{\mathcal S}}\sum_{n\in \mathcal S}\left ( t_n-\sum_{m\in \mathcal S}a_mt_mk(\boldsymbol x_n,\boldsymbol x_m) \right )\tag {15}$$
其中 $N_{\mathcal S}$ 是⽀持向量的总数。
接下来我们可以将最⼤边缘分类器⽤带有简单⼆次正则化项的最⼩化误差函数表示，形式为

$$\sum_{n=1}^NE_\infty(y(\boldsymbol x_n)t_n-1)+\lambda\left \| \boldsymbol w \right \|^2\tag {16}$$
其中 $E_\infty(z)$ 是一个函数，当 $z\geq 0$ 时，函数值为零，其他情况下函数值为 $\infty$。这就确保了限制条件 (5) 成⽴，只要正则化参数满⾜ $\lambda >0$，那么它的精确值就没有作⽤，这样就将模型参数优化为了最⼤化边缘。

#### 重叠类分布

&emsp;&emsp;上述讨论中我们假设训练数据点在特征空间 $\phi(\boldsymbol x)$ 中是线性可分的，解得的⽀持向量机在原始输⼊空间 $\boldsymbol x$ 中会对训练数据进⾏精确划分，虽然对应的决策边界是⾮线性的。然⽽实际中类条件分布可能重叠，这种情况下对训练数据的精确划分会导致较差的泛化能⼒。

&emsp;&emsp;这时我们可以通过修改公式 (16) 误差函数，使得数据点允许在决策边界的错误分类的一侧，但需要增加⼀个惩罚项，这个惩罚项随着与决策边界的距离的增⼤⽽增⼤，令这个惩罚项是距离的线性函数⽐较⽅便。为了完成这⼀点，我们引⼊松弛变量 (slack variable) $\xi_n \geq0$，其中 $n=1,\dots,N$，每个训练数据点都有⼀个松弛变量。对于位于正确的边缘边界内部的点或者边界上的点有 $\xi_n=0$ ，对于其他点有 $\xi_n=\left | t_n-y(\boldsymbol x_n) \right |$ ，而对于决策边界 $y(\boldsymbol x_n)=0$ 上的点有 $\xi_n=1, \xi_n>1$ 就是错误分类的点，这样公式 (5) 的限制条件就变为

$$t_ny(\boldsymbol x_n)\geq1-\xi_n,~~~~~~,n=1,\dots,N\tag{16}$$
$\xi_n=0$ 的数据点是位于边缘或边缘正确一侧的正确分类的点，$0<\xi_n\leq1$ 的点是位于边缘内部，但是在决策边界正确⼀侧的点，$\xi_n>1$ 的点是位于决策边界的错误⼀侧的点。这就允许⼀些训练数据点被错误分类。虽然松弛变量允许类分布的重叠，但这个框架对于异常点很敏感，因此误分类的惩罚随着 $\xi$ 线性增加。
现在⽬标是最⼤化边缘，同时以⼀种⽐较柔和的⽅式惩罚位于边界错误⼀侧的点。于是，改为最⼩化

$$C\sum_{n=1}^N\xi_n+\frac{1}{2}\left \| \boldsymbol w \right \|^2\tag{18}$$
其中参数 $C>0$ 类似于一个正则化系数，控制两个惩罚项的折中关系。在极限 $C\rightarrow\infty$ 情况下就是线性可分的支持向量机，现在在公式 (17) 以及 $\xi_n\geq0$ 的条件下最⼩化公式 (18)。对应拉格朗⽇函数为

$$L(\boldsymbol w,b,\boldsymbol \xi,\boldsymbol a,\boldsymbol \mu)=\frac{1}{2}\left \| \boldsymbol w \right \|^2+C\sum_{n=1}^N\xi_n-\sum_{n=1}^Na_n\left \{ t_ny(\boldsymbol x_n)-1+\xi_n \right \}-\sum_{n=1}^N\mu_n\xi_n\tag{19}$$
其中 $\left \{ a_n\geq0 \right \}$ 和 $\left \{ \mu_n\geq0 \right \}$ 是拉格朗日乘数，公式 (19) 对应的 (12) 中的 KKL 条件分别为 $a_n\geq0,t_ny(\boldsymbol x_n)-1+\xi_n\geq0,a_n(t_ny(\boldsymbol x_n)-1+\xi_n)=0,\mu_n\geq0,\xi_n\geq0,\mu_n\xi_n=0$, 现在对 $\boldsymbol w,b,\left \{ \xi_n \right \}$ 进⾏优化，使公式 (19) 对其导数为 0

$$\frac{\partial L}{\partial \boldsymbol w} = 0\Rightarrow \boldsymbol w=\sum_{n=1}^Na_nt_n\phi( \boldsymbol x_n)\tag{20}$$
$$\frac{\partial L}{\partial b} = 0\Rightarrow \sum_{n=1}^Na_nt_n=0\tag{21}$$
$$\frac{\partial L}{\partial \xi_n} = 0\Rightarrow a_n=C-\mu_n\tag{22}$$
代入公式 (19) 消除 $\boldsymbol w,b,\left \{ \xi_n \right \}$ 可得拉格朗日函数

$$\tilde{L}(\boldsymbol a)=\sum_{n=1}^Na_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^Na_na_mt_nt_mk(\boldsymbol x_n,\boldsymbol x_m)\tag{23}$$
这与公式 (10) 形式上完全相同，唯⼀的区别就是限制条件有些差异。由于拉格朗日乘数 $\left \{ a_n\geq0 \right \}$ 及 $\left \{ \mu_n\geq0 \right \}$，根据公式 (22) 可知关于对偶变量 $\left \{ a_n \right \}$ 最大化公式 (23) 需要满足的限制为 $0\leq a_n\leq C$, $\sum_{n=1}^Na_nt_n=0$, 这同样又是一个二次规划问题，对新数据的预测同样为公式 (11)，同样我们不关心那些使 $a_n=0$ 的训练数据点，所有支持向量都应该满足

$$t_ny(\boldsymbol x_n)=1-\xi_n\tag{24}$$

&emsp;&emsp;再考虑这些数据点中，如果 $a_n<C$ ，那么根据限制条件 (22) 有 $\mu_n>0$ ，再根据拉格朗日函数 (19) 的 KKL 条件，可得 $\xi_n=0$ ，表明这些点位于边缘上，而对于 $a_n=C$ 的点，可根据 $\xi_n$ 是否大于 1 判断是否在错误分类的一边。
为了确定参数 $b$ ，因为 $0<a_n<C$ 的支持向量满足 $\xi_n=0$ 即 $t_ny(\boldsymbol x_n)=1$ ，所以也可以推导出公式 (14) 和 (15) 的表达式，唯一不同的是， $b$ 的解的形式为

$$b=\frac{1}{N_{\mathcal M}}\sum_{n\in \mathcal M}\left ( t_n-\sum_{m\in \mathcal S}a_mt_mk(\boldsymbol x_n,\boldsymbol x_m) \right )\tag{25}$$
其中 $\mathcal M$ 表示 $0<a_n<C$ 的集合，这是因为确定参数 $\boldsymbol w$ 时我们只考虑了 $a_n=0$ 的点，这些事位于边缘上的点，而利用 $t_ny(\boldsymbol x_n)=1$ 得出公式 (14) 这样的表达式时我们考虑的是 $0<a_n<C$ 这些点。

#### 回归问题的 SVM

&emsp;&emsp;现在将支持向量机推广到回归模型，假设一个简单的线性回归模型，我们最小化误差函数

$$\frac{1}{2}\sum_{n=1}^N\left \{ y_n-t_n \right \}^2+\frac{\lambda}{2}\left \| \boldsymbol w \right \|^2\tag{26}$$
为了保持支持向量机的稀疏性，⼆次误差函数会被替换为⼀个 $\epsilon-$ 不敏感误差函数，如果预测 $y(\boldsymbol x)$ 和⽬标 $t$ 的差绝对值⼩于 $\epsilon$ ，那么这个误差函数给出的误差等于零，其中 $\epsilon>0$ 。 $\epsilon-$ 不敏感误差函数的⼀个简单例⼦是

$$E_\epsilon \left ( y\left ( \boldsymbol x \right ) -t \right ) =\left\{\begin{matrix} 0,~~~~~~\left | y\left ( \boldsymbol x \right ) -t \right | < \epsilon  \\ \left | y\left ( \boldsymbol x \right ) -t \right | - \epsilon,~~~~~~~~other \end{matrix}\right.  $$
在不敏感区域之外，会有⼀个与误差相关联的线性代价，如下图所示，绿色代表原平方和误差函数，红色代表支持向量机的 \epsilon- 不敏感误差函数
​
<div align=center>
<img src="images/8_1_svm2.png" width="70%"/>
</div>

&emsp;&emsp;最小化正则化的 $\epsilon-$ 不敏感误差函数，同样引入一个正则化系数 $C$ ，形式为

$$C\sum_{n=1}^NE_\epsilon \left ( y\left ( \boldsymbol x \right ) -t \right )+\frac{1}{2}\left \| \boldsymbol w \right \|^2\tag{27}$$ 
由于 $E_{\epsilon} \left ( y\left ( \boldsymbol x \right ) -t \right )$ 是一个分段函数，无法直接求导，但是 $E_{\epsilon} $ 总会取得 0 和 $\left | y\left ( \boldsymbol x \right ) -t \right | - \epsilon$ 中的较大值，所以我们可以引入一个松弛变量同时满足大于 0 和 $\left | y\left ( \boldsymbol x \right ) -t \right | - \epsilon$，由于绝对值的存在，我们需要引入两个松弛变量 $\xi_n\geq0,\hat \xi_n\geq0$ ，其中 $\xi_n\geq0$ 对应于 $t_n>y(\boldsymbol x_n)+\epsilon$ 的数据点，而 $\hat \xi_n\geq0$ 对应于 $t_n<y(\boldsymbol x_n)-\epsilon$ 的数据点，如下图所示
​
<div align=center>
<img src="images/8_1_svm3.png" width="80%"/>
</div>

&emsp;&emsp;⽬标点位于 $\epsilon-$ 管道内的条件是 $y(\boldsymbol x_n)-\epsilon\leq t_n\leq y(\boldsymbol x_n)+\epsilon$。引⼊松弛变量使得数据点能够位于管道之外，只要松弛变量不为零即可，对应的条件变为

$$t_n\leq y(\boldsymbol x_n)+\epsilon+\xi_n,t_n\geq y(\boldsymbol x_n)-\epsilon-\hat \xi_n\tag{28}$$ 
于是线性回归的支持向量机误差函数变为

$$C\sum_{n=1}^N\left ( \xi_n+\hat \xi_n \right )+\frac{1}{2}\left \| \boldsymbol w \right \|^2\tag{29}$$
对公式 (29) 做限制条件 (28) 以及 $\xi_n\geq0,\hat \xi_n\geq0$ 下的拉格朗日函数，引入拉格朗日乘数 $a_n\geq 0,\hat a_n\geq 0,\xi_n\geq 0,\hat \xi_n\geq 0$ ，然后最优化

$$\begin{aligned} L(\boldsymbol w,b,\boldsymbol \xi,\boldsymbol {\hat \xi}, \boldsymbol \mu,\boldsymbol {\hat \mu},\boldsymbol a,\boldsymbol {\hat a})=&C\sum_{n=1}^N\left ( \xi_n+\hat \xi_n \right )+\frac{1}{2}\left \| \boldsymbol w \right \|^2-\sum_{n=1}^N\left ( \mu_n\xi_n+\hat \mu_n\hat \xi_n \right )\\ &-\sum_{n=1}^Na_n\left ( \epsilon+\xi_n+y_n-t_n \right )-\sum_{n=1}^N\hat a_n\left ( \epsilon+\hat \xi_n+y_n-t_n \right ) \end{aligned}$$
同样使用公式 (1) （此时将公式 (1) 看做线性回归模型）替换 $y(\boldsymbol x)$ ，然后令拉格朗⽇函数关于 $\boldsymbol w,b,\boldsymbol \xi,\boldsymbol {\hat \xi}$ 的导数为零，有

$$\frac{\partial L}{\partial \boldsymbol w} =0\Rightarrow\boldsymbol w=\sum_{n=1}^N\left ( a_n-\hat a_n \right )\phi(\boldsymbol x_n)\tag{31}$$
$$\frac{\partial L}{\partial b}=0\Rightarrow \sum_{n=1}^N\left ( a_n-\hat a_n \right )=0\tag{32}$$
$$\frac{\partial L}{\partial \xi_n}=0\Rightarrow a_n+\mu_n=C\tag{33}$$
$$\frac{\partial L}{\partial \hat\xi_n}=0\Rightarrow \hat a_n+\hat\mu_n=C\tag{34}$$
使⽤这些结果代换拉格朗⽇函数中对应的变量，可以转换为对偶问题到关于 $\left \{ a_n \right \},\left \{ \hat a_n \right \}$ 最⼤化

$$\begin{aligned} \tilde L\left ( \boldsymbol a,\boldsymbol {\hat a} \right )=&-\frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\left ( a_n-\hat a_n \right )\left ( a_m-\hat a_m \right )k\left ( \boldsymbol x_n,\boldsymbol x_m \right )\\ &-\epsilon\sum_{n=1}^N\left ( a_n+\hat a_n \right )+\sum_{n=1}^N\left ( a_n-\hat a_n \right )t_n \end{aligned}$$
同样根据公式 (32) 和 (33)，我们可以得出限制条件 $0\leq a_n\leq C,0\leq \hat a_n\leq C$ ，然后将公式 (32) 代入公式 (1)，对于新的预测变量，就变成了

$$y(\boldsymbol x)=\sum_{n=1}^N\left ( a_n-\hat a_n \right )k\left ( \boldsymbol x,\boldsymbol x_n \right )+b\tag{35}$$
再次根据拉格朗日函数 (29) 的 KKT 条件，让系数与限制的乘积为零，有

$$a_n( \epsilon+\xi_n+y_n-t_n )=0\tag{36}$$
$$\hat a_n( \epsilon+\hat \xi_n+y_n-t_n )=0\tag{37}$$
$$(C-a_n )\xi_n=0\tag{38}$$
$$(C-\hat a_n )\hat\xi_n=0\tag{39}$$

&emsp;&emsp;逐一分析，首先看公式 (36)，当 $\epsilon+\xi_n+y_n-t_n=0$ 时，数据点要么位于 $\epsilon-$ 管道的上边界上 $(\xi_n=0)$ ，要么位于上边界的下⽅ $(\xi_n>0)$，类似地，公式 (37) 也可以这样推导，得出数据点位于下边界上或下边界上方的结论。此外两个限制 $\epsilon+\xi_n+y_n-t_n=0$ 和 $\epsilon+\hat \xi_n+y_n-t_n=0$ 不兼容，通过将两式相加，由于 $\epsilon,\xi_n$ 非负，所以 $a_n,\hat a_n$ 至少一个为零或都为零。

&emsp;&emsp;同样地，支持向量应该是那些对预测有贡献，在参数中可以出现的点，这些点需要满足 $a_n\ne 0$ 或 $\hat a_n\ne 0$ ，位于 $\epsilon-$ 管道边界上或管道外部，我们再次得到了⼀个稀疏解。

&emsp;&emsp;关于参数 $b$ ，考虑⼀个数据点，满⾜ $0< a_n< C$ 。根据公式 (38) (36)，⼀定有 $\xi_n=0$ ，$\epsilon+y_n-t_n=0$ ，使⽤公式 (1) 求解 $b$ ，有

$$b=t_n-\epsilon-\boldsymbol w^T\phi(\boldsymbol x_n)=t_n-\epsilon-\sum_{m=1}^N\left ( a_m-\hat a_m \right )k\left ( \boldsymbol x_n,\boldsymbol x_m \right )\tag{40}$$
&emsp;&emsp;可以看出，支持向量机的线性回归和线性分类解法思路相差不大，通过对这个过程的推导，我们都可以证明支持向量机的稀疏性，支持向量机也得以广泛应用。