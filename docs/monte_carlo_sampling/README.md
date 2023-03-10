# 采样方法 Sampling Method

&emsp;&emsp;对于⼤多数实际应⽤中的概率模型来说，精确推断是不可⾏的，因此不得不借助于某种形式的近似。上一章讨论了变分推断的近似⽅法，本章就考虑基于数值采样的近似推断⽅法，也被称为蒙特卡罗采样方法 (Monte Carlo Sampling Method)。

&emsp;&emsp;在贝叶斯神经网络中，我们提到过针对连续变量积分通过蒙特卡罗采样离散化的处理

$$\int \ln p(\mathcal D|\boldsymbol w)q(\boldsymbol w)\mathrm d\boldsymbol w\approx\frac{1}{K}\sum_{k=1}^{K}\ln p(\mathcal D|\boldsymbol w_k)\tag1$$

&emsp;&emsp;这样处理的本质是，在上述问题模型描述中，我们感兴趣的是⾮观测变量 $\boldsymbol w$ 上的似然概率分布 $p(\mathcal D|\boldsymbol w)$，从而用于计算在已知数据集 $\mathcal D$ 上的期望，并对新输入的数据进行预测。因此，我们希望解决的问题涉及到关于⼀个概率分布 $p(\boldsymbol z)$ 寻找某个函数 $f(\boldsymbol z)$ 的期望。在连续变量的情形下，需要计算下⾯的期望

$$\mathbb E[f]=\int p(\boldsymbol z)f(\boldsymbol z)\mathrm d\boldsymbol z\tag2$$

&emsp;&emsp;如下图所示是单一连续变量的情形

<div align=center>
<img src="images/12_1_monte1.png" width="60%"/>
</div>

&emsp;&emsp;这其实是一种离散上的加权求和，使用解析的方法求出这种期望比较困难，这时候引入蒙特卡罗采样⽅法，其⼀般思想是得到从概率分布 $p(\boldsymbol z)$ 中独⽴抽取的⼀组变量 $\boldsymbol z^{(l)}$，其中 $l=1,\dots,L$。这使得期望可以通过有限和的⽅式计算，即

$$\hat f=\frac{1}{L}\sum_{l=1}^Lf(\boldsymbol z^{(l)})\tag3$$

&emsp;&emsp;只要样本 $\boldsymbol z^{(l)}$ 是从概率分布 $p(\boldsymbol z)$ 中抽取的，那么 $\mathbb E[\hat f]=\mathbb E[f]$，因此估计 $\hat f$ 具有正确的均值。估计 $\hat f$ 的⽅差为

$$\mathrm {var}[\hat f]=\frac{1}{L}\mathbb E[(f-\mathbb E[f])^2]\tag4$$

&emsp;&emsp;这是函数 $f(\boldsymbol z)$ 在概率分布 $p(\boldsymbol z)$ 下的⽅差。估计的精度不依赖于 $\boldsymbol z$ 的维度，并且对于数量相对较少的样本 $\boldsymbol z^{(l)}$，可能会达到较⾼的精度，在大规模的应⽤中，10 个或 20 个独⽴样本就⾜够以⾼精度对期望做出估计。

&emsp;&emsp;然⽽样本 $\left \{ \boldsymbol z^{(l)} \right \}$ 可能不是相互独⽴的，因此有效样本数量可能远远⼩于表⾯上的样本数量。在上图中，如果 $f(\boldsymbol z)$ 在 $p(\boldsymbol z)$ 较⼤的区域中的值较⼩，而在 $p(\boldsymbol z)$ 较小的区域中的取值较大，那么期望就可能由 $p(\boldsymbol z)$ ⼩概率的区域控制，为了达到⾜够精度，需要相对较⼤的样本数量。

## 6.1 蒙特卡罗采样 Monte Carlo Sampling

&emsp;&emsp;本节研究从⼀个给定的概率分布中⽣成随机样本的⼀些⽅法。由于样本是通过计算机算法⽣成的，因此这些样本实际上是伪随机数 (pseudo-random numbers)，它们通过计算的⽅法确定。我们假定算法⽣成的是 $(0,1)$ 之间均匀分布的伪随机数，这对于很多编程语言很容易实现。

#### 标准概率分布

&emsp;&emsp;既然有了均匀分布的随机数，我们就来考虑如何从⾮均匀分布中⽣成随机数。假设 $z$ 在区间 $(0,1)$ 上均匀分布，使⽤某个函数 $f(\cdot)$ 对 $z$ 进⾏变换，即 $y=f(z)$。$y$ 上的概率分布为

$$p(y)=p(z)\left | \frac{\mathrm d z}{\mathrm d y} \right |\tag5$$

&emsp;&emsp;由于 $p(z)=1$，我们希望选择⼀个函数 $f(z)$ 使产⽣出的 $y$ 具有某种分布形式 $p(y)$，对公式 (5) 进⾏积分，有

$$z=h(y)\equiv \int _{-\infty}^yp(\hat y)\mathrm d\hat y\tag6$$

&emsp;&emsp;这时 $y=h^{-1}(z)$，这表明如果使⽤⼀个函数来对 $z$ 的均匀分布的随机数进⾏变换，这个函数就是所求概率分布的不定积分的反函数。

&emsp;&emsp;考虑一个指数分布

$$p(y)=\lambda\exp(-\lambda y)\tag7$$
其中 $0\leq y<\infty$，公式 (6) 的积分下界为 0，代入有 $h(y)=1-\exp(-\lambda y)$，然后将均匀分布的变量 $z$ 使⽤ $y=-\lambda^{-1}\ln (1-z)$ 进⾏变换，那么 $y$ 就会服从一个指数分布。这种变换方法对于一些简单的概率分布可⾏，我们还需要寻找⼀些更⼀般的⽅法。

#### 重要性采样

&emsp;&emsp;如果要使⽤公式 (2) 计算期望，那我们就要可能从一些复杂概率分布中采样，重要性采样 (importance sampling) ⽅法提供了直接近似期望的框架，它本⾝并没有提供从概率分布 $p(\boldsymbol z)$ 中直接采样的⽅法。公式 (3) 给出的期望的有限和近似依赖于概率分布 $p(\boldsymbol z)$ 的采样。然⽽，假设直接从 $p(\boldsymbol z)$ 中采样⽆法完成，但是对于任意给定的 $\boldsymbol z$ 值，可以很容易计算 $p(\boldsymbol z)$。⼀种简单的计算期望的⽅法是将 $\boldsymbol z$ 空间离散化为均匀的格点，将被积函数使⽤求和的⽅式计算，形式为

$$\mathbb E[f]\simeq \sum_{l=1}^Lp(\boldsymbol z^{(l)})f(\boldsymbol z^{(l)})\tag8$$

&emsp;&emsp;这种⽅法⼀个明显问题是求和式中的项的数量随着 $\boldsymbol z$ 的维度指数增长。此外，如果感兴趣的概率分布值较大的区域限制在 $\boldsymbol z$ 空间⼀个很⼩的区域，那么均匀采样⾮常低效，只有⾮常⼩的⼀部分样本会对求和式产⽣较大贡献，而理想状态是从 $p(\boldsymbol z)$ 或 $p(\boldsymbol z)f(\boldsymbol z)$ 的值较⼤的区域中采样。

&emsp;&emsp;重要性采样是基于对一个提议分布 $q(\boldsymbol z)$ 的使⽤，我们从提议分布中采样，这有点类似于变分推断中的近似分布，都是认为很难在原始概率分布上进行操作，然后引入另外一个分布 $q$。通过 $q(\boldsymbol z)$ 中的样本 $\left \{ \boldsymbol z^{(l)} \right \}$ 的有限和的形式来表示期望

$$\begin{align} \mathbb E[f]&=\int f(\boldsymbol z)p(\boldsymbol z)\mathrm d\boldsymbol z\\ &=\int f(\boldsymbol z)\frac{p(\boldsymbol z)}{q(\boldsymbol z)}q(\boldsymbol z)\mathrm d\boldsymbol z\\ &\simeq \frac{1}{L}\sum_{l=1}^L\frac{p(\boldsymbol z^{(l)})}{q(\boldsymbol z^{(l)})}f(\boldsymbol z^{(l)})\\ \end{align}\tag9$$

$r_l=\frac{p(\boldsymbol z^{(l)})}{q(\boldsymbol z^{(l)})}$ 被称为重要性权重，修正了由于从错误的概率分布中采样引⼊的偏差。

&emsp;&emsp;概率分布 $p(\boldsymbol z)$ 的计算结果没有归⼀化，即 $p(\boldsymbol z)=\frac{\tilde p(\boldsymbol z)}{Z_p}$，其中 $\tilde p(\boldsymbol z)$ 很容易计算出来，⽽ $Z_p$ 未知。类似地，也可以使⽤重要采样分布 $q(\boldsymbol z)=\frac{\tilde q(\boldsymbol z)}{Z_q}$，它具有相同的性质，于是有

$$\mathbb E[f]=\int f(\boldsymbol z)p(\boldsymbol z)\mathrm d\boldsymbol z\simeq \frac{Z_q}{Z_p} \frac{1}{L}\sum_{l=1}^L\frac{\tilde p(\boldsymbol z^{(l)})}{\tilde q(\boldsymbol z^{(l)})}f(\boldsymbol z^{(l)})\tag{10}$$

&emsp;&emsp;使⽤同样的样本集合来计算⽐值 $\frac{Z_p}{Z_q}$，结果为

$$\frac{Z_p}{Z_q} =\frac{1}{Z_p} \int \tilde p(\boldsymbol z)\mathrm d\boldsymbol z= \int \frac{\tilde p(\boldsymbol z)}{\tilde q(\boldsymbol z)}q(\boldsymbol z)\mathrm d\boldsymbol z\simeq \frac{1}{L}\sum_{l=1}^L\frac{\tilde p(\boldsymbol z^{(l)})}{\tilde q(\boldsymbol z^{(l)})}\tag{11}$$

&emsp;&emsp;因此有

$$\mathbb E[f]=\simeq \sum_{l=1}^L\frac{\frac{\tilde p(\boldsymbol z^{(l)})}{\tilde q(\boldsymbol z^{(l)})}}{\frac{\sum_m\tilde p(\boldsymbol z^{(m)})}{\tilde q(\boldsymbol z^{(m)})}}f(\boldsymbol z^{(l)})\tag{12}$$

&emsp;&emsp;重要性采样⽅法依赖于采样分布 $q(\boldsymbol z)$ 与所求的概率分布 $p(\boldsymbol z)$ 的匹配程度，经常出现 $p(\boldsymbol z)$ 变化剧烈，并且⼤部分值较大的区域集中于 $\boldsymbol z$ 空间的⼀个相对较⼩的区域，此时重要性权重 $\left \{ r_l \right \}$ 由⼏个具有较⼤值的权值控制，剩余的权值相对较⼩。因此，有效样本集数量会⽐使用的样本数量 $L$ ⼩很多，如果没有样本落在 $p(\boldsymbol z)f(\boldsymbol z)$ 较⼤的区域中，那么问题会更加严重。此时，$r_l 和 r_lf(\boldsymbol z^{(l)})$ 的表⾯上的⽅差可能很⼩，使期望的估计可能错得离谱。因此，重要性采样⽅法的主要缺点是它具有产⽣任意错误的结果的可能性，这也强调了采样分布 $q(\boldsymbol z)$ 不应该 在 $p(\boldsymbol z)$ 较⼤的区域中取得较⼩或为零的值。

#### 马尔科夫链蒙特卡罗 Markov Chain Monte Carlo

&emsp;&emsp;上⼀节所讨论的计算期望的重要性采样⽅法在⾼维空间中具有很⼤的局限性。因此，我们还需要⼀个更⼀般的框架，被称为马尔科夫链蒙特卡罗 (Markov chain Monte Carlo, MCMC)，可以从⼀⼤类概率分布中进⾏采样，并且很好地应对样本空间维度的增长。

&emsp;&emsp;和重要性采样相同，我们从提议分布 $q$ 中采样，但是我们记录的是当前状态 $\boldsymbol z^{(\tau)}$，以及依赖于这个当前状态的提议分布 $q(\boldsymbol z|\boldsymbol z^{(\tau)})$，从⽽样本序列 $\boldsymbol z^{(1)},\boldsymbol z^{(2)},\dots$ 组成了⼀个马尔科夫链。如果有 $p(\boldsymbol z)=\frac{\tilde p(\boldsymbol z)}{Z_p}$，那么对于任意的 $\boldsymbol z$ 值都可以计算 $\tilde p(\boldsymbol z)$，$Z_p$ 的值可能未知。提议分布本⾝很容易采样。在算法的每次迭代中，从提议分布中⽣成⼀个候选样本 $\boldsymbol z^\star$，然后根据⼀个恰当的准则接受这个样本。

&emsp;&emsp;在基本的 Metropolis 算法中，假定提议分布对称，即 $q(\boldsymbol z_A|\boldsymbol z_B)=q(\boldsymbol z_B|\boldsymbol z_A)$ 对于所有的 $\boldsymbol z_A$ 和 $\boldsymbol z_B$ 成⽴。这样，候选的样本被接受的概率为

$$A(\boldsymbol z^\star,\boldsymbol z^{(\tau)})=\min\left ( 1,\frac{\tilde p( \boldsymbol z^\star)}{\tilde p(\boldsymbol z^{(\tau)})} \right )\tag{11}$$

&emsp;&emsp;然后在单位区间 (0,1) 的均匀分布中随机选择⼀个数 $\mu$，如果 $A(\boldsymbol z^\star,\boldsymbol z^{(\tau)})>\mu$ 就接受这个样本。如果从 $\boldsymbol z^{(\tau)}$ 到 $\boldsymbol z^\star$ 引起了 $p(\boldsymbol z)$ 的值的增⼤，那么这个候选样本会被保留。如果候选样本被保留，那么 $\boldsymbol z^{(\tau+1)}=\boldsymbol z^\star$，否则候选样本点 $\boldsymbol z^\star$ 被丢弃， $\boldsymbol z^{(\tau+1)}$ 设置为 $\boldsymbol z^{(\tau)}$，然后从概率分布中再抽取⼀个候选样本。

&emsp;&emsp;⼀阶马尔科夫链被定义为⼀系列随机变量 $\boldsymbol z^{(1)},\dots,\boldsymbol z^{(M)}$，使得下⾯的条件独⽴性质对于 $m\in \left \{ 1,\dots,M-1 \right \}$ 成⽴

$$p(\boldsymbol z^{(m+1)}|\boldsymbol z^{(1)},\dots,\boldsymbol z^{(m)})=p(\boldsymbol z^{(m+1)}|\boldsymbol z^{(m)})\tag{12}$$

&emsp;&emsp;这可以表示成链形的有向图，之后可以按照下⾯的⽅式具体化⼀个马尔科夫链：给定初始变量的概率分布 $p(\boldsymbol z^{(0)})$，以及后续变量的条件概率，⽤转移概率 (transition probability) $T_m(\boldsymbol z^{(m)},\boldsymbol z^{(m+1)})\equiv p(\boldsymbol z^{(m+1)}|\boldsymbol z^{(m)})$ 的形式表示。如果对于所有的 $m$，转移概率都相同，那么这个马尔科夫链被称为同质的。对于⼀个特定的变量，边缘概率可以根据前⼀个变量的边缘概率⽤链式乘积的⽅式表示出来，形式为

$$p(\boldsymbol z^{(m+1)})=\sum_{\boldsymbol z^{(m)}}p(\boldsymbol z^{(m+1)}|\boldsymbol z^{(m)})p(\boldsymbol z^{(m)})\tag{13}$$

&emsp;&emsp;对于⼀个概率分布来说，如果马尔科夫链中的每⼀步都让这个概率分布保持不变，那么我们说这个概率分布关于马尔科夫链是不变的。对于⼀个转移概率为 $T(\boldsymbol z',\boldsymbol z)$ 的同质马尔科夫链来说，如果

$$p^\star(\boldsymbol z)=\sum_{\boldsymbol z'}T(\boldsymbol z',\boldsymbol z)p^\star(\boldsymbol z')\tag{14}$$

&emsp;&emsp;那么概率分布 $p^\star(\boldsymbol z)$ 是不变的。⼀个给定的马尔科夫链可能有多个不变的概率分布。确保所求的概率分布不变的⼀个充分条件是令转移概率满⾜细节平衡 (detailed balance) 性质，定义为

$$p^\star(\boldsymbol z)T(\boldsymbol z,\boldsymbol z')=p^\star(\boldsymbol z')T(\boldsymbol z',\boldsymbol z)\tag{15}$$

&emsp;&emsp;对特定的概率分布 $p^\star(\boldsymbol z)$ 成⽴。满⾜关于特定概率分布的细节平衡性质的转移概率会使得那个概率分布具有不变性，因为

$$\sum_{\boldsymbol z'}p^\star(\boldsymbol z')T(\boldsymbol z',\boldsymbol z)=\sum_{\boldsymbol z'}p^\star(\boldsymbol z')T(\boldsymbol z,\boldsymbol z')=p^\star(\boldsymbol z)\sum_{\boldsymbol z'}p(\boldsymbol z'|\boldsymbol z)=p^\star(\boldsymbol z)\tag{16}$$

&emsp;&emsp;满⾜细节平衡性质的马尔科夫链被称为可翻转的。
