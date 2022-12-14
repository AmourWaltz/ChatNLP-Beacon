# 5 概率分布 Probability Distribution

&emsp;&emsp;不确定性 (uncertainty) 是机器学习中一个重要概念，它一般由测量误差，温度漂移等因素引起，也可以由采样数据的有限性引起，在实际应用中有重要的指导意义。这时候就需要概率论的知识来描述模型方法的不确定性，从而提供了⼀个合理的框架来量化计算，同时需要概率分布来描述数据的分布，概率论也因此成了模式识别的⼀个中⼼基础。本文先介绍一些概率论的重要基础概念引入话题，再接着讲述两种重要的分布。

## 5.1 概率密度 Probability Density

&emsp;&emsp;虽然生活中我们更多使用离散概率描述事件，但是对于机器学习的推导，更多情况下需要分析的对象是连续变量，这时候就需要引出概率密度 (probability density)。如果⼀个实值变量 $x$ 的概率落在区间 $(x+\delta x)$ 的概率由 $p(x)\delta x$ 给出 $(\delta x\rightarrow0)$ ，那么 $p(x)$ 叫做 $x$ 的概率密度。$x$ 位于区间 $(a,b)$ 的概率为：

$$p(x\in(a,b))=\int_a^b p(x)\mathrm dx\tag1$$

&emsp;&emsp;由于概率是⾮负的，并且 $x$ 的值⼀定位于实轴上的某个位置，因此概率密度⼀定满足 $p(x)\ge 0$ 以及 $\int_{-\infty}^{\infty}p(x)\mathrm dx=1$。

&emsp;&emsp;在变量以⾮线性的形式变化的情况下，概率密度函数通过 Jacobian 因⼦变换为不同的函数形式。考虑两个变量 $x,y$ 具有如下非线性变化关系 $x=g(y)$，那么函数 $f(x)$ 就变成了 $\tilde{f}(y)=f(g(y))$。考虑⼀个概率密度函数 $p_x(x)$，它对应于变量 $y$ 的密度函数为 $p_y(y)$，对于很⼩的 $\delta x$ 的值，落在区间 $(x,x+\delta x)$ 内的观测会被变换到区间 $(y,y+\delta y)$ 中。其中 $p_x(x)\delta x\simeq p_y(y)\delta y$，那么有

$$p_y(y)=p_x(x)\left | \frac{dx}{dy} \right | = p_x(g(y))|{g}'(y)|\tag2$$

&emsp;&emsp;表明 $p_x(x)$ 和 $p_y(y)$ 是不同的密度函数，同时也表明概率密度最⼤值取决于变量的选择。

&emsp;&emsp;位于区间 $(-\infty, z)$ 的 $x$ 的概率由累积分布函数 (cumulative distribution function) 给出。定义为：

$$P(x)=\int_{-\infty }^zp(x)\mathrm dx\tag3$$

&emsp;&emsp;概率密度的加和规则和乘积 $y$ 规则的形式为

$$p(x)=\int p(x,y)\mathrm dy\tag4$$
$$p(x,y)=p(x|y)p(y)\tag5$$

&emsp;&emsp;加和规则的正确性可以⾮形式化地观察出来：把每个实数变量除以区间的宽度 $\Delta$，然后考虑这些区间上的概率分布，取极限 $\Delta\rightarrow0$，把求和转化为积分，就得到了预期的结果。

#### 期望和协方差 Expectation and Covariance

&emsp;&emsp;涉及到概率的⼀个重要的操作是寻找函数的加权平均值。在概率分布 $p(x)$ 下，函数 $f(x)$ 的平均值被称为 $f(x)$ 的期望 (expectation)，记作 $\mathbb{E}[f]$。连续变量期望以对应概率密度的积分表示为

$$\mathbb E[f]=\int p(x)f(x)\mathrm dx\tag6$$

&emsp;&emsp;如果考虑多变量函数的期望，我们可以使⽤下标来表明被平均的是哪个变量，例如 $\mathbb E_x[f(x,y)]$ 表示 $f(x,y)$ 关于 $x$ 的分布均值。需要注意，$\mathbb E_x[f(x,y)]$ 是 $y$ 的函数。

&emsp;&emsp;也可以考虑关于⼀个条件分布的条件期望 (conditional expectation)，即

$$\mathbb E_x[f(x)|y]=\int p(x|y)f(x)\mathrm dx\tag7$$

&emsp;&emsp;$f(x)$ 的⽅差 (variance) 被定义为

$$\mathrm{var}[f]=\mathbb E[(f(x)-\mathbb E[f(x)])^2]=\mathbb E[f(x)^2]-\mathbb E[f(x)]^2\tag8$$

&emsp;&emsp;对于两个随机变量 $x$ 和 $y$，协⽅差 (covariance) 被定义为

$$\mathrm{cov}[x,y]=\mathbb E\left [ \left \{ x-\mathbb E[x] \right \} \left \{ y-\mathbb E[y] \right \} \right ]=\mathbb E_{x,y}[xy]-\mathbb E[x]\mathbb E[y]\tag9$$

&emsp;&emsp;它表示在多⼤程度上 $x$ 和 $y$ 会共同变化。如果 $x$ 和 $y$ 相互独⽴，那么它们的协⽅差为 0。

&emsp;&emsp;在两个随机向量 $\boldsymbol x$ 和 $\boldsymbol y$ 的情形下，协⽅差是⼀个矩阵

$$\mathrm{cov}[\boldsymbol x,\boldsymbol y]=\mathbb E\left [ \left \{ \boldsymbol x-\mathbb E[\boldsymbol x] \right \} \left \{ \boldsymbol y^T-\mathbb E[\boldsymbol y^T] \right \} \right ]=\mathbb E_{\boldsymbol x,\boldsymbol y}[\boldsymbol x\boldsymbol y^T]-\mathbb E[\boldsymbol x]\mathbb E[\boldsymbol y^T]\tag{10}$$

## 5.2 贝叶斯概率 Bayes Theorem

&emsp;&emsp;在前几篇文章的最大似然估计中，我们对模型参数 $\boldsymbol w$ 进⾏推断时，在观察到数据之前，我们首先可以有⼀些关于参数 $\boldsymbol w$ 的假设，这以先验概率 $p(\boldsymbol w)$ 的形式给出；根据 $\boldsymbol w$ 表达预测目标数据 $\mathcal D$ 可以通过条件概率 $p(\mathcal D|\boldsymbol w)$ 表达，贝叶斯定理为

$$p(\boldsymbol w|\mathcal D)=\frac{p(\mathcal D|\boldsymbol w)p(\boldsymbol w)}{p(\mathcal D)}\tag{11}$$

&emsp;&emsp;它让我们能够通过后验概率 $p(\boldsymbol w|\mathcal D)$，在观测到 $\mathcal D$ 之后估计 $\boldsymbol w$ 的不确定性。贝叶斯定理右侧量 $p(\mathcal D|\boldsymbol w)$ 由观测数据集 $\mathcal D$ 来估计，可以被看成参数 $\boldsymbol w$ 的函数，称为似然函数 (likelihood function)。它表达了在不同的参数向量 $\boldsymbol w$ 下，观测数据出现的可能性的⼤⼩。贝叶斯公式右侧分母是⼀个归⼀化常数，确保了左侧的后验概率分布是⼀个合理的概率密度，积分为 1，对公式两侧关于 $\boldsymbol w$ 进⾏积分， 我们可⽤后验概率分布和似然函数来表达贝叶斯定理的分母

$$p(\mathcal D)=\int p(\mathcal D|\boldsymbol w)p(\boldsymbol w)\mathrm d\boldsymbol w\tag{12}$$

&emsp;&emsp;在似然函数 $p(\mathcal D|\boldsymbol w)$ 中，$\boldsymbol w$ 被认为是一个固定的参数，它的值由某种形式的估计来确定，这个估计的误差通过考察可能的数据集 $\mathcal D$ 的概率分布来得到，这也是我们使用大部分机器学习或深度学习模型常见的做法。而贝叶斯定理的观点则认为 $\boldsymbol w$ 是不确定的， 我们通过观察到的数据 $\mathcal D$ 估计 $\boldsymbol w$ 的概率分布来表达参数的不确定性。

&emsp;&emsp;贝叶斯观点的⼀个优点是天然包含先验概率。例如，假定投掷⼀枚普通的硬币 3 次，每次都是正⾯朝上，经典的最⼤似然模型在估计硬币正⾯朝上的概率时结果会是 1，表示所有未来的投掷都会是正⾯朝上。相反，⼀个带有合理的先验的贝叶斯的⽅法将不会得出这么极端的结论。在第一篇文章就讲过，如果使用带贝叶斯估计的模型相当于增加了 $L_2$ 正则化，可以有效防止过拟合，尤其在有限数据集上这种效果更为明显。

&emsp;&emsp;然而先验概率的选择是十分麻烦的，一种针对贝叶斯⽅法的⼴泛批评就是先验概率的选择通常是为了计算的⽅便⽽不是为了反映出任何先验的知识。实际上当先验选择不好的时候，贝叶斯⽅法有很⼤的可能性会给出错误的结果。

&emsp;&emsp;为了描述机器学习模型方法中的不确定性，我们需要研究一些特殊的概率分布。我们讨论的概率分布是在给定有限次观测 $\boldsymbol x_1,\boldsymbol x_2,...,\boldsymbol x_N$ 的前提下，对随机变量 $\boldsymbol x$ 的概率分布 $p(\boldsymbol x)$ 建模，这个问题被称为密度估计 (density estimation)，同时假定数据点是独⽴同分布。需要注意密度估计问题本质上是病态的，因为产⽣有限的观测数据集的概率分布有⽆限多种，任何在数据点 $\boldsymbol x_1,\boldsymbol x_2,...,\boldsymbol x_N$ 处⾮零的概率分布 $p(\boldsymbol x)$ 都是⼀个潜在的候选，所以我们才要研究一些很常见的，例如之前在线性回归和分类问题中所遇到的伯努利分布和高斯分布等典型概率分布的一般性质。

&emsp;&emsp;我们要研究的都是参数分布 (parametric distribution)，表示少量可调节的参数控制了整个概率分布，例如高斯分布的均值和方差。在之前的最大似然方法中，我们在给定观察数据集的条件下，确定最优的参数值。但在贝叶斯观点中，给定观察数据，我们需要先引⼊参数的先验分布，然后使⽤贝叶斯定理来计算对应后验概率分布。

&emsp;&emsp;这时候我们需要引入共轭先验 (conjugate prior)，它使后验概率分布的函数形式与先验概率相同，从而使贝叶斯分析得到极⼤的简化。例如，伯努利分布参数的共轭先验叫做 Beta 分布，多项式分布参数的共轭先验叫做狄利克雷分布 (Dirichlet distribution)，⾼斯分布均值的共轭先验是另⼀个⾼斯分布。关于伯努利分布等不再赘述，本文重点讨论这些共轭先验分布。

## 5.3 Beta 分布 Beta Distribution

&emsp;&emsp;首先设定二元随机变量 $x\in\left \{ 0,1 \right \}$，$x=1$ 的概率被记作参数 $\mu$，$x$ 的伯努利概率分布记作 $\mathrm {Bern}(x|\mu)=\mu^x(1-\mu)^{1-x}$，均值和方差分别为 $\mathbb E[x]=\mu,\mathrm {var}[x]=\mu(1-\mu)$，伯努利分布的参数 $\mu$ 的最⼤似然解也是数据集 $x=1$ 所占的⽐例，依旧是通过取负对数然后求到的方法求解。

&emsp;&emsp;我们也可以求解给定采样数据大小为 $N$ 的条件下，观测到 $x=1$ 的数量 $m$ 的概率分布，这被称为⼆项分布 (binomial distribution)。采样 $N$ 次以后，这个概率正比于 $\mu^m(1-\mu)^{N-m}$。为了得到归⼀化系数，就需要考虑所有采样 $N$ 次出现 $m$ 种情况的排列组合，⼆项分布可以写成

$$\mathrm{Bin}(m|N,\mu)=\frac{N!}{(N-m)!m!}\mu^m(1-\mu)^{N-m}=\mathcal C_N^m\mu^m(1-\mu)^{N-m}\tag{13}$$
其中 $\mathbb E[m]=\sum_{m=0}^{N}m\mathrm {Bin}(m|N,\mu)=N\mu, \mathrm{var}[m]=\sum_{m=0}^N(m-\mathbb E[m])^2\mathrm{Bin}(m|N,\mu)=N\mu(1-\mu)$。

&emsp;&emsp;为了⽤贝叶斯的观点看待这个问题，需要引⼊⼀个关于 $\mu$ 的先验概率分布 $p(\mu),(0\leq\mu\leq1)$。考虑⼀种形式简单的先验分布，我们注意到似然函数是某个因⼦与 $\mu^x(1-\mu)^{1-x}$ 的乘积的形式。如果我们选择⼀个正⽐于 $\mu$ 和 $1-\mu$ 的幂指数先验概率分布，那么后验概率分布（正⽐于先验和似然函数的乘积）就会有着与先验分布相同的形式，这个性质被叫做共轭性 (conjugacy)。我们把对应于伯努利分布的先验分布选择为 Beta 分布，定义为

$$\mathrm {Beta}(\mu|a,b)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}\tag{14}$$

&emsp;&emsp;其中 Gamma 函数 $\Gamma(x)\equiv\int_{0}^{\infty}\mu^{x-1}e^{-\mu}\mathrm d\mu$，使用分部积分法可得 $\Gamma(x+1)=x\Gamma(x)$。公式 (13) 保证了 Beta 分布式的归⼀化， 即

$$\int_0^1\mathrm{Beta}(\mu|a,b)\mathrm d\mu=1\tag{15}$$
其中均值和方差分别为 $\mathbb E[\mu]=\frac{a}{a+b}$，$\mathrm{var}[\mu]=\frac{ab}{(a+b)^2(a+b+1)}$。$\mu$ 的后验概率分布可以通过把 Beta 先验与二项似然函数相乘得到后验概率形式

$$p(\mu|m,l,a,b)\propto \mu^{m+a-1}(1-\mu)^{l+b-1}\tag{16}$$

&emsp;&emsp;再通过归⼀化得到

$$p(\mu|m,l,a,b)=\frac{\Gamma(m+a+l+b)}{\Gamma(m+a)\Gamma(l+b)}u^{m+a-1}(1-\mu)^{l+b-1}\tag{17}$$
其中 $l=N-m$。实际上，它仅仅是另⼀个 Beta 分布，这也反映出先验关于似然函数的共轭性质。我们看到，如果⼀个数据集⾥有 $m$ 次观测为 $x=1$，有 $l$ 次观测为 $x=0$，那么从先验概率到后验概率，$a$ 的值变⼤了 $m$，$b$ 的值变⼤了 $l$。我们可以简单地把先验概率中的超参数 $a$ 和 $b$ 分别看成 $x=1$ 和 $x=0$ 的有效观测数 (effective number of observation)。

&emsp;&emsp;另外，如果我们接下来观测到更多的数据，那么后验概率分布可以扮演先验概率的角⾊。假想每次取⼀个观测值，然后在每次观测之后更新当前的后验分布。在每个阶段，上一次观测后的后验概率作为先验概率，与似然函数相乘再进行归一化得到新的后验概率。后验概率是⼀个 Beta 分布，对于 $x=1$ 和 $x=0$ 的观测总数（先验的和实际的）和参数 $a$ 和 $b$ 相关。观测到⼀个 $x=1$ 对应于把 $a$ 的值加 $1$，⽽观测到 $x=0$ 会使 $b$ 加 $1$。下图反应了这一过程，先验概率为 Beta 分布，参数为 $a=2,b=2$，似然函数由二项分布公式 (13) 给出，其中 $N=m=1$，对应于 $x=1$ 的⼀次观测，从而后验概率分布为 Beta 分布，参数为 $a=3,b=2$。

<div align=center>
<img src="images/5_3_beta1.png" width="80%"/>
</div>

&emsp;&emsp;如果接受了贝叶斯观点，那么学习过程中的顺序⽅法可以⾃然得出，它与先验和似然函数的选择⽆关，只取决于数据独⽴同分布的假设。顺序⽅法每次使⽤⼀个或⼀⼩批观测值，然后在使⽤下⼀个观测值之前丢掉它们。在实时学习场景中，输⼊为⼀个稳定持续的数据流，模型必须在观测到所有数据之前就进⾏预测，同时也不需要把所有的数据都存储到内存⾥，在很多机器学习和深度学习模型中都广泛使用，例如小批量梯度下降算法。

&emsp;&emsp;在上面的例子中，如果我们想尽可能好地预测下⼀次的输出，那应该估计给定观测数据集 $\mathcal D$ 的情况下 $x$ 的预测分布，预测形式为

$$p(x=1|\mathcal D)=\int_0^1p(x=1|\mu)p(\mu|\mathcal D)\mathrm d\mu=\int_0^1\mu p(\mu|\mathcal D)\mathrm d\mu=\mathbb E[\mu|\mathcal D]\tag{18}$$

&emsp;&emsp;使用公式 (17) 并代入 Beta 分布的均值，可得

$$p(x=1|\mathcal D)=\frac{a+m}{a+m+b+l}\tag{19}$$

&emsp;&emsp;这个结果可简单表述为 $x=1$ 的观测结果（包括实际观测值和假想的先验值）所占的⽐例。当数据集⽆限⼤时 $m,l\rightarrow \infty$，此时公式 (19) 变成了最⼤似然的结果。所以，贝叶斯的结果和最⼤似然的结果在数据集规模趋于⽆穷时会一致。对于有限规模的数据集，$\mu$ 的后验均值总是位于先验均值和 $\mu$ 的最⼤似然估计之间。

&emsp;&emsp;观察 Beta 分布的方差也会发现，如果 $a\rightarrow\infty 或者 b\rightarrow\infty$，那么⽅差就趋于零。这个现象表明，随着观测到越来越多的数据，后验概率表示的不确定性将会持续下降。考虑⼀个⼀般的贝叶斯推断问题，参数为 $\boldsymbol \theta$，以及对应的观测数据集 $\mathcal D$，由联合概率分布 $p(\boldsymbol \theta|\mathcal D)$ 描述。我们有 $\mathbb E_{\boldsymbol \theta}[\boldsymbol \theta]=\mathbb E_{\mathcal D}[\mathbb E_{\boldsymbol \theta}[\boldsymbol \theta|\mathcal D]]$，其中

$$\mathbb E_{\boldsymbol \theta}[\boldsymbol \theta]=\int p(\boldsymbol \theta)\boldsymbol \theta\mathrm d\boldsymbol \theta\tag{20}$$
$$\mathbb E_{\mathcal D}[\mathbb E_{\boldsymbol \theta}[\boldsymbol \theta|\mathcal D]]=\int\left \{ \int p(\boldsymbol \theta)\boldsymbol \theta\mathrm d\boldsymbol \theta \right \} p(\mathcal D)\mathrm d\mathcal D\tag{21}$$

&emsp;&emsp;这表明 $\boldsymbol \theta$ 的后验均值在产⽣数据集的整个分布上做平均等于 $\boldsymbol \theta$ 的先验均值。类似地，我们可以证明

$$\mathrm{var}_{\boldsymbol \theta}[\boldsymbol \theta]=\mathbb E_{\mathcal D}[\mathrm{var}_{\boldsymbol \theta}[\boldsymbol \theta|\mathcal D]]+\mathrm{var}_{\boldsymbol \theta}[\mathbb E_{\boldsymbol \theta}[\boldsymbol \theta|\mathcal D]]\tag{22}$$

&emsp;&emsp;左侧项是 $\boldsymbol \theta$ 的先验⽅差。右侧第⼀项是 $\boldsymbol \theta$ 的平均后验⽅差，第⼆项是 $\boldsymbol \theta$ 的后验均值的⽅差。这个结果表明，平均来看 $\boldsymbol \theta$ 的后验⽅差⼩于先验⽅差，后验均值的⽅差越⼤，这个⽅差的减⼩就越⼤，当然这个结论也是在平均的情况下成立。

## 5.4 狄利克雷分布 Dirichlet Distribution

&emsp;&emsp;二项分布是描述只有两种取值的变量的概率分布，实际中我们更多遇到多种可能取值的离散变量，我们采用 one-hot 编码方式，输入向量 $\boldsymbol x$ 在 $K=6$ 个类别中属于类别 $C_3$ 时就可以表示为 $(0,0,1,0,0,0)^T$，对应 $x_3=1$ 的情况。如果我们⽤参数 $\mu_k$ 表示 $x_k=1$ 的概率，那么 $\boldsymbol x$ 的分布就是

$$p(\boldsymbol x|\boldsymbol \mu)=\prod_{k=1}^{K}\mu_k^{x_k}\tag{23}$$
其中 $\boldsymbol \mu=\left ( \mu_1,...,\mu_K \right )^T$，参数 $\mu_k$ 满⾜ $\mu_k\geq 0$ 和 ${\textstyle \sum_{k}^{}} \mu_k=1$，可以看成伯努利分布对于多个输出的推⼴，反之伯努利也可以看作 $K=2$ 时的特例，很容易推导出归一化

$$\sum_{\boldsymbol x}p(\boldsymbol x|\boldsymbol \mu)=\sum_{k=1}^{K}\mu_k=1\tag{24}$$

以及期望形式

$$\mathbb E[\boldsymbol x|\boldsymbol \mu]=\sum_{\boldsymbol x}p(\boldsymbol x|\boldsymbol \mu)\boldsymbol x=(\mu_1,...,\mu_K)^T=\boldsymbol \mu\tag{25}$$

&emsp;&emsp;现在考虑⼀个有 $N$ 个独⽴观测值 $\boldsymbol x_1,...,\boldsymbol x_N$ 的数据集 $\mathcal D$。对应的似然函数的形式为

$$p(\mathcal D|\boldsymbol \mu)=\prod_{n=1}^{N}\prod_{k=1}^{K}\mu_k^{x_{nk}}=\prod_{k=1}^{K}\mu_k^{( {\textstyle \sum_{n}^{}}x_{nk}) }=\prod_{k=1}^{K}\mu_k^{m_k}\tag{26}$$

&emsp;&emsp;可以看到似然函数对于 $N$ 个数据点的依赖通过 $m_k$，表示观测到 $x_k=1$ 的次数。这被称为分布的充分统计量 (sufficient statistics)。

&emsp;&emsp;为了找到 $\boldsymbol \mu$ 的最⼤似然解，我们需要关于 $\mu_k$ 最⼤化 $\ln p(\mathcal D|\boldsymbol \mu)$，并且要限制 ${\textstyle \sum_{k}^{}} \mu_k=1$。这可以通过拉格朗⽇乘数实现，在第二篇文章中有提及，即最⼤化

$$\sum_{k=1}^Km_k\ln \mu_k+\lambda \left ( \sum_{k=1}^K\mu_k-1 \right )\tag{27}$$
令其关于 $\mu_k$ 的导数等于 0，我们有 $\mu_k=-\frac{m_k}{\lambda}$，代入限制条件可得 $\lambda = -N$，所以最大似然解为 $\mu_k^{ML}=\frac{m_k}{N}$，它是 $N$ 次观测中，$x_k=1$ 的观测所占的⽐例。

&emsp;&emsp;现在考虑 $m_1,...,m_K$ 在参数 $\boldsymbol \mu$ 和观测总数 $N$ 条件下的联合分布，其形式为

$$\mathrm{Mult}(m_1,m_2,...,m_K|\boldsymbol \mu,N)=\frac{N!}{\prod_{k=1}^{K} m_k}\prod_{k=1}^{K}\mu_k^{m_k}\tag{28}$$
称为多项式分布 (multinomial distribution)。归⼀化系数是把 $N$ 个物体分成⼤⼩为 $m_1,...,m_K$ 的 $K$ 组的⽅案总数。

&emsp;&emsp;同样地，对应于多项式分布的参数 $\left \{ u_k \right \}$ 的一组共轭先验分布为

$$p(\boldsymbol \mu | \boldsymbol \alpha)\propto \prod_{k=1}^{K}\mu_k^{\alpha_k-1}\tag{29}$$

&emsp;&emsp;关于 $\mu_k$ 的限制条件同上，其中 $\boldsymbol \alpha=(\alpha_1,...,\alpha_K)^T$。概率的归⼀化形式为

$$\mathrm {Dir}(\boldsymbol \mu|\boldsymbol \alpha)=\frac{\Gamma(\alpha_0)}{\prod_{k=1}^{K}\Gamma(\alpha_k)}\prod_{k=1}^{K}\mu_k^{\alpha_k-1}\tag{30}$$
这被称为狄利克雷分布 (Dirichlet distribution)，其中 $\alpha_0=\sum_{k=1}^{K}\alpha_k$。⽤似然函数乘以先验，我们得到了参数 $\left \{ u_k \right \}$ 的后验分布，形式为

$$p(\boldsymbol \mu|\mathcal D,\boldsymbol \alpha)\propto p(\mathcal D|\boldsymbol \mu)p(\boldsymbol \mu|\boldsymbol \alpha)\propto\prod_{k=1}^{K}\mu_k^{\alpha_k+m_k-1}\tag{31}$$

&emsp;&emsp;我们看到后验分布的形式又变成了狄利克雷分布，这说明，狄利克雷分布确实是多项式分布的共轭先验。通过确定归一化系数，可得

$$p(\boldsymbol \mu|\mathcal D,\boldsymbol \alpha)=\mathrm {Dir}(\boldsymbol \mu|\boldsymbol \alpha+\boldsymbol m)=\frac{\Gamma(\alpha_0+N)}{\prod_{k=1}^{K}\Gamma(\alpha_k+m_k)}\prod_{k=1}^{K}\mu_k^{\alpha_k+m_k-1}\tag{32}$$
其中，$\boldsymbol m=(m_1,...,m_K)^T$。 与⼆项分布的先验的 Beta 分布相同，我们可以把狄利克雷分布的参数 $\alpha_k$ 看成 $x_k=1$ 时的有效观测数。

## 5.5 高斯分布 Gaussian Distribution

&emsp;&emsp;⾼斯分布，也称正态分布，是连续随机变量的模型中应用最广泛的分布。在第一篇文章就简单讨论过一元高斯分布的性质，本文重点讨论多元高斯分布，对于 $D$ 维向量 $\boldsymbol x$，其多元高斯分布形式是

$$\mathcal N(\boldsymbol x|\boldsymbol \mu,\boldsymbol \Sigma)=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\left | \boldsymbol \Sigma \right |^{\frac{1}{2}} }\exp\left \{ -\frac{1}{2}(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu) \right \}\tag{33}$$
其中，$\boldsymbol \mu$ 是⼀个 $D$ 维均值向量，$\boldsymbol \Sigma$ 是⼀个 $D\times D$ 的协⽅差矩阵，$\left | \boldsymbol \Sigma \right |$ 是 $\boldsymbol \Sigma$ 的⾏列式。

&emsp;&emsp;⾼斯分布会在许多问题中产⽣，可以从多个不同的角度来理解。例如，我们已知对于⼀元实值随机变量，使熵取得最⼤值时就服从于⾼斯分布，代表了数据分布的最大不确定性，该性质对于多元⾼斯分布也成⽴。

&emsp;&emsp;多个随机变量的和也会产⽣⾼斯分布，根据中⼼极限定理 (central limit theorem)，一般情况下，⼀组随机变量之和的概率分布随着随机变量数量的增加⽽逐渐趋向⾼斯分布。给定 $N$ 个一元变量 $x_1,\dots,x_N$，都服从于区间 $[0,1]$ 上的均匀分布，然后考虑其均值 $\frac{1}{N}(x_1+\dots+x_N)$ 的分布。在实际应⽤中，随着 $N$ 的增加，分布会趋于⾼斯分布。根据这个结论，前一章所讲的⼆项分布（⼆元随机变量 $x$ 在 $N$ 次观测中出现次数 $m$ 的分布）将会在 $N\rightarrow\infty$ 时趋向⾼斯分布，如下图所示，对于不同的 $N$ 值，$N$ 个均匀分布的随机变量均值的直⽅图。 可以看到随着 $N$ 的增加，分布趋向于⾼斯分布。

<div align=center>
<img src="images/5_5_gauss1.png" width="85%"/>
</div>

#### 高斯分布的熵值计算

&emsp;&emsp;对于给定的协⽅差，具有最⼤熵的多元概率分布是⾼斯分布。概率分布 $p(\boldsymbol x)$ 的熵为

$$H[\boldsymbol x]=-\int p(\boldsymbol x)\ln p(\boldsymbol x)\mathrm d\boldsymbol x\tag{34}$$

&emsp;&emsp;针对 $p(\boldsymbol x)$ 最大化 $H[\boldsymbol x]$，其中 $p(\boldsymbol x)$ 满足：1. 可归⼀化；2. 具有均值 $\boldsymbol \mu$；3. 具有协⽅差 $\boldsymbol \Sigma$。

&emsp;&emsp;我们使用拉格朗日乘数法来引入限制条件，需要注意我们需要三个不同的拉格朗日乘数器，首先引入一个单变量 $\lambda$ 针对归一化条件，然后是一个 $D$ 维向量 $\boldsymbol m$ 针对均值条件，以及一个 $D\times D$ 维的矩阵 $\boldsymbol L$ 针对协方差限制，这样我们有

$$\begin{align} \tilde{\boldsymbol H}[p]=&-\int p(\boldsymbol x)\ln p(\boldsymbol x)\mathrm d\boldsymbol x+\lambda\left (\int p(\boldsymbol x)\mathrm d\boldsymbol x-1\right )+\boldsymbol m^T\left (\int p(\boldsymbol x)\boldsymbol x\mathrm d\boldsymbol x - \boldsymbol \mu\right )\\&+\mathrm {Tr}\left \{ \left (\boldsymbol L\int p(\boldsymbol x)(\boldsymbol x-\boldsymbol \mu)(\boldsymbol x-\boldsymbol \mu)^T\mathrm d\boldsymbol x-\boldsymbol \Sigma\right )\right \} \end{align}\tag{35}$$

&emsp;&emsp;将公式 (3) 中所有积分项写在一起，定义被积函数 $F(\boldsymbol x) 为p(\boldsymbol x)\ln p(\boldsymbol x)+\lambda\left ( p(\boldsymbol x)-1\right )+\boldsymbol m^T\left (p(\boldsymbol x)\boldsymbol x - \boldsymbol \mu\right )+\mathrm {Tr} \left \{\boldsymbol L\left (p(\boldsymbol x)(\boldsymbol x-\boldsymbol \mu)(\boldsymbol x-\boldsymbol \mu)^T-\boldsymbol \Sigma\right )\right \}$，该函数被称为 $p(\boldsymbol x)$ 的泛函数，根据泛函数的导数定义，我们令 $\frac{\partial F(\boldsymbol x)}{\partial p(\boldsymbol x)} = 0$，可得

$$\frac{\partial F(\boldsymbol x)}{\partial p(\boldsymbol x)} = -1-\ln p(\boldsymbol x)+\lambda+\boldsymbol m^T\boldsymbol x+\mathrm{Tr}\left \{ \boldsymbol L(\boldsymbol x-\boldsymbol \mu)(\boldsymbol x - \boldsymbol \mu)^T \right \}\tag{36}$$
$$\begin{align} p(\boldsymbol x)&=\exp\left \{ \lambda-1 + \boldsymbol m^T\boldsymbol x+(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol L(\boldsymbol x - \boldsymbol \mu) \right \}\\ &=\exp\left \{ \lambda-1 +\left(\boldsymbol x-\boldsymbol \mu+\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m\right )^T\boldsymbol L\left (\boldsymbol x - \boldsymbol \mu+\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m\right )+ \boldsymbol \mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m\right \} \end{align}\tag{37}$$

&emsp;&emsp;令 $\boldsymbol y=\boldsymbol x-\boldsymbol \mu+\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m$，然后代入归一化和均值的约束条件，可得以下两个公式

$$\int \exp\left \{ \lambda-1+\boldsymbol y^T\boldsymbol L\boldsymbol y+\boldsymbol\mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m \right \}\left ( \boldsymbol y+\boldsymbol \mu-\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m \right )\mathrm d\boldsymbol y=\boldsymbol \mu\tag{38}$$ 
$$\int \exp\left \{ \lambda-1+\boldsymbol y^T\boldsymbol L\boldsymbol y+\boldsymbol\mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m \right \}\mathrm d\boldsymbol y=1\tag{39}$$
其中 $\exp\left \{ \lambda-1+\boldsymbol y^T\boldsymbol L\boldsymbol y+\boldsymbol\mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m \right \}\boldsymbol y$ 为奇函数，公式 (7) 代入 (6) 可得 $-\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m=\boldsymbol 0$，进一步可得 $\boldsymbol m=\boldsymbol 0$。证明一元高斯分布的熵值最大化时我们利用已知条件 $\int \exp(-x^2)\mathrm dx=\sqrt{\pi}$，多元高斯分布的推导需要利用矩阵的指数函数的性质，依次按照一元高斯分布的方式计算矩阵每一个元素，然后再求和，就可以得出 $\boldsymbol L=-\frac{1}{2}\boldsymbol \Sigma$，并且有 $\lambda-1=\ln\left \{ \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{|\boldsymbol \Sigma|^{\frac{1}{2}}} \right \}$。全部代入，就可以得出当概率分布 $p(\boldsymbol x)$ 的熵最大时，其函数形式满足高斯分布 (1)。

&emsp;&emsp;求⾼斯分布的熵，可以得到 $H[x]=\frac{1}{2}\left \{ 1+\ln (2\pi\sigma^2) \right \}$。

#### 高斯分布的几何形式

&emsp;&emsp;考虑⾼斯分布的⼏何形式，⾼斯分布对于 $\boldsymbol x$ 的依赖是通过下⾯的⼆次型

$$\Delta^2=(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu)\tag{40}$$

&emsp;&emsp;这个⼆次型出现在指数位置上，$\Delta$ 被叫做 $\boldsymbol \mu$ 和 $\boldsymbol x$ 之间的马⽒距离 (Mahalanobis distance)。 当 $\boldsymbol \Sigma$ 是单位矩阵时，就变成了欧式距离。对于 $\boldsymbol x$ 空间中这个⼆次型是常数的曲⾯，⾼斯分布也是常数。

&emsp;&emsp;⾸先，我们注意到矩阵 $\boldsymbol \Sigma$ 可以取为对称矩阵，⽽不失⼀般性，因为任何⾮对称项都会从指数中消失。这一点可以通过把协⽅差矩阵的逆矩阵 $\boldsymbol \Sigma^{-1}=\boldsymbol \Lambda$ 写成对称矩阵 $\boldsymbol \Lambda^S$ 和反对称矩阵 $\boldsymbol \Lambda^A$ 的和，证明反对称项不会出现在⾼斯分布的指数项中，其中 $\Lambda_{ij}^S=\frac{\Lambda_{ij}+\Lambda_{ji}}{2}$，$\Lambda_{ij}^A=\frac{\Lambda_{ij}-\Lambda_{ji}}{2}$。多元高斯分布的指数项可以写作 $\frac{1}{2}\sum_{i=1}^{D}\sum_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}(x_j-\mu_j)$，由于反对称矩阵具有性质 $\Lambda_{ij}^A=-\Lambda_{ji}^A$，易证指数项展开后反对称矩阵每一项都会相互抵消掉。由于对称矩阵的逆矩阵还是对称矩阵，因此我们也可以令协⽅差矩阵为对称矩阵⽽不失⼀般性。

&emsp;&emsp;现在考虑协⽅差矩阵的特征向量⽅程 $\boldsymbol \Sigma\boldsymbol \mu_i=\lambda_i\boldsymbol \mu_i$，由于 $\boldsymbol \Sigma$ 是实对称矩阵，因此它的特征值也是实数，并且特征向量可以被选成单位正交的，即 $\boldsymbol \mu_i^T\boldsymbol \mu_j=I_{ij}$，其中 $\boldsymbol I$ 表示单位矩阵。协⽅差矩阵 $\boldsymbol \Sigma$ 可以表示成特征向量的展开形式，$\boldsymbol \Sigma=\sum_{i=1}^{D}\lambda_i\boldsymbol \mu_i\boldsymbol \mu_i^T,\boldsymbol \Sigma^{-1}=\sum_{i=1}^{D}\frac{1}{\lambda_i}\boldsymbol \mu_i\boldsymbol \mu_i^T$，代入公式 (8)，二次型就变成了 $\Delta^2=\sum_{i=1}^{D}\frac{y_i^2}{\lambda_i^2}$，定义 $y_i=\boldsymbol \mu_i^T(\boldsymbol x-\boldsymbol \mu)$，可以把 $y_i$ 表示成单位正交向量 $\boldsymbol \mu_i$ 关于原始的 $x_i$ 坐标经过平移和旋转后形成的新的坐标系。

&emsp;&emsp;当⼆次型为常数时，它表示的是一个曲⾯。如果所有的特征值 $\lambda_i$ 都是正数，那么这些曲⾯表示椭球⾯，椭球中⼼位于 $\boldsymbol \mu$，椭球的轴的⽅向沿着 $\boldsymbol \mu_i$，沿着轴向的缩放因⼦为 $\lambda_i^{\frac{1}{2}}$，如下图所示。对于定义的⾼斯分布，我们要求协⽅差矩阵的所有特征值 $\lambda_i$ 严格⼤于零，否则分布将不能被正确地归⼀化。

<div align=center>
<img src="images/5_5_gauss2.png" width="65%"/>
</div>

#### 高斯分布的均值和方差

&emsp;&emsp;现在考察⾼斯分布的矩，已知参数 $\boldsymbol \mu$ 和 $\boldsymbol \Sigma$。⾼斯分布下 $\boldsymbol x$ 的期望为

$$\begin{align} \mathbb E[\boldsymbol x]&=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\left | \boldsymbol \Sigma \right |^{\frac{1}{2}} }\int\exp\left \{ -\frac{1}{2}(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu) \right \}\boldsymbol x\mathrm d\boldsymbol x\\ &=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\left | \boldsymbol \Sigma \right |^{\frac{1}{2}} }\int\exp\left \{ -\frac{1}{2}\boldsymbol z^T\boldsymbol \Sigma^{-1}\boldsymbol z \right \}(\boldsymbol z+\boldsymbol \mu)\mathrm d\boldsymbol z \end{align}\tag{41}$$

&emsp;&emsp;使用等量代换后可以发现，$\exp\left \{ -\frac{1}{2}\boldsymbol z^T\boldsymbol \Sigma^{-1}\boldsymbol z\right \}\boldsymbol z$ 是 $\boldsymbol z$ 的奇函数，结合归一化条件，很容易得出 $\mathbb E[\boldsymbol x]=\boldsymbol \mu$。

&emsp;&emsp;我们现在考虑⾼斯分布的⼆阶矩。在⼀元变量的情形下，⼆阶矩由 $\mathbb E[x^2]=\mu^2+\sigma^2$ 给出。对于多元⾼斯分布，有 $D^2$ 个由 $\mathbb E[x_ix_j]$ 给出的⼆阶矩，可以聚集在⼀起组成矩阵 $\mathbb E[\boldsymbol x\boldsymbol x^T]$。这个矩阵可以写成

$$\begin{align} \mathbb E[\boldsymbol x\boldsymbol x^T]&=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\left | \boldsymbol \Sigma \right |^{\frac{1}{2}} }\int\exp\left \{ -\frac{1}{2}(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu) \right \}\boldsymbol x\boldsymbol x^T\mathrm d\boldsymbol x\\ &=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\left | \boldsymbol \Sigma \right |^{\frac{1}{2}} }\int\exp\left \{ -\frac{1}{2}\boldsymbol z^T\boldsymbol \Sigma^{-1}\boldsymbol z \right \}(\boldsymbol z+\boldsymbol \mu)(\boldsymbol z+\boldsymbol \mu)^T\mathrm d\boldsymbol z \end{align}\tag{42}$$

&emsp;&emsp;再次应用等量代换，并且涉及到 $\boldsymbol z$ 的奇函数项将由于对称性⽽变为零。项 $\boldsymbol \mu\boldsymbol \mu^T$ 是常数，可以从积分中拿出，然后使用归一化条件积分项为 1。考虑涉及到 $\boldsymbol z\boldsymbol z^T$ 的项，我们可以再次使⽤协⽅差矩阵的特征向量展开，以及特征向量集合的完备性，得到 $y_i=\boldsymbol\mu_j^T\boldsymbol z$，因此有

$$\begin{align} \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\left | \boldsymbol \Sigma \right |^{\frac{1}{2}} }\int\exp\left \{ -\frac{1}{2}\boldsymbol z^T\boldsymbol \Sigma^{-1}\boldsymbol z \right \}\boldsymbol z\boldsymbol z^T\mathrm d\boldsymbol z&=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\left | \boldsymbol \Sigma \right |^{\frac{1}{2}} }\sum_{i=1}^D\sum_{j=1}^D\boldsymbol \mu_i\boldsymbol \mu_j\int\exp\left \{ -\sum_{k=1}^D\frac{y_k^2}{2\lambda_k} \right \}y_iy_j\mathrm d\boldsymbol y\\ &=\sum_{i=1}^D\boldsymbol \mu_i\boldsymbol \mu_i^T\lambda_i=\boldsymbol \Sigma \end{align}\tag{43}$$

&emsp;&emsp;第二步等式右侧积分所有 $i\ne j$ 由于对称性会等于零，最终我们有 $\mathbb E[\boldsymbol x\boldsymbol x^T]=\boldsymbol\mu\boldsymbol\mu^T+\boldsymbol\Sigma$。把均值减掉，就给出了随机变量 $\boldsymbol x$ 的协⽅差 (covariance)，定义为

$$\mathrm{var}[\boldsymbol x]=\mathbb E\left [ \left ( \boldsymbol x-\mathbb E[\boldsymbol x] \right )\left ( \boldsymbol x-\mathbb E[\boldsymbol x] \right )^T \right ]=\boldsymbol \Sigma\tag{44}$$

&emsp;&emsp;$\boldsymbol \Sigma$ 因此被称为协方差矩阵。虽然⾼斯分布被⼴泛⽤作概率密度模型，但它有⼀些巨⼤的局限性。考虑分布中⾃由参数的数量，通常一个对称协⽅差矩阵 $\boldsymbol \Sigma$ 有 $\frac{D(D+1)}{2}$ 个独⽴参数，$\boldsymbol \mu$ 中有另外 $D$ 个独⽴参数，总计有 $\frac{D(D+3)}{2}$ 个参数。参数数量随着 $D$ 平⽅式增长，因此对⼤矩阵进⾏求逆等运算会⽆法计算。一种解决这个问题的⽅式是限制协⽅差矩阵的形式。如果我们考虑对角 (diagonal) 矩阵，即 $\boldsymbol \Sigma=\mathrm{diag}(\sigma_i^2)$，那么 我们就只有 $2D$ 个独⽴参数。由于常数高斯密度对应的曲面轮廓是与轴对齐的椭球型，我们可以进⼀步限制协⽅差矩阵成正⽐于单位矩阵，即 $\boldsymbol \Sigma=\sigma^2\boldsymbol I$，被称为各向同性 (isotropic) 的协⽅差，这时模型只有 $D+1$ 个独⽴参数，并且常数概率密度是球⾯，但是这样做也极⼤地限制了概率分布的描述形式。

#### 条件高斯分布

&emsp;&emsp;多元⾼斯分布的⼀个重要性质是，如果两组变量是联合⾼斯分布，那么以⼀组变量为条件，另⼀组变量同样是⾼斯分布。类似地，任何⼀个变量的边缘分布也是⾼斯分布。

&emsp;&emsp;⾸先考虑条件概率的情形，假设 $\boldsymbol x$ 是⼀个服从⾼斯分布 $\mathcal N(\boldsymbol x|\boldsymbol \mu,\boldsymbol \Sigma)$ 的 $D$ 维向量。我们把 $\boldsymbol x$ 划分成两个不相交的⼦集 $\boldsymbol x_a$ 和 $\boldsymbol x_b$。不失⼀般性，我们可以令 $\boldsymbol x_a$ 为 $\boldsymbol x$ 的前 $M$ 个分量，令 $\boldsymbol x_b$ 为剩余 $D-M$ 个分量，因此可以划分 $\boldsymbol x=\begin{pmatrix} \boldsymbol x_a \\ \boldsymbol x_b \end{pmatrix}$，$\boldsymbol \mu=\begin{pmatrix} \boldsymbol \mu_a \\ \boldsymbol \mu_b \end{pmatrix}$，$\boldsymbol \Sigma=\begin{pmatrix} \boldsymbol \Sigma_{aa}&\boldsymbol \Sigma_{ab} \\ \boldsymbol \Sigma_{ba}&\boldsymbol \Sigma_{bb} \end{pmatrix}$。由于⾼斯分布的⼀些性质使⽤协方差的逆矩阵，也称作精度矩阵 (precision matrix) 表示，形式会更简单。我们定义 $\boldsymbol \Lambda\equiv\boldsymbol \Sigma^{-1}$，以及 $\boldsymbol \Lambda=\begin{pmatrix} \boldsymbol \Lambda_{aa}&\boldsymbol \Lambda_{ab} \\ \boldsymbol \Lambda_{ba}&\boldsymbol \Lambda_{bb} \end{pmatrix}$，由于对称矩阵的逆矩阵也是对称矩阵，因此 $\boldsymbol \Lambda_{aa}$ 和 $\boldsymbol \Lambda_{bb}$ 也是对称的，⽽ $\boldsymbol \Lambda_{ab}=\boldsymbol \Lambda_{ba}^T$。

&emsp;&emsp;⾸先，我们寻找条件概率分布 (conditional probability distribution) $p(\boldsymbol x_a|\boldsymbol x_b)$ 的表达式，根据概率的乘积规则，条件分布可以根据联合分布 $p(\boldsymbol x)=p(\boldsymbol x_a,\boldsymbol x_b)$ 很容易计算出来。只需把 $\boldsymbol x_b$ 固定为观测值，然后对得到的表达式进⾏归⼀化，得到 $\boldsymbol x_a$ 的⼀个概率分布。我们考虑⾼斯分布指数项中出现的⼆次型，然后在计算的最后阶段重新考虑归⼀化系数，因此有

$$\begin{align} -\frac{1}{2}(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu) = &-\frac{1}{2}(\boldsymbol x_a-\boldsymbol \mu_a)^T\boldsymbol \Lambda_{aa}(\boldsymbol x_a-\boldsymbol \mu_a)-\frac{1}{2}(\boldsymbol x_a-\boldsymbol \mu_a)^T\boldsymbol \Lambda_{ab}(\boldsymbol x_b-\boldsymbol \mu_b)\\ &-\frac{1}{2}(\boldsymbol x_b-\boldsymbol \mu_b)^T\boldsymbol \Lambda_{ba}(\boldsymbol x_a-\boldsymbol \mu_a)-\frac{1}{2}(\boldsymbol x_b-\boldsymbol \mu_b)^T\boldsymbol \Lambda_{bb}(\boldsymbol x_b-\boldsymbol \mu_b) \end{align}\tag{44}$$
将其看作 $\boldsymbol x_a$ 的函数，这又是⼀个⼆次型，因此对应的条件分布 $p(\boldsymbol x_a|\boldsymbol x_b)$ 是⾼斯分布。然后通过观察公式 (13) 找到 $p(\boldsymbol x_a|\boldsymbol x_b)$ 的均值和协⽅差的表达式。⼀个⼀般的⾼斯分布 $\mathcal N(\boldsymbol x|\boldsymbol \mu,\boldsymbol \Sigma)$ 的指数项可以写成

$$-\frac{1}{2}(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu)=-\frac{1}{2}\boldsymbol x^T\boldsymbol \Sigma^{-1}\boldsymbol x+\boldsymbol x^T\boldsymbol \Sigma^{-1}\boldsymbol\mu+C\tag{45}$$
其中 $C$ 为常数，并且我们用到 $\boldsymbol \Sigma$ 的对称性。如果采用公式 (14) 的表示方式，我们可以令 $\boldsymbol x$ 的⼆阶项系数矩阵等于协⽅差矩阵的逆矩阵 $\boldsymbol \Sigma^{-1}$，令 $\boldsymbol x$ 的线性项系数等于 $\boldsymbol \Sigma^{-1} \boldsymbol \mu$，我们就可以得到 $\boldsymbol \mu$。整理 $\boldsymbol x_a$ 的所有二阶项，可得条件概率分布的协方差为 $\boldsymbol \Sigma_{a|b}=\boldsymbol \Lambda_{aa}^{-1}$，再整理 $\boldsymbol x_a 的所有一阶项为 \boldsymbol x_a^T\left \{ \boldsymbol \Lambda_{aa}\boldsymbol \mu_a-\boldsymbol \Lambda_{ab}(\boldsymbol x_b-\boldsymbol \mu_b) \right \}$，使用对称性和已求解的协方差矩阵，可得 $\boldsymbol \mu_{a|b}=\boldsymbol \mu_{a}-\boldsymbol\Lambda_{aa}^{-1}\boldsymbol\Lambda_{ab}(\boldsymbol x_b-\boldsymbol \mu_b)$，或者记为 $\boldsymbol \mu_{a|b}=\boldsymbol \mu_{a}+\boldsymbol\Sigma_{ab}\boldsymbol\Sigma_{bb}^{-1}(\boldsymbol x_b-\boldsymbol \mu_b)$，$\boldsymbol \Sigma_{a|b}=\boldsymbol \Sigma_{aa}-\boldsymbol\Sigma_{ab}\boldsymbol\Sigma_{bb}^{-1}\boldsymbol\Sigma_{ba}$。

#### 边缘高斯分布

&emsp;&emsp;我们已证明如果联合分布 $p(\boldsymbol x_a,\boldsymbol x_b)$ 是⾼斯分布，那么条件概率分布 $p(\boldsymbol x_a|\boldsymbol x_b)$ 也是高斯分布。现在讨论边缘概率分布 (marginal probability distribution)

$$p(\boldsymbol x_a)=\int p(\boldsymbol x_a,\boldsymbol x_b)\mathrm d\boldsymbol x_b\tag{46}$$
这也是⼀个⾼斯分布。和之前⼀样，我们估计这个概率分布的策略是观察联合分布指数项的⼆次型，然后找出边缘分布 $p(\boldsymbol x_a)$ 的均值和协⽅差。联合分布的⼆次型可以参考条件高斯分布表示成公式 (13) 分块精度矩阵的形式，然后积分出 $\boldsymbol x_b$，⾸先考虑涉及 $\boldsymbol x_b$ 的项，然后配出平⽅项，我们有

$$-\frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b+\boldsymbol x_b^T\boldsymbol m=-\frac{1}{2}(\boldsymbol x_b-\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)^T\boldsymbol \Lambda_{bb}(\boldsymbol x_b-\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)+-\frac{1}{2}\boldsymbol m^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m~~~~~~(16)\\ \boldsymbol m=\boldsymbol \Lambda_{bb}\boldsymbol \mu_{b}-\boldsymbol \Lambda_{ba}(\boldsymbol x_a-\boldsymbol \mu_{a})\tag{48}$$

&emsp;&emsp;$\boldsymbol x_b$ 的相关项已经被转化为了⼀个⾼斯分布的标准⼆次型， 即公式 (16) 右侧的第⼀项，加上⼀个与 $\boldsymbol x_b$ ⽆关但与 $\boldsymbol x_a$ 相关的项。当我们取这个⼆次型作为⾼斯分布的指数项时，我们看到公式 (15) 要求的关于 $\boldsymbol x_b$ 的积分形式为

$$\int\exp\left\{-\frac{1}{2}(\boldsymbol x_b-\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)^T\boldsymbol \Lambda_{bb}(\boldsymbol x_b-\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)\right\}\mathrm d\boldsymbol x_b\tag{49}$$

&emsp;&emsp;这是⼀个在未归⼀化的⾼斯分布上做的积分，因此结果是归⼀化系数的倒数。⾼斯分布的系数与均值⽆关，只依赖于协⽅差矩阵的⾏列式。因此通过配平⽅项的⽅法，我们能够积分出 $\boldsymbol x_b$， 这样唯⼀剩余的与 $\boldsymbol x_a$ 相关的项就是公式 (16) 右侧的最后⼀项。把这⼀项与公式 (13) 中余下的与 $\boldsymbol x_a$ 相关的项结合，我们有可以写出 $\boldsymbol x_a$ 的二次项的形式为

$$-\frac{1}{2}\boldsymbol x_a^T(\boldsymbol \Lambda_{aa}-\boldsymbol \Lambda_{ab}\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba})\boldsymbol x_a+\boldsymbol x_a^T(\boldsymbol \Lambda_{aa}-\boldsymbol \Lambda_{ab}\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba})\boldsymbol\mu_a+C\tag{50}$$

&emsp;&emsp;所以可得 $\boldsymbol \Sigma_{aa}=(\boldsymbol \Lambda_{aa}-\boldsymbol \Lambda_{ab}\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba})^{-1}$，以及 $\boldsymbol \Sigma_{aa}(\boldsymbol \Lambda_{aa}-\boldsymbol \Lambda_{ab}\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba})\boldsymbol \mu_{a}=\boldsymbol \mu_{a}$，边缘概率 $p(\boldsymbol x_a)$ 的均值和协⽅差为 $\mathbb E[\boldsymbol x_a]=\boldsymbol\mu_a$，$\mathrm{cov}[\boldsymbol x_a]=\boldsymbol\Sigma_{aa}$。

&emsp;&emsp;我们关于分块⾼斯边缘分布和条件分布的结果可总结如下，

&emsp;&emsp;给定⼀个联合⾼斯分布 $\mathcal N(\boldsymbol x|\boldsymbol \mu,\boldsymbol \Sigma)$，其中 $\boldsymbol \Lambda\equiv\boldsymbol \Sigma^{-1}$，且 $\boldsymbol x=\begin{pmatrix} \boldsymbol x_a \\ \boldsymbol x_b \end{pmatrix}$，$\boldsymbol \mu=\begin{pmatrix} \boldsymbol \mu_a \\ \boldsymbol \mu_b \end{pmatrix}$，$\boldsymbol \Sigma=\begin{pmatrix} \boldsymbol \Sigma_{aa}&\boldsymbol \Sigma_{ab} \\ \boldsymbol \Sigma_{ba}&\boldsymbol \Sigma_{bb} \end{pmatrix}$，$\boldsymbol \Lambda=\begin{pmatrix} \boldsymbol \Lambda_{aa}&\boldsymbol \Lambda_{ab} \\ \boldsymbol \Lambda_{ba}&\boldsymbol \Lambda_{bb} \end{pmatrix}$。此时

&emsp;&emsp;条件概率分布 $p(\boldsymbol x_a|\boldsymbol x_b)=\mathcal N(\boldsymbol x_a|\boldsymbol \mu_{a|b},\boldsymbol \Lambda_{aa}^{-1})$， $\boldsymbol \mu_{a|b}=\boldsymbol \mu_{a}-\boldsymbol\Lambda_{aa}^{-1}\boldsymbol\Lambda_{ab}(\boldsymbol x_b-\boldsymbol \mu_b)$，

&emsp;&emsp;边缘概率分布 $p(\boldsymbol x_a)=\mathcal N(\boldsymbol x_a|\boldsymbol \mu_{a},\boldsymbol \Sigma_{aa})$。

#### 高斯变量的贝叶斯定理

&emsp;&emsp;这⾥我们重新定义给定⼀个⾼斯边缘分布 $p(\boldsymbol x)$ 和⼀个⾼斯条件分布 $p(\boldsymbol y|\boldsymbol x)$，其中 $p(\boldsymbol y|\boldsymbol x)$ 的均值是 $\boldsymbol x$ 的线性函数，协⽅差与 $\boldsymbol x$ ⽆关，与前文所描述的 $\boldsymbol x_a$ 与 $\boldsymbol x_b$ 的分布类似，这是线性⾼斯模型 (linear Gaussian model) 的⼀个例⼦，我们想找到边缘概率分布 $p(\boldsymbol x)$ 和条件概率分布 $p(\boldsymbol x|\boldsymbol y)$。首先令边缘概率分布和条件概率分布的形式如下

$$p(\boldsymbol x)=\mathcal N(\boldsymbol x|\boldsymbol \mu,\boldsymbol \Lambda^{-1})\tag{51}$$ 
$$p(\boldsymbol y|\boldsymbol x)=\mathcal N(\boldsymbol y|\boldsymbol A\boldsymbol x+\boldsymbol b,\boldsymbol L^{-1})\tag{52}$$
其中，$\boldsymbol \mu,\boldsymbol A,\boldsymbol b$ 控制均值参数，$\boldsymbol \Lambda,\boldsymbol L$ 是精度矩阵。如果 $\boldsymbol x$ 的维度为 $M$，$\boldsymbol y$ 的维度为 $D$，那么矩阵 $\boldsymbol A$ 的⼤⼩为 $D\times M$。

&emsp;&emsp;⾸先，我们寻找 $\boldsymbol x$ 和 $\boldsymbol y$ 联合分布的表达式，定义 $\boldsymbol z=\begin{pmatrix} \boldsymbol x \\ \boldsymbol y \end{pmatrix}$，联合概率分布的对数为

$$\ln p(\boldsymbol z)=\ln p(\boldsymbol x)+\ln p(\boldsymbol y|\boldsymbol x)=-\frac{1}{2}(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol \Lambda(\boldsymbol x-\boldsymbol \mu) -\frac{1}{2}(\boldsymbol x-\boldsymbol A\boldsymbol x-\boldsymbol b)^T\boldsymbol L(\boldsymbol x-\boldsymbol A\boldsymbol x-\boldsymbol b) +C\tag{53}$$

&emsp;&emsp;与之前相同，这是 $\boldsymbol z$ 的分量的⼀个⼆次函数，因此 $p(\boldsymbol z)$ 是⼀个⾼斯分布，为了找到这个⾼斯分布的精度矩阵，我们考虑公式 (22) 的第⼆项，可以写成

$$\begin{align} &-\frac{1}{2}\boldsymbol x^T(\boldsymbol \Lambda+\boldsymbol A^T\boldsymbol L\boldsymbol A)\boldsymbol x-\frac{1}{2}\boldsymbol y^T\boldsymbol L\boldsymbol y+\frac{1}{2}\boldsymbol y^T\boldsymbol L\boldsymbol A\boldsymbol x+\frac{1}{2}\boldsymbol x^T\boldsymbol A^T\boldsymbol L\boldsymbol y\\ &=-\frac{1}{2}\begin{pmatrix} \boldsymbol x \\ \boldsymbol y \end{pmatrix}^T \begin{pmatrix} \boldsymbol \Lambda+\boldsymbol A^T\boldsymbol L\boldsymbol A& -\boldsymbol A^T\boldsymbol L\\ -\boldsymbol L\boldsymbol A&\boldsymbol L \end{pmatrix}\begin{pmatrix} \boldsymbol x \\ \boldsymbol y \end{pmatrix}=-\frac{1}{2}\boldsymbol z^T\boldsymbol R\boldsymbol z \end{align}\tag{54}$$

&emsp;&emsp;因此 $\boldsymbol z$ 上的⾼斯分布的精度矩阵为 $\begin{pmatrix} \boldsymbol \Lambda+\boldsymbol A^T\boldsymbol L\boldsymbol A& -\boldsymbol A^T\boldsymbol L\\ -\boldsymbol L\boldsymbol A&\boldsymbol L \end{pmatrix}$，再使用类似公式 (14) 的方法，找到 $\boldsymbol z$ 的线性项，可以求出均值为 $\begin{pmatrix} \boldsymbol \Lambda+\boldsymbol A^T\boldsymbol L\boldsymbol A& -\boldsymbol A^T\boldsymbol L\\ -\boldsymbol L\boldsymbol A&\boldsymbol L \end{pmatrix}^{-1} \begin{pmatrix} \boldsymbol \Lambda\boldsymbol \mu-\boldsymbol A^T\boldsymbol L\boldsymbol b\\ \boldsymbol L\boldsymbol b \end{pmatrix}$，化简后为 $\begin{pmatrix} \boldsymbol \mu \\ \boldsymbol A\boldsymbol\mu+\boldsymbol b \end{pmatrix}$。

&emsp;&emsp;接下来寻找 $p(\boldsymbol y)$ 边缘分布的表达式，这个边缘分布是通过对 $\boldsymbol x$ 积分得到的。而条件分布 $p(\boldsymbol x|\boldsymbol y)$ 则可以通过贝叶斯定理推断出来，这里直接给出结论，有兴趣可自行推导。

&emsp;&emsp;给定 $\boldsymbol x$ 的⼀个边缘⾼斯分布，以及在给定 $\boldsymbol x$ 的条件下 $\boldsymbol y$ 的条件⾼斯分布，形式为分别如公式 (20) (21) 所示，那么 $\boldsymbol y$ 的边缘分布以及给定 $\boldsymbol y$ 的条件下 $\boldsymbol x$ 的条件分布为

$$p(\boldsymbol y)=\mathcal N(\boldsymbol y|\boldsymbol A\boldsymbol \mu+\boldsymbol b,\boldsymbol L^{-1}+\boldsymbol A\boldsymbol \Lambda^{-1}\boldsymbol A^T)\tag{55}$$ 
$$p(\boldsymbol x|\boldsymbol y)=\mathcal N(\boldsymbol x|\boldsymbol \Sigma\left \{ \boldsymbol A^T\boldsymbol L(\boldsymbol y-\boldsymbol b)+\boldsymbol \Lambda\boldsymbol \mu\right \},\boldsymbol \Sigma)\tag{56}$$
其中 $\boldsymbol \Sigma=(\boldsymbol \Lambda+\boldsymbol A^T\boldsymbol L\boldsymbol A)^{-1}$。

#### 高斯分布的最大似然估计

&emsp;&emsp;给定⼀个数据集 $\boldsymbol X=(\boldsymbol x_1,\dots,\boldsymbol x_N)^T$，其中观测 $\left \{ \boldsymbol x_n \right \}$ 假定是独⽴地从多元⾼斯分布中抽取的。我们使⽤最⼤似然法估计分布的参数，对数似然函数为

$$\ln p(\boldsymbol X|\boldsymbol\mu,\boldsymbol\Sigma)=-\frac{ND}{2}\ln(2\pi)-\frac{N}{2}\ln|\boldsymbol\Sigma|-\frac{1}{2}\sum_{n=1}^{N}(\boldsymbol x_n-\boldsymbol\mu)^T\boldsymbol\Sigma^{-1}(\boldsymbol x_n-\boldsymbol\mu)\tag{57}$$

&emsp;&emsp;似然函数对数据集的依赖只通过 $\sum_{n=1}^{N}\boldsymbol x_n,\sum_{n=1}^{N}\boldsymbol x_n\boldsymbol x_n^T$ 体现，这被称为⾼斯分布的充分统计量。对数似然函数关于 $\boldsymbol\mu$ 的导数为 $\sum_{n=1}^{N}\boldsymbol\Sigma^{-1}(\boldsymbol x_n-\boldsymbol\mu)$，令导数为 0，可得均值的最大似然估计 $\boldsymbol\mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}\boldsymbol x_n$，这是数据点的观测集合的均值。关于 $\boldsymbol\Sigma$ 的最⼤化更加复杂，但我们可以利⽤了对称性和正定性的限制，得到 $\boldsymbol\Sigma_{ML}=\frac{1}{N}\sum_{n=1}^{N}(\boldsymbol x_n-\boldsymbol\mu_{ML})(\boldsymbol x_n-\boldsymbol\mu_{ML})^T$。

#### 高斯分布的贝叶斯推断

&emsp;&emsp;最⼤似然框架给出了对于参数 $\boldsymbol\mu$ 和 $\boldsymbol\Sigma$ 的点估计。现在通过引⼊这些参数的先验分布，介绍贝叶斯⽅法。⾸先，考虑⼀个⼀元⾼斯随机变量 $x$，我们假设⽅差 $\sigma^2$ 是已知的。我们从⼀组 $N$ 次观测 $\boldsymbol X=\left \{ \boldsymbol x_1,\dots,\boldsymbol x_N\right \}^T$ 中推断均值 $\mu$。此时似然函数是给定 $\mu$ 的情况下，观测数据集出现的概率，可以看成 $\mu$ 的函数，由下式给出

$$p(\boldsymbol X|\mu)=\prod_{n=1}^{N}p(x_n|\mu)=\frac{1}{(2\pi\sigma^2)^{\frac{N}{2}}}\exp\left \{ -\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2 \right \}\tag{58}$$

&emsp;&emsp;似然函数 $p(\boldsymbol X|\mu)$ 不是 $\mu$ 的概率密度，没有被归⼀化。似然函数的形式为 $\mu$ 的⼆次型指数形式。因此如果我们把先验分布 $p(\mu)$ 选成⾼斯分布，那么它就是似然函数的⼀个共轭分布，因为对应后验概率是两个 $\mu$ 的⼆次函数指数乘积，也是⼀个⾼斯分布。于是我们令先验概率分布为

$$p(\mu)=\mathcal N(\mu|\mu_0,\sigma_0^2)\tag{59}$$

&emsp;&emsp;从⽽后验概率为

$$p(\mu|\boldsymbol X)\propto p(\boldsymbol X|\mu)p(\mu)\tag{60}$$

&emsp;&emsp;进⾏对指数项的完全平⽅项等简单计算，可以证明后验概率的形式为

$$p(\mu|\boldsymbol X)=\mathcal N(\mu|\mu_N,\sigma_N^2)\tag{61}$$
其中 $\mu_N=\frac{\sigma^2}{N\sigma_0^2+\sigma^2}\mu_0+\frac{N\sigma_0^2}{N\sigma_0^2+\sigma^2}\mu_{ML}$，$\frac{1}{\sigma_N^2}=\frac{1}{\sigma_0^2}+\frac{N}{\sigma^2}$。可以看到后验分布的均值是先验均值 $\mu_0$ 和最⼤似然解 $\mu_{ML}$ 的折中，这与之前所讨论后验分布介于先验和最大似然之间的结论是一致。如果观测数据点 $N=0$，那么后验均值就变成了先验均值。对于 $N\rightarrow\infty$，后验均值由最⼤似然解给出。类似地，考虑后验分布⽅差的结果，⽅差的倒数可以用来表示精度，并且精度是可以相加的，因此后验概率的精度等于先验的精度加上每⼀个观测数据点所贡献的精度。当我们增加观测数据点时，精度持续增加，对应于后验分布的⽅差持续减少。没有观测数据点 $N=0$ 时，后验方差为先验⽅差， ⽽如果 $N\rightarrow\infty$，⽅差 $\sigma_N^2$ 趋于零， 后验分布在最⼤似然解附近变成了⽆限⼤的尖峰。对于有限的 $N$ 值， 如果我们取极限 $\sigma_0^2\rightarrow\infty$，先验⽅差会变为⽆穷⼤，后验均值就变成了最⼤似然结果，⽽后验⽅差为 $\sigma_N^2=\frac{\sigma^2}{N}$。

&emsp;&emsp;现在我们假设均值是已知的，来推断⽅差。同之前⼀样，如果我们选择先验分布的共轭形式，那么计算将会得到极⼤的简化，使用方差倒数 $\lambda=\frac{1}{\sigma^2}$。$\lambda$ 的似然函数为

$$p(\boldsymbol X|\mu)=\prod_{n=1}^{N}p(x_n|\mu,\lambda^{-1})\propto\lambda^{\frac{N}{2}}\exp\left \{ -\frac{\lambda}{2}\sum_{n=1}^{N}(x_n-\mu)^2 \right \}\tag{62}$$

对应的共轭先验因此应该正⽐于 $\lambda$ 的幂指数，也正⽐于指数项中 $\lambda$ 的线性函数。这对应于 Gamma 分布，定义为

$$\mathrm{Gam}(\lambda|a,b)=\frac{1}{\Gamma(a)}b^a\lambda^{a-1}\exp(-b\lambda)\tag{63}$$

$\Gamma(a)$ 是前一篇文章中的 Gamma 函数，保证了公式的归⼀化。如果 $a>0$，那么 Gamma 分布有⼀个有穷的积分。如果 $a\geq1$，那么分布本⾝是有穷的。Gamma 分布的均值和⽅差为 $\mathbb E[\lambda]=\frac{a}{b},\mathrm{var}[\lambda]=\frac{a}{b^2}$，如下图是一些不同的 $a$ 和 $b$ 的分布。

<div align=center>
<img src="images/5_5_gauss3.png" width="65%"/>
</div>

&emsp;&emsp;考虑⼀个先验分布 $\mathrm{Gam}(\lambda|a_0,b_0)$。如果我们乘以公式 (29) 的似然函数，那么得到后验

$$p(\lambda|\boldsymbol X)\propto\lambda^{a_0-1}\lambda^{\frac{N}{2}}\exp\left \{-b_0\lambda -\frac{\lambda}{2}\sum_{n=1}^{N}(x_n-\mu)^2 \right \}\tag{64}$$

&emsp;&emsp;我们可以把它看成形式为 $\mathrm{Gam}(\lambda|a_N,b_N)$ 的 Gamma 分布，其中 $a_N=a_0+\frac{N}{2}$，$b_N=b_0+\frac{1}{2}\sum_{n=1}^{N}(x_n-\mu)^2=b_0+\frac{N}{2}\sigma^2_{ML}$，其中 $\sigma^2_{ML}$ 是⽅差的最⼤似然估计。我们看到观测 $N$ 个数据点的效果是把系数 $a$ 的值增加 $2N$。因此我们可以把先验分布中的参数 $a_0$ 看成 $2a_0$ 个 “有效” 先验观测。类似地，$N$ 个数据点对参数 $b$ 贡献了 $\frac{N}{2}\sigma^2_{ML}$，其中 $\sigma^2_{ML}$ 是⽅差，同样可以把先验分布中的 $b_0$ 看成 “有效” 先验观测。对于指数族分布来说，把共轭先验看成有效假想数据点是⼀个很通⽤的思想。

## 5.6 混合高斯模型 Mixture of Gaussians

&emsp;&emsp;虽然⾼斯分布有⼀些重要的分析性质，但是当它遇到实际数据集时，也会有巨⼤的局限性。因为高斯分布是单峰的，而实际情况往往有很多是多峰分布，可以通过将基本的概率分布进⾏线性组合，这样的叠加⽅法被称为混合模型 (mixture distributions)。⾼斯分布的线性组合可以给出相当复杂的概率密度形式，通过使⽤⾜够多的⾼斯分布，并且调节它们的均值和⽅差以及线性组合的系数，⼏乎所有的连续概率密度都能够以任意的精度近似。

&emsp;&emsp;考虑 $K$ 个⾼斯概率密度的叠加，形式为

$$p(\boldsymbol x)=\sum_{k=1}^{K}\pi_k\mathcal N(\boldsymbol x|\boldsymbol \mu_k,\boldsymbol \Sigma_k)\tag{65}$$

&emsp;&emsp;这被称为混合⾼斯 (mixture of Gaussians)。每⼀个⾼斯概率密度 $\mathcal N(\boldsymbol x|\boldsymbol \mu_k,\boldsymbol \Sigma_k)$ 被称为混合分布的⼀个成分，并且有⾃⼰的均值 $\boldsymbol \mu_k 和协⽅差 \boldsymbol \Sigma_k$。参数 $\pi_k$ 被称为混合系数，如果对公式 (32) 两侧关于 $\boldsymbol x$ 进⾏积分，因为各个⾼斯成分都是归⼀化的，我们可以得到 $\sum_{k=1}^K\pi_k=1$，同时为了满⾜ $p(\boldsymbol x)\geq0$，我们进一步限制 $0\leq\pi_k\leq1$。

&emsp;&emsp;根据概率的加和规则和乘积规则，边缘概率密度为 $p(\boldsymbol x)=\sum_{k=1}^{K}p(k)p(\boldsymbol x|k)$，其中我们把 $\pi_k=p(k)$ 看成选择第 $k$ 个成分的先验概率，把密度 $\mathcal N(\boldsymbol x|\boldsymbol \mu_k,\boldsymbol \Sigma_k)=p(\boldsymbol x|k)$ 看成以 $k$ 为条件的 $\boldsymbol x$ 的概率。

&emsp;&emsp;⾼斯混合分布的形式由参数 $\boldsymbol \pi,\boldsymbol \mu,\boldsymbol \Sigma$ 控制，其中 $\boldsymbol \pi=\left \{ \pi_1,\dots,\pi_K \right \}$，$\boldsymbol \mu=\left \{ \boldsymbol \mu_1,\dots,\boldsymbol \mu_K \right \}$，$\boldsymbol \Sigma=\left \{ \boldsymbol \Sigma_1,\dots,\boldsymbol \Sigma_K \right \}$。⼀种确定这些参数值的⽅法是使⽤最⼤似然法。 根据公式 (30)，对数似然函数为

$$\ln p(\boldsymbol X|\boldsymbol\pi,\boldsymbol\mu,\boldsymbol\Sigma)=\sum_{n=1}^N\ln\left\{ \sum_{k=1}^{K}\pi_k\mathcal N(\boldsymbol x_n|\boldsymbol \mu_k,\boldsymbol \Sigma_k) \right \}\tag{66}$$

&emsp;&emsp;这种情形⽐⼀元⾼斯分布复杂得多，因为对数中存在⼀个求和式，这就导致参数的最⼤似然解不再有⼀个封闭形式的解析解。可以使用期望最大化 (expectation maximization) 算法来求解，这个在第 15 章节将会详细讨论。




