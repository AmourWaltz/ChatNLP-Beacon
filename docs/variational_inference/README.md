# 5 变分法 Variational Inference

## 5.1 变分法的物理推导

## 5.2 变分近似推断

&emsp;&emsp;概率模型的应⽤中⼀个重要任务是在给定观测数据变量 $\boldsymbol X$ 的条件下，计算潜在变量 $\boldsymbol Z$ 的后验概率分布 $p(\boldsymbol Z|\boldsymbol X)$ 以及这个概率分布的期望，模型也可能是⼀个贝叶斯模型，其中任何未知参数都有⼀个先验概率分布，并且被整合到了潜在变量 $\boldsymbol Z$ 中。在贝叶斯神经网络一节中，使用变分法的思路就是，把数据集 $\mathcal D$ 看做观测数据，模型参数 $\boldsymbol w$ 看做潜在变量，然后计算参数的后验概率分布 $p(\boldsymbol w|\mathcal D)$。之所以要这样做，是因为对于实际应⽤中的许多模型来说，计算后验概率分布或者它的期望是不可⾏的，贝叶斯神经网络引入变分法就是因为，神经网络中的非线性变换使得似然分布的指数项不是均值参数的二次型，后验概率形式复杂，不能解析地求出，其他原因也有可能是潜在空间的维度太⾼不便于计算。

&emsp;&emsp;在这种情况下，就需要借助近似⽅法，本文介绍一些确定性近似方法，对于大规模数据很适⽤，我们称其为变分推断 (Variational Inference)。变分法很早就应用于一些物理问题，例如牛顿使用变分法解决最速下降曲线的的问题。这里就需要再提及泛函数的概念，泛函数也是一个函数映射，不同的是，它以⼀个已知的函数作为输⼊，通过泛函数映射后的值作为输出。比如信息熵 $H[p]$，它的输⼊是⼀个概率分布 $p(x)$，返回 $H[p]=-\int p(x)\ln p(x)\mathrm dx$ 作为输出。

&emsp;&emsp;如果需要最优化的量是一个泛函，那我们就可以通过变分法寻找近似解。寻找近似解就是假设一个函数或分布，限制其可选择函数形式的范围，然后去逼近真实的函数或分布，在概率推断应用中，限制条件的形式是可分解的假设。我们在上一节最后讨论 EM 算法时也引入了变分的概念，这里考虑一种更一般地情况，即假设一个贝叶斯模型，每个参数都有一个先验概率分布，这个模型也可以有潜在变量参数，这里我们就把潜在变量和模型参数作为一个整体去考虑，记作 $\boldsymbol Z$，然后把观测变量即数据集记作 $\boldsymbol X$，如果 $\boldsymbol X,\boldsymbol Z$ 都是已知的，那么其联合概率分布 $p(\boldsymbol X,\boldsymbol Z)$ 也很容易得出，但是为了验证模型参数在数据集上的性能，我们更关心参数的后验概率分布 $p(\boldsymbol Z|\boldsymbol X)$ 以及在数据集上的表现 $p(\boldsymbol X)$，为了得到这些近似，就需要将对数边缘概率分解，使用一个近似分布 $q(\boldsymbol Z)$ 逼近后验概率分布，可得

$$\ln p(\boldsymbol X)=\mathcal L(q)+\mathrm {KL}(q||p)\tag1$$
$$\mathcal L(q)=\int q(\boldsymbol Z)\ln \left \{ \frac{p(\boldsymbol X,\boldsymbol Z)}{q(\boldsymbol Z)} \right \}\mathrm d\boldsymbol Z\tag2$$ 
$$\mathrm {KL}(q||p)=-\int q(\boldsymbol Z)\ln \left \{ \frac{p(\boldsymbol Z|\boldsymbol X)}{q(\boldsymbol Z)} \right \}\mathrm d\boldsymbol Z\tag3$$

&emsp;&emsp;同样地，我们通过关于概率分布 $q(\boldsymbol Z)$ 的最优化来使下界 $\mathcal L(q)$ 达到最⼤值，这等价于最⼩化 KL 散度。如果允许任意选择 $q(\boldsymbol Z)$，那么下界的最⼤值出现在 KL 散度等于零的时刻，此时 $q(\boldsymbol Z)$ 等于后验概率分布 $p(\boldsymbol Z|\boldsymbol X)$，这一步又回到贝叶斯神经网络一节中的变分方法，我们的目的就是最小化近似概率分布 $q(\boldsymbol Z)$ 与后验分布 $p(\boldsymbol Z|\boldsymbol X)$ 的 KL 散度。在这个过程中，需要假定对真实概率分布是不可操作的。

&emsp;&emsp;于是我们转⽽考虑概率分布 $q(\boldsymbol Z)$ 的⼀个受限制的类别，然后寻找这个类别中使得 KL 散度达到最⼩值的概率分布。先充分限制 $q(\boldsymbol Z)$ 可以取得的概率分布的范围，使得这个范围中的所有概率分布都是可处理的概率分布。同时还要保证这个范围充分广泛灵活，从⽽能够提供对真实后验概率分布的⼀个⾜够好的近似。施加限制条件的唯⼀⽬的是为了计算⽅便，并且在这个限制条件下，应使⽤尽可能丰富的近似概率分布。特别地，对于⾼度灵活的概率分布来说，没有 “过拟合” 现象，使⽤灵活的近似仅仅使得我们更好地近似真实的后验概率分布。

&emsp;&emsp;限制近似概率分布的范围的⼀种⽅法是使⽤参数概率分布 $q(\boldsymbol Z|\boldsymbol \theta)$，它由参数集合 $\boldsymbol \theta$ 控制。这样下界 $\mathcal L(q)$ 变成了 $\boldsymbol \theta$ 的函数，例如在贝叶斯神经网络一节中，我们假设 $q(\boldsymbol Z)$ 服从高斯分布，那么参数集合 $\boldsymbol \theta$ 就是高斯分布的均值和方差。

&emsp;&emsp;这里考虑另一种分解近似 (factorized approximations) 方法，我们直接限制概率分布 $q(\boldsymbol Z)$ 的范围，将 $\boldsymbol Z$ 的元素划分成 M 个互不相交的组，记作 $\boldsymbol Z_i$，其中 $i=1,\dots,M$，然后假定 $q$ 分布关于这些分组可以进行分解，即

$$q(\boldsymbol Z)=\prod_{i=1}^{M}q_i(\boldsymbol Z_i)\tag4$$

&emsp;&emsp;在所有具有公式 (4) 的形式的概率分布 $q(\boldsymbol Z)$ 中，我们现在寻找下界 $\mathcal L(q)$ 最⼤的概率分布。对 $\mathcal L(q)$ 关于所有的概率分布 $q_i(\boldsymbol Z_i)$ 进⾏⼀个⾃由形式的变分优化，通过关于每个因⼦进⾏最优化来完成整体的最优化。⾸先将公式 (4) 代⼊公式 (2)，然后分离出依赖于⼀个因⼦ $q_i(\boldsymbol Z_i)$ 的项，记作 $q_i$，这样有

$$\begin{align} \mathcal L(q)&=\int \prod_{i}q_i\left \{ \ln p(\boldsymbol X,\boldsymbol Z)-\sum_i \ln q_i \right \}\mathrm d\boldsymbol Z\\ &=\int q_i\left \{ \int\ln p(\boldsymbol X,\boldsymbol Z)\prod_{i\ne j} q_i \mathrm d\boldsymbol Z_i\right \}\mathrm d\boldsymbol Z_i-\int q_j \ln q_j \mathrm d\boldsymbol Z_j+C\\ &=\int q_j \ln \tilde p(\boldsymbol X,\boldsymbol Z_j)\mathrm d\boldsymbol Z_j-\int q_j \ln q_j\mathrm d\boldsymbol Z_j+C\\ \end{align}\tag4$$ 
$$\ln \tilde p(\boldsymbol X,\boldsymbol Z_j)=\mathbb E_{i\ne j}[\ln p(\boldsymbol X,\boldsymbol Z)]+C\tag6$$
其中 $C$ 为常数，现在假设保持 $\left \{ q_{i\ne j} \right \}$ 固定，关于概率分布 $q_j(\boldsymbol Z_j)$ 的所有可能的形式最⼤化公式 (5) 中的 $\mathcal L(q)$。因为公式 (5) 是 $q_j(\boldsymbol Z_j)$ 和 $\tilde p(\boldsymbol X,\boldsymbol Z_j)$ 之间的 KL 散度负值。因此最⼤化公式 (5) 等价于最⼩化 KL 散度，且最⼩值出现在 $q_j(\boldsymbol Z_j)=\tilde p(\boldsymbol X,\boldsymbol Z_j)$ 的位置。于是最优解 $q_j^\star(\boldsymbol Z_j)$ 的⼀般表达式形式为

$$\ln q_j^\star(\boldsymbol Z_j)=\mathbb E_{i\ne j}[\ln p(\boldsymbol X,\boldsymbol Z)]+C\tag7$$

&emsp;&emsp;这是变分法应⽤的基础，这个解表明，为了得到因⼦ $q_j$ 的最优解的对数，我们只需考虑所有潜在变量和可见变量上的联合概率分布的对数，然后关于所有其他的因⼦ $\left \{ q_{i\ne j} \right \}$ 取期望即可。

&emsp;&emsp;公式 (7) 中的常数通过对概率分布 $q_j^\star(\boldsymbol Z_j)$ 进行归一化的方式设定，取两侧指数

$$q_j^\star(\boldsymbol Z_j)=\frac{\exp (\mathbb E_{i \ne j}[\ln p(\boldsymbol X,\boldsymbol Z)])}{\int \exp (\mathbb E_{i \ne j}[\ln p(\boldsymbol X,\boldsymbol Z)])\mathrm d\boldsymbol Z_j}\tag8$$

&emsp;&emsp;关于公式 (7) 也可以使用 EM 算法，⾸先初始化所有因⼦ $q_j(\boldsymbol Z_j)$，然后在各个因⼦上进⾏循环，每⼀轮⽤⼀个修正后的估计来替换当前因⼦。这个修正后的估计由公式 (7) 的右侧给出，计算时使⽤了当前对于所有其他因⼦的估计，算法保证收敛，因为下界关于每个因⼦ $q_j(\boldsymbol Z_j)$ 是⼀个凸函数。

#### 分解近似的性质

&emsp;&emsp;变分推断是基于真实后验概率分布的分解近似，现在考虑⼀下使⽤分解概率分布的⽅式近似⼀个⼀般的概率分布。对于使⽤分解的⾼斯分布近似⼀个⾼斯分布的问题，考虑两个相关的变量 $\boldsymbol z=(z_1,z_2)$ 上的⾼斯分布 $p(\boldsymbol z)=\mathcal N(\boldsymbol z|\boldsymbol \mu,\boldsymbol \Lambda^{-1})$，其中均值和精度的元素为

$$\boldsymbol \mu=\begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, ~~~~~~~~\boldsymbol \Lambda=\begin{pmatrix} \Lambda_{11} & \Lambda_{12}\\ \Lambda_{21} & \Lambda_{22} \end{pmatrix}\tag9$$

&emsp;&emsp;根据精度矩阵的对称性，$\Lambda_{12}=\Lambda_{21}$，现在使用一个分解的高斯分布 $q(\boldsymbol z)=q(z_1)q(z_2)$ 来近似这个分布。⾸先使⽤公式 (7) 来寻找最优因⼦ $q_1^\star(z_1)$ 的表达式。在等式右侧只需要保留那些与 $z_1$ 有函数依赖关系的项即可，所有其他项都可以被整合到归⼀化常数中。因此有

$$\begin{align} \ln q_1^\star(z_1)&=\mathbb E_{z_2}[\ln p(\boldsymbol z)]+C\\ &=\mathbb E_{z_2}\left [-\frac{1}{2}(z_1-\mu_1)^2\Lambda_{11}- (z_1-\mu_1)\Lambda_{12}(z_2-\mu_2)\right ]+C\\ &=-\frac{1}{2}z_1^2\Lambda_{11}+z_1\mu_1\Lambda_{11}-z_1\Lambda_{12}(\mathbb E[z_2]-\mu_2)+C\\ \end{align}\tag{10}$$

&emsp;&emsp;接下来观察到这个表达式的右侧是 $z_1$ 的⼆次函数，因此可以将 $q_1^\star(z_1)$ 看成⼀个⾼斯分布。我们不需要假设 $q(z_i)$ 是⾼斯分布，但通过对所有可能的分布 $q(z_i)$ 上的 KL 散度变分最优化也能推导出这个结果。使⽤配平⽅的⽅法，我们可以得到这个⾼斯分布的均值和⽅差，有

$$q_1^\star(z_1)=\mathcal N(z_1|m_1,\Lambda_{11}^{-1})\tag{11}$$ 
$$m_1=\mu_1-\Lambda_{11}^{-1}\Lambda_{12}(\mathbb E[z_2]-\mu_2)\tag{12}$$

&emsp;&emsp;根据对称性，$q_2^\star(z_2)$ 也是一个高斯分布，可以写成

$$q_2^\star(z_2)=\mathcal N(z_2|m_2,\Lambda_{22}^{-1})\tag{13}$$ 
$$m_2=\mu_2-\Lambda_{22}^{-1}\Lambda_{21}(\mathbb E[z_1]-\mu_1)\tag{14}$$

&emsp;&emsp;这些解是相互偶合的，即 $q_1^\star(z_1)$ 依赖于关于 $q_2^\star(z_2)$ 计算的期望，反之亦然。这些解的求解方式通常将变分解看成重估计⽅程，然后在变量之间循环，更新这些解，直到满⾜某个收敛准则。但是这⾥可以找到⼀个解析解，由于 $\mathbb E[z_1]=m_1$ 且 $\mathbb E[z_2]=m_2$，如果取 $\mathbb E[z_1]=\mu_1$ 且 $\mathbb E[z_2]=\mu_2$，那么这两个⽅程会得到满⾜。只要概率分布⾮奇异，那么这个解是唯⼀解。 这个结果下图所示，两种形式的 KL 散度的对⽐。绿⾊轮廓线对应于两个变量 $z_1$ 和 $z_2$ 上的相关⾼斯分布 $p(z)$ 的 3 个标准差的位置，红⾊轮廓线表示相同变量上的近似分布 $q(z)$ 的同样位置。(a) 图中，参数通过最⼩化 $\mathrm {KL}[q||p]$ 的⽅式获得，(b) 图中参数通过最⼩化相反的 KL 散度 $\mathrm {KL}[p||q]$ 获得。虽然 (a) 图均值被正确地描述了，但是 $q(z)$ 的⽅差由 $p(z)$ 的最⼩⽅差⽅向确定，沿着垂直⽅向的⽅差被强烈低估了，即分解变分近似对后验概率分布的近似倾向于紧凑。而 (b) 图中的 KL 散度被⽤于另⼀种近似推断的框架中，被称为期望传播 (expectation propagation)。

<div align=center>
<img src="images/11_1_var1.png" width="80%"/>
</div>

在期望传播中，我们一般考虑最⼩化 $\mathrm {KL}[p||q]$ 的问题，KL 散度可以写成

$$\mathrm{KL}(p||q)=-\int p(\boldsymbol Z)\left [ \sum_{i=1}^{M}\ln q_i(\boldsymbol Z_i) \right ]\mathrm d\boldsymbol Z+C\tag{15}$$
其中常数项就是 $p(\boldsymbol Z)$ 的熵，因此不依赖于 $q(\boldsymbol Z)$。现在可以关于每个因⼦ $q_j(\boldsymbol Z_j)$ 进⾏最优化。使⽤拉格朗⽇乘数法，可得

$$q_j^{\star}(\boldsymbol Z_j)=\int p(\boldsymbol Z)\prod_{i \ne j}d\boldsymbol Z_i=p(\boldsymbol Z_j)\tag{16}$$

&emsp;&emsp;这种情况下 $q_j(\boldsymbol Z_j)$ 的最优解等于对应的边缘概率分布 $p(\boldsymbol Z)$。

&emsp;&emsp;这两个结果的区别可以⽤下⾯的⽅式理解。$\boldsymbol Z$ 空间中 $p(\boldsymbol Z)$ 接近等于零的区域对于 KL 散度

$$\mathrm {KL}(q||p)=-\int q(\boldsymbol Z)\ln \left \{ \frac{p(\boldsymbol Z)}{q(\boldsymbol Z)} \right \}\mathrm d\boldsymbol Z\tag{17}$$

&emsp;&emsp;有⼀个⼤的正数的贡献，除⾮ $q(\boldsymbol Z)$ 也接近零。因此最⼩化这种形式的 KL 散度会使得 $q(\boldsymbol Z)$ 避开 $p(\boldsymbol Z)$ 很⼩的区域。相反地，使得 $\mathrm {KL}[p||q]$ 取得最⼩值的概率分布 $q(\boldsymbol Z)$ 在 $p(\boldsymbol Z)$ ⾮零的区域也是⾮零的。

#### 一元高斯分布

&emsp;&emsp;现在使⽤⼀元变量 $x$ 上的⾼斯分布来说明分解变分近似，在给定 $x$ 观测值的数据集 $\mathcal D=\left \{ x_1,\dots,x_N \right \}$ 的情况下，我们推断均值 $\mu$ 和精度 $\tau$ 的后验概率分布。假设数据独⽴地从⾼斯分布中采样，那么似然函数为

$$p(\mathcal D|\mu,\tau)=\left ( \frac{\tau}{2\pi} \right )^{\frac{N}{2}}\exp \left \{ -\frac{\tau}{2}\sum_{n=1}^N(x_n-\mu)^2 \right \}\tag{18}$$

&emsp;&emsp;现在引⼊ $\mu$ 和 $\tau$ 的共轭先验分布，形式为

$$p(\mu|\tau)=\mathcal N(\mu|\mu_0,(\lambda_0\tau)^{-1})~~~~~~(19)\\ p(\tau)=\mathrm{Gam}(\tau|a_0,b_0)\tag{20}$$

&emsp;&emsp;对于这个问题，后验概率可以求出精确解，并且形式还是⾼斯-Gamma 分布。考虑对后验概率分布的⼀个分解变分近似，形式为

$$q(\mu,\tau)=q_\mu(\mu)q_\tau(\tau)\tag{21}$$
我们对变分近似的概率分布这样操作，但是真实后验概率分布不可以按照这种形式进⾏分解。最优因⼦ $q_\mu(\mu)$ 和 $q_\tau(\tau)$ 可以从公式 (7) 中得到。对于 $q_\mu(\mu)$，有

\begin{align} \ln q_\mu^\star(\mu)&=\mathbb E_\tau[\ln p(\mathcal D|\mu,\tau)+\ln p(\mu|\tau)]+C\\ &=-\frac{\mathbb E[\tau]}{2}\left \{ \lambda_0(\mu-\mu_0)^2+\sum_{n=1}^N(x_n-\mu)^2 \right \}+C \end{align}~~~~~~(22)\\
对 $\mu$ 配平⽅，可以看到 $q_\mu(\mu)$ 是⼀个⾼斯分布 $\mathcal N(\mu|\mu_N,\lambda_N^{-1})$，其中，均值和⽅差为

$$\mu_N=\frac{\lambda_0\mu_0+N\bar{x}}{\lambda_0+N}\tag{23}$$ 
$$\lambda_N=(\lambda_0+N)\mathbb E[\tau]\tag{24}$$
对于 $N\rightarrow \infty$，就是最大似然的结果，其中 $\mu_N=\bar x$，精度为无穷大。

&emsp;&emsp;因子 $q_\tau(\tau)$ 的最优解为

$$\begin{align} \ln q_\tau^\star(\tau)&=\mathbb E_\mu[\ln p(\mathcal D|\mu,\tau)+\ln p(\mu|\tau)]+\ln p(\tau)+C\\ &=(a_0-1)\ln \tau-b_0\tau+\frac{N+1}{2}\ln \tau-\frac{\tau}{2}\mathbb E_{\mu}\left [ \sum_{n=1}^N(x_n-\mu)^2+\lambda_0(\mu-\mu_0)^2 \right ]+C \end{align}\tag{24}$$

&emsp;&emsp;因此 $q_\tau(\tau)$ 是⼀个 Gamma 分布。

&emsp;&emsp;与之前一样，我们不假设最优概率分布 $q_\mu(\mu)$ 和 $q_\tau(\tau)$ 的具体形式，它们从似然函数和对应共轭先验分布中⾃然地得到，因此得到了最优概率分布 $q_\mu(\mu)$ 和 $q_\tau(\tau)$ 的表达式，每个表达式依赖关于其他概率分布计算得到的矩。因此，⼀种寻找解的⽅法仍旧是 EM 算法的思路，首先对 $\mathbb E[\tau]$ 进⾏⼀个初始的猜测，然后使⽤这个猜测来重新计算概率分布 $q_\mu(\mu)$。给定这个修正的概率分布之后，接下来计算所需的矩 $\mathbb E[\mu]$ 和 $\mathbb E[\mu^2]$，并且使⽤这些矩来重新计算概率分布 $q_\tau(\tau)$，依次类推，直至收敛。
