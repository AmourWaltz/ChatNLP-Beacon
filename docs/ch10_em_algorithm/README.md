# 10 期望最大化算法 EM Algorithm

&emsp;&emsp;之前的篇章大部分讲的都是监督学习模型，本章讨论一些非监督学习的方法。虽然现在监督学习算法占据了主流，但不得不说非监督学习才是更符合自然状态的算法，因为我们获得的原始数据大部分都是没有标签的，非监督学习就是用于应对这种没有人工干预的数据的，因此在吴军老师《数学之美》一书中，就将本文的主角————期望最大化 (Expectation Maximum) 算法，称作上帝的算法，我们不需要手动设置标签，也不需要借助任何先验知识，就可以完成一些分类等任务，

&emsp;&emsp;在高斯分布一节末尾部分，我们引出了混合高斯模型。在此先用一般的高斯分布重新描述一遍问题，机器学习的任务是，根据一组 M 维数据训练模型然后对新数据进行预测，对于参数化模型而言，我们需要学习出模型的参数。假设 $N$ 个数据 $\boldsymbol x_1,\boldsymbol x_2,\dots,\boldsymbol x_N$ 通过一组高斯分布独立采样获得，那么学习任务就是估计出这个高斯分布的均值方差 $\boldsymbol \theta,\boldsymbol \Sigma$，其中 $\boldsymbol \theta$ 也是 $M$ 维，分别对应训练数据每一维度的均值，$\boldsymbol \Sigma$ 是 $M\times M$ 维协方差矩阵，代表数据所有维度两两之间的方差。先假设一组 $\boldsymbol \theta,\boldsymbol \Sigma$，在这组参数下的高斯分布可以采样得到训练数据 $\boldsymbol x_1,\boldsymbol x_2,\dots,\boldsymbol x_N 的概率为 \prod_{n=1}^{N}p(\boldsymbol x_n|\boldsymbol \theta,\boldsymbol \Sigma)$，也可表示为联合概率分布 $p(\boldsymbol x_1,\boldsymbol x_2,\dots,\boldsymbol x_N|\boldsymbol \theta,\boldsymbol \Sigma)$，这种根据未知固定参数进行估计的方法就是似然估计，如果联合概率分布值越大，就说明我们假设的参数越接近真实分布的参数，所以学习思路就是使联合概率分布值尽可能的大，这就是在第一章提到的极大似然估计，利用取负对数，很容易进行求解。

&emsp;&emsp;这时再来考虑混合高斯分布情况，假设 $N$ 个数据 $\boldsymbol x_1,\boldsymbol x_2,\dots,\boldsymbol x_N$ 通过 $K$ 个高斯分布独立采样获得，每个高斯分布的均值方差分别为 $\boldsymbol \theta_k,\boldsymbol \Sigma_k$，每个数据都只从某一个高斯分布采样获得，仍旧假设一组均值和方差，此外还需要再假设数据 $\boldsymbol x_n$ 从高斯分布 $\mathcal N(\boldsymbol \theta_k,\boldsymbol \Sigma_k)$ 中采样的概率为 $p(k)$，也就是说 $\boldsymbol x_n$ 被采样的概率 $\sum_{k=1}^{K}p(k)p(\boldsymbol x_n|\boldsymbol \theta_k,\boldsymbol \Sigma_k)$，所有数据在这组参数下采样获得的概率为 $\prod_{n=1}^{N}\sum_{k=1}^{K}p(k)p(\boldsymbol x_n|\boldsymbol \theta_k,\boldsymbol \Sigma_k)$，如果仍采用极大似然估计取负对数，就会得到

$$\sum_{n=1}^N\ln\left\{ \sum_{k=1}^{K}p(k)\mathcal N(\boldsymbol x_n|\boldsymbol \mu_k,\boldsymbol \Sigma_k) \right \}\tag1$$

&emsp;&emsp;在讨论如何最⼤化公式 (1) 这个函数之前，需要先强调一下由于奇异性的存在造成的⾼斯混合模型的最⼤似然框架中的⼀个问题。考虑⼀个⾼斯混合模型，它的分量的协⽅差矩阵为 $\boldsymbol \Sigma_k=\sigma_k^2\boldsymbol I$，其中 $\boldsymbol I$ 是单位矩阵，该结论对于⼀般的协⽅差矩阵仍成⽴。假设混合模型的第 $j$ 个分量的均值 $\mu_j$ 与某个数据点完全相同，即对于某个数据点 $\boldsymbol x_n$，有 $\boldsymbol \mu_j=\boldsymbol x_n$。这样，这个数据点会为似然函数贡献⼀项，形式为

$$\mathcal N(\boldsymbol x_n|\boldsymbol x_n,\sigma_j^2\boldsymbol I)=\frac{1}{(2\pi)^{\frac{1}{2}}}\frac{1}{\sigma_j^{D}}\tag2$$

&emsp;&emsp;如果考虑极限 $\sigma_j\rightarrow 0$，那么这⼀项趋于⽆穷⼤，因此对数似然函数也会趋于⽆穷⼤，对数似然函数的最⼤化不是⼀个具有良好定义的问题，因为这种奇异性总会发⽣在任何⼀个 “退化” 到⼀个具体的数据点上的⾼斯分量上。这个问题在单⼀⾼斯分布中没有出现，因为即使单⼀⾼斯分布退化到⼀个数据点上，由其他数据点产⽣的似然函数贡献也会有可乘的因⼦，这些因⼦会以指数速度趋于零，从⽽使整体的似然函数趋于零⽽非⽆穷⼤。然⽽，⼀旦在混合概率分布中存在两个或以上分量，其中⼀个分量具有有限的⽅差，因此对所有的数据点都会赋予⼀个有限的概率值，⽽另⼀个分量会收缩到⼀个具体的数据点，因此会给对数似然函数贡献⼀个不断增加的值。这种奇异性解释了最⼤似然⽅法中出现的过拟合现象。

&emsp;&emsp;公式 (1) 的情形较为复杂，因为对数中存在⼀个求和式，我们无法再使用一般的解析法去求解。于是期望最大化 (EM) 算法就被派上用场。

## 10.1 K-均值聚类 K-means Clustering

&emsp;&emsp;在介绍 EM 算法之前，先来引入 K-均值聚类的方法。K-均值聚类算法的思路不难理解：假设一个一维的由 K 个高斯分布组成的混合高斯模型，存在 $K$ 个均值和方差，我们仍旧先假设一组均值和方差 $\left \{ \mu_1,\sigma_1,\mu_2,\sigma_2,\dots,\mu_K,\sigma_K \right \}$，然后对于 $N$ 个训练数据 $\left \{ x_1, x_2,\dots, x_N \right \}$，进行以下步骤：

&emsp;&emsp;找到距离每个数据点最近的那个均值，把数据点分为 $K$ 组，每一组的均值分别为 $\left \{ \mu_1,\mu_2,\dots,\mu_K \right \}$。
将这 K 组数据的数据点重新计算各自的均值，得到一组新的均值 $\left \{ \mu_1^{(1)},\mu_2^{(1)},\dots,\mu_K^{(1)} \right \}$，然后根据这一组新的均值，寻找所有数据点距离最近的均值，然后重新进行分组。
重复进行步骤 2，不断更新均值，直到所有数据点不再发生变动。
每次更新均值都会发生一个小的变动，并且这个过程最终是可以收敛的，同时保证将距离近的点聚集在一起。这个过程需要定义数据间相似程度的距离函数，这个距离的度量应该能使同一类的数据距离较近，不同类数据距离较远。在上述寻找高斯分布均值的例子中，由于高斯分布在均值附近概率密度最大，因此越接近一个高斯分布的均值，就越有可能服从这个高斯分布。我们期望的结果是，距离相近的数据服从同一个高斯分布，这样同一个高斯分布中各个数据点到均值的平均距离 $d$ 较近，而不同高斯分布的均值之间的距离 $D$ 较远，我们希望的迭代过程是每次迭代时，$d$ 减小而 $D$ 增大。

&emsp;&emsp;假设第一到 $K$ 组分布分别有 $n_1,n_2,\dots,n_k$ 个点，每一组所有数据和均值的平均距离为 $d_1,d_2,\dots,d_k$，那么所有数据到均值的平均距离

$$d=\frac{n_1d_1+n_2d_2+\dots+n_kd_k}{k}\tag2$$

&emsp;&emsp;如果第 $i$ 个高斯分布的均值和第 $j$ 个高斯分布的均值之间的距离是 $D_{ij}$，考虑每组的数据个数，均值之间的平均距离

$$D=\sum_{i=1}^K\sum_{j=1}^K\frac{D_{ij}}{K(K-1)}\tag3$$

&emsp;&emsp;这实际上就是一个通过减小类内差异，增加类间差异进行优化的聚类算法，$K$ 均值算法一般将平⽅欧⼏⾥得距离作为数据点与代表向量之间不相似程度的度量，这不仅限制了能够处理的数据变量的类型，⽽且使得聚类中⼼的确定对于异常点不具有鲁棒性。

&emsp;&emsp;这种算法可以更为一般地概括为，有一组数据点，我们首先根据现有模型计算所有数据输入到模型的计算结果，这个过程称为期望 (Expectation, E) 计算过程；接下来重新计算参数，最大化期望值如上述问题的 $D,-d$，这就是最大化 (Maximum, M) 过程，这一类算法都可以称为 EM 算法。当然上述例子只是简单阐述了这种思想，K 均值算法本⾝经常被⽤于在EM算法之前初始化⾼斯混合模型的参数，对于如何使用 EM 算法处理混合高斯模型，还需进行复杂的数值计算过程。

## 10.2 EM 算法 Expectation Maximum Algorithm

&emsp;&emsp;在第九章最后，我们将⾼斯混合模型看成⾼斯分量的简单线性叠加，⽬标是提供⼀类⽐单独⾼斯分布更强⼤的概率模型，下面使用离散潜在变量 (latent variables) 来描述⾼斯混合模型，然后引出 EM 算法的求解。如果我们定义观测变量（例如数据集 $\mathcal D$ ）和潜在变量（例如参数 $\boldsymbol w$ ）的⼀个联合概率分布，那么对应的观测变量本⾝的概率分布可以通过求边缘概率的⽅法得到

$$p(\mathcal D)=\int p(\mathcal D,\boldsymbol w)\mathrm d \boldsymbol w=\int p(\mathcal D|\boldsymbol w)p(\boldsymbol w)\mathrm d \boldsymbol w\tag4$$

&emsp;&emsp;这使得观测变量上的复杂边缘概率分布可以通过观测变量与潜在变量组成的扩展空间上的联合概率分布来表示。因此，潜在变量的引⼊使复杂的概率分布可以由简单的分量组成。这是 EM 算法处理混合高斯的一个关键。将公式 (1) 中的 $p(k)$ 记作 $\pi_k$，⾼斯混合概率分布可以写成⾼斯分布的线性叠加的形式，即

$$p(\boldsymbol x)=\sum_{k=1}^K\pi_k\mathcal N\left ( \boldsymbol x|\boldsymbol \mu_k,\boldsymbol \Sigma_k \right )\tag5$$

&emsp;&emsp;然后引入一个二值随机变量 $\boldsymbol z$，采用 one-hot 编码，其中⼀个特定的元素 $z_k$ 等于 1，其余所有元素等于 0。根据边缘概率分布 $p(\boldsymbol z)$ 和条件概率分布 $p(\boldsymbol x|\boldsymbol z)$ 定义联合概率分布 $p(\boldsymbol x,\boldsymbol z)$， $\boldsymbol z$ 的边缘概率分布根据混合系数 $\pi_k$ 进⾏赋值，即

$$p(z_k=1)=\pi_k\tag6$$

$\pi_k$ 应满足 $p(k)$ 的约束条件，这个概率分布也可以写成

$$p(\boldsymbol z)=\prod_{k=1}^{K}\pi_k^{z_k}\tag7$$

&emsp;&emsp;给定 $\boldsymbol z$ 的⼀个特定值，$\boldsymbol x$ 的条件概率分布是⼀个⾼斯分布

$$p(\boldsymbol x|\boldsymbol z)=\prod_{k=1}^{K}\mathcal N(\boldsymbol x|\boldsymbol \mu_k,\boldsymbol \Sigma_k)^{z_k}\tag8$$

&emsp;&emsp;联合概率分布为 $p(\boldsymbol z)p(\boldsymbol x|\boldsymbol z)$，从⽽ $\boldsymbol x$ 的边缘概率分布可以通过将联合概率分布对所有可能的 $\boldsymbol z$ 求和的方式得到，即

$$p(\boldsymbol x)=\sum_{\boldsymbol z}p(\boldsymbol z)p(\boldsymbol x|\boldsymbol z)=\sum_{k=1}^K\pi_k\mathcal N\left ( \boldsymbol x|\boldsymbol \mu_k,\boldsymbol \Sigma_k \right )\tag9$$

&emsp;&emsp;$\boldsymbol x$ 的边缘概率分布是公式 (5) 的⾼斯混合分布，我们已经⽤ $p(\boldsymbol x)=\sum_{\boldsymbol z}p(\boldsymbol x,\boldsymbol z)$ 的⽅式表示了边缘概率分布，因此对于每个数据点 $\boldsymbol x_n$，存在⼀个对应的潜在变量 $\boldsymbol z_n$。

&emsp;&emsp;利用潜在变量表示⾼斯混合分布可以得到⼀个等价的公式，这样做能够对联合概率分布 $p(\boldsymbol x,\boldsymbol z)$ 操作，会极大地简化计算，通过引⼊ EM 算法，即可看到这⼀点。

&emsp;&emsp;另⼀个起着重要作⽤的量是给定 $\boldsymbol x$ 的条件下，$\boldsymbol z$ 的条件概率。⽤ $\gamma(z_k)$ 表示 $p(z_k=1|\boldsymbol x)$，它的值可以使⽤贝叶斯定理求出

$$\begin{align} \gamma(z_k)\equiv p(z_k=1|\boldsymbol x)&=\frac{p(z_k=1)p(\boldsymbol x|z_k=1)}{\sum_{j=1}^Kp(z_j=1)p(\boldsymbol x|z_j=1)}\\ &=\frac{\pi_k\mathcal N(\boldsymbol x|\boldsymbol \mu_k,\boldsymbol \Sigma_k)}{\sum_{j=1}^K\pi_j\mathcal N(\boldsymbol x|\boldsymbol \mu_j,\boldsymbol \Sigma_j)} \end{align}\tag{10}$$

&emsp;&emsp;我们将 $\pi_k$ 看成 $z_k=1$ 的先验概率，将 $\gamma(z_k)$ 看成观测到 $\boldsymbol x$ 之后对应的后验概率，$\gamma(z_k)$ 也可以被看做分量 $k$ 对于 “解释” 观测值 $\boldsymbol x$ 的 “责任” (responsibility)。

#### ⽤于⾼斯混合模型的 EM

&emsp;&emsp;接下来讨论⼀种寻找带有潜在变量的高斯混合模型的最⼤似然解的 EM 算法，⾸先写下似然函数的最⼤值必须满⾜的条件，令公式 (1) 关于⾼斯分量的均值 $\boldsymbol \mu_k$ 等于零，有

$$0=\sum_{n=1}^N\frac{\pi_k\mathcal N(\boldsymbol x_n|\boldsymbol \mu_k,\boldsymbol \Sigma_k)}{\underbrace{\sum_j\pi_j\mathcal N(\boldsymbol x_n|\boldsymbol \mu_j,\boldsymbol \Sigma_j)}_{\gamma(z_{nk})}} {\boldsymbol \Sigma_{k}^{-1}} (\boldsymbol x_n-\boldsymbol \mu_k)\tag{11}$$

&emsp;&emsp;可以看到公式 (10) 给出的后验概率很⾃然地出现在等式右侧，两侧同时乘以 $\boldsymbol \Sigma_k$，可得

$$\boldsymbol \mu_k=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\boldsymbol x_n~~~~~~(12)\\ N_k=\sum_{n=1}^N\gamma(z_{nk})\tag{13}$$

&emsp;&emsp;将 $N_k$ 看做分配到类别 $k$ 的数据点的有效数量，其中第 $k$ 个⾼斯分量的均值 $\boldsymbol \mu_k$ 通过对数据集⾥所有的数据点求加权平均得到，其中数据点 $\boldsymbol x_k$ 的权因⼦由后验概率 $\gamma(z_{nk})$ 给出，表示分量 $k$ 对⽣成 $\boldsymbol x_n$ 的责任。

&emsp;&emsp;如果令公式 (1) 关于 $\boldsymbol \Sigma_k$ 的导数等于零，然后⽤⼀个类似的推理过程，有

$$\boldsymbol \Sigma_k=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})(\boldsymbol x_n-\boldsymbol \mu_k)(\boldsymbol x_n-\boldsymbol \mu_k)^T\tag{14}$$

&emsp;&emsp;与之前⼀样，每个数据点都有⼀个权值，权值等于对应的后验概率，分母为与对应分量相关联的数据点的有效数量。

&emsp;&emsp;最后关于混合系数 $\pi_k$ 最大化公式 (1)，考虑限制条件 $\sum_{k=1}^K\pi_k=1$，根据拉格朗日乘数法，即最大化

$$\ln p\left ( \boldsymbol X|\boldsymbol \pi,\boldsymbol \mu,\boldsymbol \Sigma \right )+\lambda\left ( \sum_{k=1}^K\pi_k-1 \right )\tag{15}$$

&emsp;&emsp;令其关于 $\pi_k$ 一阶导为 0，可得

$$\sum_{n=1}^N\gamma(z_{nk})+\lambda=0\tag{16}$$
将公式两侧乘以 $\pi_k$，然后使⽤限制条件对 $k$ 求和，可得 $\lambda=-N$。使⽤这个结果消去 $\lambda$ 即

$$\pi_k=\frac{N_k}{N}\tag{17}$$
从⽽第 $k$ 个分量的混合系数为那个分量对于解释数据点的 “责任” 的平均值。可以说，这个结论是符合我们常规认知的，如果一个分量可解释的数据点越多，那么也就意味着对于一个新的数据点，服从这个分量的高斯分布的概率 $p(k)$ 越大。

&emsp;&emsp;公式 (12) (14) (17) 并没有给出混合模型参数的⼀个解析解，因为 $\gamma(z_{nk})$ 通过公式 (10) 以⼀种复杂的⽅式依赖于这些参数。然⽽，这些结果确实给出了⼀个简单的迭代⽅法来寻找问题的最⼤似然解，这个迭代过程就是 EM 算法应⽤于⾼斯混合模型的⼀个实例。⾸先为均值 $\boldsymbol \mu$ 协⽅差 $\boldsymbol \Sigma$、混合系数 $\boldsymbol \pi$ 选择⼀个初始值，然后交替进⾏ E 步骤和 M 步骤的更新，在 E 步骤中，使⽤参数的当前值计算公式 (10) 给出的后验概率，然后将计算出的概率⽤于 M 步骤，使⽤公式 (12) (14) (17) 重新估计均值、⽅差和混合系数。在进⾏这⼀步骤时，⾸先使⽤公式 (12) 计算新的均值，然后使⽤新的均值通过公式 (14) 找到协⽅差，这与单⼀⾼斯分布的对应结果保持⼀致，每次通过 E 步骤和 M 步骤对参数的更新确保了对数似然函数的增⼤。在实际应⽤中，当对数似然函数的变化量或者参数的变化量低于某个阈值时，我们就认为算法收敛。

&emsp;&emsp;与 K-均值算法相⽐，EM 算法在收敛前，经历了更多次的迭代，每次迭代需要更多的计算量。因此，通常先使用 K-均值算法找到⾼斯混合模型的⼀个合适的初始化值，接下来使⽤ EM 算法进⾏调节。协⽅差矩阵可以很⽅便地初始化为通过 K-均值算法找到的聚类的样本协⽅差，混合系数可以被设置为分配到对应类别中的数据点所占的⽐例。与最⼤化对数似然函数基于梯度的⽅法相同，算法必须避免似然函数带来的奇异性，即⾼斯分量退化到⼀个具体的数据点。通常对数似然函数会有多个局部极⼤值，EM 算法一般只找到一个局部极大值。

#### 潜在变量的 EM 算法

&emsp;&emsp;EM 算法的⽬标是找到具有潜在变量的模型的最⼤似然解，将所有观测数据的集合记作 $\boldsymbol X$，其中第 $n$ ⾏表示 $\boldsymbol x_n^T$。类似地，将所有潜在变量的集合记作 $\boldsymbol Z$，对应第 $n$ 行为 $\boldsymbol z_n^T$。所有模型参数的集合被记作 $\boldsymbol\theta$，类似于混合高斯模型的均值，协方差和混合系数，因此对数似然函数为

$$\ln p(\boldsymbol X|\boldsymbol \theta)=\ln \left \{ \sum_{\boldsymbol Z}p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta) \right \}\tag{18}$$

&emsp;&emsp;由于对潜在变量的求和位于对数的内部，即使联合概率分布 $p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)$ 属于指数族分布，边缘概率分布 $p(\boldsymbol X|\boldsymbol \theta)$ 通常也不是指数族分布。求和式的出现阻⽌了对数运算直接作⽤于联合概率分布，使得最⼤似然解的形式更加复杂。

&emsp;&emsp;假定对于 $\boldsymbol X$ 中的每个观测，都有潜在变量 $\boldsymbol Z$ 的对应值，将 $\left \{ \boldsymbol X,\boldsymbol Z \right \}$ 称为完整数据集，而实际的数据集 $\boldsymbol X$ 是不完整的。完整数据集的对数似然函数的形式为 $\ln p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)$，并且假定对这个完整数据的对数似然函数进⾏最⼤化是很容易的。

&emsp;&emsp;然⽽实际应⽤中，没有完整数据集 $\left \{ \boldsymbol X,\boldsymbol Z \right \}$，只有不完整的数据 $\boldsymbol X$。关于潜在变量 $\boldsymbol Z$ 的取值仅仅来源于后验概率分布 $p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta)$。由于不能使⽤完整数据的对数似然函数，因此我们反过来考虑在潜在变量的后验概率分布下参数的期望值，这对应于 EM 算法中的 E 步骤，在接下来的 M 步骤中，我们最⼤化这个期望。如果当前对于参数的估计为 $\boldsymbol \theta^{old}$，那么⼀次连续的 EM 步骤会产⽣⼀个修正的估计 $\boldsymbol \theta^{new}$。

&emsp;&emsp;在 E 步骤，我们使⽤当前的参数值 $\boldsymbol \theta^{old}$ 寻找潜在变量的后验概率分布 $p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta^{old})$。然后使⽤这个后验概率分布计算完整数据对数似然函数对于⼀般的参数值的期望，这个期望被记作 $\mathcal Q(\boldsymbol \theta,\boldsymbol \theta^{old})$，由下式给出

$$\mathcal Q(\boldsymbol \theta,\boldsymbol \theta^{old})=\sum_{\boldsymbol Z}p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta^{old})\ln p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)\tag{19}$$

&emsp;&emsp;在 M 步骤中，最大化下式

$$\boldsymbol \theta^{new}=\arg\max_{\boldsymbol \theta} \mathcal Q(\boldsymbol \theta,\boldsymbol \theta^{old})\tag{20}$$
来确定修正后的参数估计 $\boldsymbol \theta^{new}$。对数操作直接作⽤于联合概率分布 $p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta)$，因此对 M 步骤的最⼤化是可以计算的。

&emsp;&emsp;可以看出引入潜在变量后计算形式上得到了极大的简化，这是一种较为宏观的思路，通过潜在变量和后验概率共同作用避免了直接做对数函数上的求和操作，但整体思路与前一种方法无太大差异，仍然是初始化一组参数，这相当于对所有数据做了一个假设分布，因为我们已经假设了完整数据集 $\ln p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)$ 可以直接进行最大似然估计，其中数据 $\boldsymbol X$ 已知，在假设参数后可以得出潜在变量的后验概率分布 $p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta^{old})$，然后我们只需要最大化在条件 $p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta^{old}) 下 \ln p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)$ 的期望值即可，即

$$\arg \max_{\boldsymbol \theta}\mathbb E_{p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta^{old})}\left [ \ln p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta) \right ]\tag{21}$$
离散形式下正好对应公式 (19)。

#### 一般化的 EM 算法

&emsp;&emsp;根据上述推导可以看出 EM 算法是寻找具有潜在变量的概率模型的最⼤似然解的⼀种通⽤⽅法。考虑⼀个概率模型，其中所有的观测数据联合起来记作 $\boldsymbol X$，将所有潜在变量记作 $\boldsymbol Z$。联合概率分布 $p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)$ 由⼀组参数控制，记作 $\boldsymbol \theta$。优化⽬标是最⼤化似然函数

$$p(\boldsymbol X|\boldsymbol \theta)=\sum_{\boldsymbol Z}p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)tag{22}$$

&emsp;&emsp;如果 $\boldsymbol Z$ 是连续潜在变量，只需要把求和换成积分即可。

&emsp;&emsp;根据上一节的假设，直接最优化不完整数据 $p(\boldsymbol X|\boldsymbol \theta)$ 比较困难，但是最优化完整数据似然函数 $p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta)$ 就容易得多。引⼊⼀个定义在潜在变量上的分布 $q(\boldsymbol Z)$，可以观察到，对于任意的 $q(\boldsymbol Z)$，下⾯的分解成⽴

$$\ln p(\boldsymbol X|\boldsymbol \theta)=\mathcal L(q,\boldsymbol\theta)+\mathrm {KL}(q(\boldsymbol Z)||p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta))\tag{23}$$ 
$$\mathcal L(q,\boldsymbol\theta)=\sum_{\boldsymbol Z}q(\boldsymbol Z)\ln\left \{ \frac{p(\boldsymbol Z,\boldsymbol X|\boldsymbol \theta)}{q(\boldsymbol Z)} \right \}\tag{24}$$
$$\mathrm {KL}(q(\boldsymbol Z)||p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta))=-\sum_{\boldsymbol Z}q(\boldsymbol Z)\ln\left \{ \frac{p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta)}{q(\boldsymbol Z)} \right \}\tag{25}$$

&emsp;&emsp;$\mathcal L(q,\boldsymbol\theta)$ 是概率分布 $q(\boldsymbol Z)$ 的一个泛函数。这个表达式与第十章贝叶斯神经网络用条件边缘概率分布做变分法的做法是一致的，$\mathcal L(q,\boldsymbol\theta)$ 包含了 $\boldsymbol X$ 和 $\boldsymbol Z$ 的联合概率分布，KL 散度项包含了 $\boldsymbol Z$ 的条件概率分布，这一步分解可以通过互信息的一个公式证明，或者更一般地，利用概率的乘积规则

$$\ln p(\boldsymbol Z,\boldsymbol X|\boldsymbol \theta)=\ln p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta)+\ln p(\boldsymbol X,\boldsymbol \theta)\tag{26}$$

&emsp;&emsp;由于 KL 散度恒为非负，所以 $\mathcal L(q,\boldsymbol\theta)$ 是 $\ln p(\boldsymbol X,\boldsymbol \theta)$ 的一个下界。如果使用公式 (23) 定义 EM 算法，可以证明它确实最大化了似然函数。假设参数向量的当前值为 $\boldsymbol \theta^{old}$，在 E 步骤，下界 $\mathcal L(q,\boldsymbol\theta^{old})$ 关于 $q(\boldsymbol Z)$ 被最⼤化，⽽ $\boldsymbol \theta^{old}$ 保持固定， $\mathcal L(q,\boldsymbol\theta^{old})$ 的最⼤值出现在 KL 散度等于零即 ${p(\boldsymbol Z|\boldsymbol X,\boldsymbol \theta)}\equiv{q(\boldsymbol Z)}$ 的时候。

&emsp;&emsp;在 M 步骤中，分布 $q(\boldsymbol Z)$ 保持固定，下界 $\mathcal L(q,\boldsymbol\theta)$ 关于 $\boldsymbol \theta$ 进⾏最⼤化，得到某个新值 $\boldsymbol \theta^{new}$。这会使得下界 $\mathcal L(q,\boldsymbol\theta)$ 增大，同时对应的对数似然函数也增⼤。由于概率分布 $q(\boldsymbol Z)$ 由旧的参数值确定，并且在 M 步骤中保持固定，因此它不会等于新的后验概率分布 $p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta^{new})$，从⽽ KL 散度⾮零。于是，对数似然函数的增加量⼤于下界的增加量。如果将 $p(\boldsymbol X,\boldsymbol Z|\boldsymbol \theta^{old})$ 代⼊公式 (24)，在 E 步骤之后，下界的形式为

$$\begin{align} \mathcal L(q,\boldsymbol\theta)&=\sum_{\boldsymbol Z}p(\boldsymbol Z,\boldsymbol X|\boldsymbol \theta^{old})\ln p(\boldsymbol Z,\boldsymbol X|\boldsymbol \theta)-\sum_{\boldsymbol Z}p(\boldsymbol Z,\boldsymbol X|\boldsymbol \theta^{old})\ln p(\boldsymbol Z,\boldsymbol X|\boldsymbol \theta^{old})\\ &=\mathcal Q(\boldsymbol \theta,\boldsymbol \theta^{old})+C \end{align}\tag{27}$$
其中 $C$ 为常数，是分布 $q$ 的熵，与 $\boldsymbol \theta$ 无关，所以在 M 步骤中，最⼤化的量是完整数据对数似然函数的期望，因此也证明了 EM 算法的 E 步骤和 M 步骤都增⼤了对数似然函数的⼀个良好定义的下界的值，并且完整的 EM 循环会使得模型的参数向着使对数似然函数增⼤的⽅向进⾏改变。
