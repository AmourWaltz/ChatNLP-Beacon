# 核方法 Kernel Method

&emsp;&emsp;本文的核方法与第一章的线性基函数模型联系甚紧。我们在前文中已经提及，对于一些回归问题中无法使用线性公式拟合，或者分类问题中线性不可分的输入数据，可以使用一个非线性变换的基函数 $\phi(\cdot)$，将原始数据映射到高维，比如多项式拟合，就是将原始输入 $x$ 映射到一个高维空间 $[x^2,x^3,\dots,x^n]$，这样几乎可以拟合任意的曲线，或者使任何数据都可分。映射到高维空间，可以看做是一种特征提取，这时我们的问题就转化成如何选取合适的基函数。理论上，任何形式的有限维度的数据都可以通过非线性变换映射到高维空间从而线性可分，但是选取这样的非线性变换需要的代价很大，这时核方法就可以巧妙地解决这个问题。

&emsp;&emsp;为了避免显式的在线性模型的预测函数中出现基函数 $\phi(\cdot)$，我们需要引入一个核函数，核函数的形式很简单

$$k(\boldsymbol x,\boldsymbol x')=\phi(\boldsymbol x)^T\phi(\boldsymbol x')\tag1$$
可以看做是对两个输入向量 $\boldsymbol x,\boldsymbol x'$ 分别做基函数的非线性映射，然后对映射后的高维向量做内积转换到一维空间。由于核函数的输出是个标量值，很容易进行计算操作。

## 3.1 核函数 Kernel Method

#### 对偶表示 Dual Representation

&emsp;&emsp;我们所引入的核函数都是具有固定形式的，最简单的是选取基函数 $\phi(\boldsymbol x)=\boldsymbol x$ 时，得到核函数 $k(\boldsymbol x,\boldsymbol x')=\boldsymbol x^T\boldsymbol x'$，这被称为线性核，后面还会介绍其他更常用的核函数。这样引入核函数是因为，如果从正向思维推导，在第一步选择基函数时我们可选的类型就有很多，仅仅是幂级数的选择就很难确定也很难做到精确，并且如果基函数映射后的空间维度较高，正向计算的运算量也是巨大的；相反，核函数的形式确定相对比较容易，我们会在下文展开讨论，其次，可以避免基函数映射的复杂计算，这相当于一个逆向过程，我们先确定核函数的形式，再倒推出映射关系。这时候如果有一种方法能使核函数替换掉线性模型中的基函数 $\phi(\cdot)$，就可以有效解决这些问题，于是我们引出对偶表示 (dual representation)。

&emsp;&emsp;许多线性参数模型可以被转化为⼀个等价的对偶表示，对偶表示中，原模型的预测函数就被转化为训练数据点处计算的核函数的线性组合。使⽤对偶表示形式，核函数可以⾃然生成。考虑⼀个线性基函数模型 $y(\boldsymbol x)=\boldsymbol w^T\phi(\boldsymbol x)$，其参数通过最⼩化正则化的平⽅和误差函数来确定。正则化的平⽅和误差函数为

$$E(\boldsymbol w)=\frac{1}{2}\sum_{n=1}^N \left \{ \boldsymbol w^T\phi(\boldsymbol x_n)-t_n \right \}^2+\frac{\lambda}{2}\boldsymbol w^T\boldsymbol w\tag2$$
其中 $\lambda\geq0$。令 $E(\boldsymbol w)$ 关于 $\boldsymbol w$ 的梯度等于零，可得 $\boldsymbol w$ 的解是向量 $\phi(\boldsymbol x_n)$ 的线性组合，其形式为

$$\boldsymbol w=-\frac{1}{\lambda}\sum_{n=1}^N \left \{ \boldsymbol w^T\phi(\boldsymbol x_n)-t_n \right \}\phi(\boldsymbol x_n)=\sum_{n=1}^N a_n \phi(\boldsymbol x_n)=\boldsymbol \Phi^T\boldsymbol a\tag3$$
其中 $\boldsymbol \Phi$ 是设计矩阵，第 $n$ ⾏为 $\phi(\boldsymbol x_n)^T$，即代表一个训练数据，向量 $\boldsymbol a=(a_1,\dots,a_N)^T$，其中

$$a_n=-\frac{1}{\lambda}\left \{ \boldsymbol w^T\phi(\boldsymbol x_n)-t_n \right \}\tag4$$
然后将 $\boldsymbol w=\boldsymbol \Phi^T\boldsymbol a$ 代入最小平方公式，可得

$$E(\boldsymbol a)=\frac{1}{2}\boldsymbol a^T\boldsymbol \Phi\boldsymbol \Phi^T\boldsymbol \Phi\boldsymbol \Phi^T\boldsymbol a-\boldsymbol a^T\boldsymbol \Phi\boldsymbol \Phi^T\boldsymbol t+\frac{1}{2}\boldsymbol t^T\boldsymbol t+\frac{\lambda}{2}\boldsymbol a^T\boldsymbol \Phi\boldsymbol \Phi^T\boldsymbol a\tag5$$
其中 $\boldsymbol t=(t_1,\dots,t_N)^T$。定义 Gram 矩阵 $\boldsymbol K=\boldsymbol \Phi\boldsymbol \Phi^T$，它是一个 $N\times N$ 的对称矩阵，其中 $K_{nm}=\phi(\boldsymbol x_n)^T\phi(\boldsymbol x_m)=k(\boldsymbol x_n,\boldsymbol x_m)$，这样我们就引入了公式 (1) 定义的核函数，现在回到最开始引入核函数的问题，我们的目的就是用核函数替换基函数变换，所以此处就将 Gram 矩阵代入平方差公式 (2)，可得

$$E(\boldsymbol a)=\frac{1}{2}\boldsymbol a^T\boldsymbol K\boldsymbol K\boldsymbol a-\boldsymbol a^T\boldsymbol K\boldsymbol t+\frac{1}{2}\boldsymbol t^T\boldsymbol t+\frac{\lambda}{2}\boldsymbol a^T\boldsymbol K\boldsymbol a\tag6$$

&emsp;&emsp;将公式 (3) 代入公式 (4) 可得

$$\boldsymbol a=(\boldsymbol K+\lambda\boldsymbol I_N)^{-1}\boldsymbol t\tag7$$

&emsp;&emsp;然后将公式 (3) (7) 代入线性基函数回归模型，对于新的输入 $\boldsymbol x$，可得

$$y(\boldsymbol x)=\boldsymbol w^T\phi(\boldsymbol x)=\boldsymbol a^T\boldsymbol \Phi\phi(\boldsymbol x)=\boldsymbol k(\boldsymbol x)^T(\boldsymbol K+\lambda\boldsymbol I_N)^{-1}\boldsymbol t\tag8$$
其中 $\boldsymbol k_n(\boldsymbol x)=k(\boldsymbol x_n,\boldsymbol x)$，它表示新输入的预测向量 $\boldsymbol x$ 与训练集中每一个数据做核函数内积，对偶公式使得最⼩平⽅解完全通过核函数表式，这被称为对偶公式，可知对 $\boldsymbol x$ 的预测由训练集数据的⽬标值的线性组合给出。向量 $\boldsymbol a$ 可以被表示为基函数 $\phi(\boldsymbol x)$ 的线性组合，从⽽可使⽤参数 $\boldsymbol w$ 恢复出原始公式。

&emsp;&emsp;在对偶公式中，我们通过对⼀个 $N\times N$ 的矩阵求逆来确定 $\boldsymbol a$，⽽在原始参数空间公式中， 我们要对⼀个 $M\times M$ 的矩阵求逆来确定 $\boldsymbol w$，其中 $M$ 为基函数变换后的空间维度，由于训练数据数量 $N$ 通常远⼤于 $M$，对偶公式似乎没有实际⽤处。然⽽，对偶公式可以完全通过核函数 $k(\boldsymbol x_n,\boldsymbol x)$ 来表示，于是就可以直接对核函数进⾏计算，避免了显式地引⼊基函数特征向量 $\phi(\boldsymbol x)$，从而隐式地使⽤⾼维特征空间。

&emsp;&emsp;如果⼀个算法的输⼊向量 $\boldsymbol x$ 只以标量积的形式出现，那么我们可以⽤⼀些其他的核来替换这个标量积，这就涉及到所谓的核技巧 (kernel trick) 或核替换 (kernel substitution)，我们需要构造合法的核函数，所谓合法，即构造的核函数对应于某个特征空间的标量积。假设有一个核函数

$$k(\boldsymbol x,\boldsymbol z)=\left( \boldsymbol x^T\boldsymbol z \right )^2\tag9$$

&emsp;&emsp;取⼆维输⼊空间 $\boldsymbol x=(x_1,x_2)$ 的情况，可以展开为基函数非线性特征映射

$$\begin{align} k(\boldsymbol x,\boldsymbol z)&=\left( \boldsymbol x^T\boldsymbol z \right )^2=\left(x_1z_1+x_2z_2 \right)^2\\ &=x_1^2z_1^2+2x_1z_1x_2z_2+x_2^2z_2^2\\ &=\left ( x_1^2,\sqrt 2x_1x_2,x_2^2 \right )\left ( z_1^2,\sqrt 2z_1z_2,z_2^2 \right )^T\\ &=\phi\left ( \boldsymbol x \right )^T\phi\left ( \boldsymbol z \right ) \end{align}\tag{10}$$

&emsp;&emsp;⼀般情况下，我们不会去直接构造基函数 $\phi(\cdot)$，而是需要找到⼀种⽅法去检验⼀个函数是否是合法的核函数。核函数 $k(\boldsymbol x,\boldsymbol x')$ 是合法核函数的充要条件是其 Gram 矩阵 $K_{nm}=k(\boldsymbol x_n,\boldsymbol x_m)$ 对所有训练数据 $\left \{ \boldsymbol x_n \right \}$ 都是半正定的，即对于实对称 Gram 矩阵 $\boldsymbol K$，若任意向量 $\boldsymbol x$，都有 $\boldsymbol{x}^T\boldsymbol{K}\boldsymbol{x}\geq0$ 恒成立，则 $\boldsymbol{K}$ 是一个半正定矩阵。

&emsp;&emsp;常见的核函数有多项式核函数

$$k(\boldsymbol x,\boldsymbol x')=(\boldsymbol x^T\boldsymbol x')^M\tag{11}$$

&emsp;&emsp;高斯核函数

$$k(\boldsymbol x,\boldsymbol x')=\exp \left ( -\frac{\left \| \boldsymbol x-\boldsymbol x' \right \|^2}{2\sigma^2} \right )\tag{12}$$

&emsp;&emsp;sigmoid 核函数

$$k(\boldsymbol x,\boldsymbol x')=\tanh \left ( a\boldsymbol x^T\boldsymbol x' + b \right )\tag{13}$$

&emsp;&emsp;借助一些构造核函数的性质，可以通过公式 (9) 的核函数完成对大部分核函数的构造。在基函数有⽆穷多的极限情况下，⼀个具有恰当先验的贝叶斯神经⽹络将会变为⾼斯过程 (Gaussian Process)，因此这就提供了神经⽹络与核⽅法之间的⼀个更深层的联系。

## 3.2 高斯过程 Gaussian Process

&emsp;&emsp;高斯过程与核方法和贝叶斯神经网络有紧密的联系，首先引入高斯过程的概念。考虑一维高斯分布

$$\mathcal{N} \left ( x| \mu , \sigma^{2} \right )=\frac{1}{\sigma \sqrt{2\pi } } \exp \left ( - \frac{\left (x-\mu \right ) ^{2} }{2\sigma ^{2} } \right )\tag{14}$$

&emsp;&emsp;它表示一个变量 $x$ 服从均值和方差分别为 $\mu,\sigma$ 的高斯分布。关于多维高斯分布

$$\mathcal{N} \left ( \boldsymbol { x}| \boldsymbol{u} ,\boldsymbol { \Sigma } \right ) = \frac{1}{ (2\pi)^{D/2} } \frac{1}{ \left | \boldsymbol { \Sigma } \right | ^{D/2} } \exp \left ( - \frac{1}{2} ( \boldsymbol { x}- \boldsymbol { \mu})^{T}\boldsymbol { \Sigma }^{-1}(\boldsymbol { x}- \boldsymbol { \mu})\right )\tag{15}$$
表示 $D$ 维变量 $\boldsymbol x$ 的高斯分布，其中均值和协方差分别为 $\boldsymbol { \mu}, \boldsymbol { \Sigma}$。这时我们再考虑一个无限维的高斯分布，它的每一个维度都服从某种高斯分布，如果我们想表示这种分布，用公式 (15) 中向量的形式显然不行，我们可惜将这个分布的无限维想象成一组连续变量，或者说一个函数 $f(\cdot)$，函数每一点都服从某个高斯分布，那我们就称这种分布为高斯过程 (Gaussian process) 。假设函数变量为 $x$，这个 $x$ 也就是无限维高斯分布的维度，其中一个维度 $x_n$ 服从 $\mathcal N(\mu_n,\sigma_n)$，高斯过程表示为 $f(x_n)\sim \mathcal N(\mu_n,\sigma_n)$。我们把所有的均值 $\mu$ 也表示成连续函数的形式，即 $x_n$ 代表维度的均值为 $m(x_n)$，那么高斯过程的均值就为 $m(x)$。对于方差，参考一维向多维的扩展，多维高斯分布的协方差矩阵就是所有维度两两之间的方差所组成的矩阵，关于高斯过程，它的协方差矩阵也需要考虑所有维度两两之间的方差，假设我们使用核函数 $k(x_i,x_j)$ 表示 $x_i$ 与 $x_j$ 维度的方差，那么这个高斯过程的协方差矩阵就可以用核函数组成的矩阵 $\boldsymbol K(x,x)$ 表示。为了便于介绍我们将无限维空间转换成一个连续空间时使用了单变量 $x$，现在考虑这个无限维空间每一维又包含 $N$ 个子空间，我们就用 $N$ 维变量 $\boldsymbol x$ 作为这个高斯过程的输入变量，这个高斯过程最终表示为

$$f(\boldsymbol x)\sim\mathcal {GP}(m(\boldsymbol x), \boldsymbol K(\boldsymbol x,\boldsymbol x))\tag{16}$$

&emsp;&emsp;现在用高斯过程考虑线性回归问题，假设向量 $\boldsymbol \phi(\boldsymbol x)$ 的元素为 $M$ 个固定基函数的线性组合，线性回归模型为 $y(\boldsymbol x,\boldsymbol w)=\boldsymbol w^T\boldsymbol \phi(\boldsymbol x)$，考虑 $\boldsymbol w$ 上的先验概率分布 $p(\boldsymbol w)=\mathcal N(\boldsymbol w|\boldsymbol 0,\alpha^{-1}\boldsymbol I)$，对于任意给定的 $\boldsymbol w$，线性回归模型就定义了 $\boldsymbol x$ 的⼀个特定的函数，那么定义的 $\boldsymbol w$ 上的概率分布就产生了 $y(\boldsymbol x)$ 上的一个概率分布。实际应⽤中，我们希望计算这个函数在某个 $\boldsymbol x$ 如训练数据点 $\boldsymbol x_1,\dots,\boldsymbol x_N$ 处的函数值，也就是 $y(\boldsymbol x_1),\dots,y(\boldsymbol x_N)$ 处的概率分布，把函数值的集合记作 $\boldsymbol y$，其中 $y_n=y(\boldsymbol x_n)$，结合线性回归模型，可表示为

$$\boldsymbol y=\boldsymbol \Phi\boldsymbol w\tag{17}$$
其中 $\boldsymbol \Phi$ 是设计矩阵，其元素为 $\Phi _{nk}=\phi_k(\boldsymbol x_n)$。根据上述定义，可知函数 $\boldsymbol y$ 就是一个高斯过程，我们的目标就是找出它的概率分布。由于 $\boldsymbol y$ 是参数 $\boldsymbol w$ 的元素给出的服从⾼斯分布的变量的线性组合，因此它本⾝也服从⾼斯分布，于是只需找到其均值和⽅差。根据 $\boldsymbol w$ 先验分布定义，$\boldsymbol y$ 均值和⽅差为

$$\mathbb E\left [ \boldsymbol y \right]=\boldsymbol \Phi\mathbb E\left [ \boldsymbol w \right]=\boldsymbol 0\tag{18}$$ 
$$\mathrm{cov}\left [ \boldsymbol y \right]=\mathbb E\left [ \boldsymbol y \boldsymbol y^T \right]=\boldsymbol \Phi\mathbb E\left [ \boldsymbol w \boldsymbol w^T \right]\boldsymbol \Phi^T=\frac{1}{\alpha}\boldsymbol \Phi\boldsymbol \Phi^T=\boldsymbol K\tag{19}$$
其中 $\boldsymbol K$ 为上一节中定义的 Gram 矩阵，元素为 $K_{nm}=k(\boldsymbol x_n,\boldsymbol x_m)=\frac{1}{\alpha}\phi(\boldsymbol x_n)^T\phi(\boldsymbol x_m)$。这就是高斯过程在线性回归模型上的表示，通常来说，⾼斯过程被定义为函数 $y(\boldsymbol x)$ 上的⼀个概率分布，使得在任意点集 $\boldsymbol x_1,\dots,\boldsymbol x_N$ 处计算的 $y(\boldsymbol x)$ 值的集合联合起来也服从⾼斯分布。在输⼊向量 $\boldsymbol x$ 是⼆维时，这也可以被称为⾼斯随机场 (Gaussian random field)。更⼀般地，可以⽤⼀种合理的⽅式为 $y(\boldsymbol x_1),\dots,y(\boldsymbol x_N)$ 赋予⼀个联合概率分布，来确定⼀个随机过程 (stochastic process) $y(\boldsymbol x)$。

&emsp;&emsp;⾼斯随机过程的联合概率分布通过均值和协⽅差唯一确定，实际应⽤中，关于 $y(\boldsymbol x)$ 的均值没有任何先验，因此根据对称性令其等于零。这等价于基函数中，令权值 $p(\boldsymbol w)$ 的先验均值为 0。之后，⾼斯过程通过给定两个变量 $\boldsymbol x_n,\boldsymbol x_m$ 处函数值 $y(\boldsymbol x_n),y(\boldsymbol x_m)$ 的协⽅差确定，这个协⽅差由核函数计算

$$\mathbb E\left [ y(\boldsymbol x_n)y(\boldsymbol x_m) \right ]=k(\boldsymbol x_n,\boldsymbol x_m)\tag{20}$$

#### 高斯过程线性回归

&emsp;&emsp;前面我们通过一个回归问题引出高斯过程，现在将高斯过程应用到回归模型，确定高斯随机过程分布并用于预测模型，考虑观测目标值的噪声，其中 $t_n=y_n+\epsilon_n，y_n=y(\boldsymbol x_n)$，且 $\epsilon_n$ 是一个高斯随机噪声变量，且对于不同的训练数据点 $\boldsymbol x_n$ 随机噪声都是独立的，考虑服从高斯分布的噪声过程，即

$$p(t_n|y_n)=\mathcal N(t_n|y_n, \beta^{-1})\tag{21}$$

&emsp;&emsp;由于每个数据的观测噪声相互独立，因此 $\boldsymbol y=(y_1,\dots,y_N)^T$ 为条件，$\boldsymbol t=(t_1,\dots,t_N)^T$ 的高斯分布是各向同性的，其联合概率分布为

$$p(\boldsymbol t|\boldsymbol y)=\mathcal N(\boldsymbol t|\boldsymbol y, \beta^{-1}\boldsymbol I_N)\tag{22}$$

&emsp;&emsp;根据高斯过程定义，边缘概率分布 $p(\boldsymbol y)$ 是一个均值为 0，协方差为 Gram 矩阵 $\boldsymbol K$ 的高斯分布，即 $p(\boldsymbol y)=\mathcal N(\boldsymbol y|\boldsymbol 0,\boldsymbol K)$。为了确定核函数 $\boldsymbol K$，我们需要明确，高斯过程是一种非参模型，不同于线性回归或分类模型中通过训练数据学习参数 $\boldsymbol w$ 后再进行预测，从核方法的定义中就可以看出，这里的协方差需要计算输入数据两两之间的相关性才能确定协方差矩阵，而对于新输入数据点的预测，也是需要与训练数据逐一进行相关性计算后再做出预测，这有点类似于 K 近邻算法。在核方法中，我们确定核函数 $\boldsymbol K$ 的方法是，对于相似的点 $\boldsymbol x_n$ 和 $\boldsymbol x_m$，对应的值 $y(\boldsymbol x_n)$ 和 $y(\boldsymbol x_m)$ 的相关性要⼤于不相似的点，这里的相似性通过构造核函数定义。

&emsp;&emsp;为了找到 $p(\boldsymbol t)$，我们需要对 $\boldsymbol y$ 积分，根据第九章公式 (22) (23) 条件概率分布的性质，可得

$$p(\boldsymbol t)=\int p(\boldsymbol t|\boldsymbol y)p(\boldsymbol y)\mathrm d \boldsymbol y=\mathcal N(\boldsymbol t|\boldsymbol 0, \boldsymbol C=\beta^{-1}\boldsymbol I_N+\boldsymbol K)\tag{23}$$
其中 $\boldsymbol C(\boldsymbol x_n,\boldsymbol x_m)=k(\boldsymbol x_n,\boldsymbol x_m)+\beta^{-1}$，由于 $y(\boldsymbol x)$ 与 $\epsilon$ 相关的⾼斯分布是独⽴的，它们的协⽅差可以简单地相加。

&emsp;&emsp;对于⾼斯过程回归，⼀个⼴泛使⽤的核函数为指数项的⼆次型加上常数和线性项，即

$$k(\boldsymbol x_n,\boldsymbol x_m)=\theta_0\exp\left \{ -\frac{\theta_1}{2}\left \| \boldsymbol x_n-\boldsymbol x_m \right \|^2 \right \}+\theta_2+\theta_3\boldsymbol x_n^T\boldsymbol x_m\tag{24}$$

&emsp;&emsp;接下来考虑在给定⼀组训练数据的情况下，对新的输⼊变量的预测。假设训练集 $\mathcal D$ 包含输入变量 $\left \{ \boldsymbol x_1,\dots,\boldsymbol x_N \right \}$ 以及对应的目标值集合 $\boldsymbol t=\left \{ t_1,\dots,t_N \right \}$，我们对新的输⼊变量 $\boldsymbol x_{N+1}$ 预测⽬标值 $t_{N+1}$。根据公式 (23) 可以记作 $p(t_{N+1}|\boldsymbol t)$，联合概率分布形式为 $p(\boldsymbol t_{N+1})$，记作

$$p(\boldsymbol t_{N+1})\sim\mathcal N(\boldsymbol t_{N+1}|\boldsymbol 0, \boldsymbol C_{N+1})\tag{25}$$
其中 $\boldsymbol C_{N+1} 是一个 (N+1)\times (N+1)$ 的协方差矩阵，形式为

$$\boldsymbol C_{N+1}=\begin{pmatrix} \boldsymbol C_N & \boldsymbol k \\ \boldsymbol k^T & c\\ \end{pmatrix}\tag{26}$$

&emsp;&emsp;这表示变量之间的相关性，其中 $\boldsymbol k$ 的元素为 $k_n(\boldsymbol x_n,\boldsymbol x_{N+1})$，$c=k(\boldsymbol x_{N+1},\boldsymbol x_{N+1})+\beta^{-1}$，根据第九章 1.4 节条件概率分布，我们将 $t_{N+1},\boldsymbol t$ 分别代入 $\boldsymbol x_a,\boldsymbol x_b$，可得均值和方差为

$$m(t_{N+1}|\boldsymbol t)=\boldsymbol k^T\boldsymbol C^{-1}_N\boldsymbol t\tag{27}$$ 
$$\sigma^2(t_{N+1}|\boldsymbol t)=c-\boldsymbol k^T\boldsymbol C^{-1}_N\boldsymbol k\tag{28}$$

&emsp;&emsp;由于 $\boldsymbol k$ 是测试输⼊向量 $\boldsymbol x_{N+1}$ 的函数，预测分布也是⼀个⾼斯分布，其均值和⽅差都依赖于 $\boldsymbol x_{N+1}$。预测分布均值可以写成 $\boldsymbol x_{N+1}$ 的形式，为

$$m(t_{N+1}|\boldsymbol t)=\sum_{n=1}^Na_nk(\boldsymbol x_n,\boldsymbol x_{N+1})\tag{29}$$
其中 $a_n$ 是 $\boldsymbol C^{-1}_N\boldsymbol t$ 的第 $n$ 个元素。

&emsp;&emsp;使⽤⾼斯过程的核⼼计算涉及到对 $N\times N$ 的矩阵求逆。标准的矩阵求逆法需要 $O(N^3)$ 次计 算，而在基函数模型中，对⼀个 $M\times M$ 的矩阵 $\boldsymbol S_N$ 求逆，需要 $O(M^3)$ 次计算；给定训练数据后，矩阵求逆的计算必须进⾏⼀次，对于每个新的预测，两种⽅法都需要进⾏向量-矩阵的乘法，在⾼斯过程中对应向量 $\boldsymbol k^T$ 与矩阵 $\boldsymbol C^{-1}_N\boldsymbol t$ 的运算，两者都是 $N$ 维，因此需要 $O(N^2)$ 次计算；线性基函数模型中变换后的特征矩阵 $\phi(\boldsymbol x)$ 与参数向量 $\boldsymbol w$ 都是 $M$ 维，因此需要 $O(M^2)$ 次计算。如果基函数的数量 $M$ ⽐数据点的数量 $N$ ⼩，那么使⽤基函数计算会更⾼效。但是，正如我们一开始就假设高斯过程是多元高斯分布在无限维的扩展一样，⾼斯过程可以处理那些只能通过⽆穷多的基函数表达的协⽅差函数。

#### 高斯过程神经网络

&emsp;&emsp;高斯过程的神经网络与线性回归的关系并非贝叶斯神经网络与线性回归的关系那样，因为高斯过程是非参模型，所以我们并不在意输出相对参数是否是线性关系，但是同样地，由于神经网络中有较多非线性映射的激活函数，这与基函数是类似的，非常耐人寻味，关于神经网络与高斯过程的联系，就可以从这些激活函数上做文章。目前已有很多相关研究。虽然通常神经网络的非线性单元只选取一个激活函数，但由于我们并不确定哪个激活函数是最优的，这时候就会借助高斯过程，可以看做是对神经网络结构不确定性的一种度量。在贝叶斯神经网络输入维度 $M\rightarrow\infty$ 的情况下，神经⽹络产⽣函数的分布将会趋于⾼斯过程。使用广义谱核 (generalized spectral kernels)，可以证明对若干个激活函数的加权就是一个高斯过程，即

$$f(\boldsymbol x)=\boldsymbol \lambda^T\cdot\phi(\boldsymbol x)=\sum_{m}\lambda^m\phi^m(\boldsymbol x)\tag{30}$$

&emsp;&emsp;对于一个神经网络的第 $l$ 个隐藏层的一个隐藏单元 $i$，其中 $\boldsymbol w^l_i$ 是第 $l$ 层 $i$ 的权重，$\boldsymbol h^{l-1}$ 是前一层的输出向量集合，作为当前层的输入向量，假设每个节点有 $m$ 个激活函数 $\phi(\cdot)$，对应系数为 $\lambda$，那么隐藏单元 $i$ 的输出为

$$h_i^{(l)}=\sum_m\lambda_i^{(l,m)}\phi_m\left ( \boldsymbol w^l_i\boldsymbol h^{l-1} \right )\tag{31}$$

&emsp;&emsp;我们可以使用参数化的方法来解决这种模型，有两类参数，分别是激活函数的系数 $\boldsymbol \lambda$ 和网络参数 $\boldsymbol w$。假设神经网络训练集 $\mathcal D$，对于输入向量 $\boldsymbol x$ 和目标向量 $\boldsymbol y$ 而言，其边缘概率分布为

$$p(\boldsymbol y|\boldsymbol x,\mathcal D)=\int\int p(\boldsymbol y|\boldsymbol x,\boldsymbol w,\boldsymbol \lambda)p(\boldsymbol w|\mathcal D)p(\boldsymbol \lambda|\mathcal D)\mathrm d\boldsymbol w\mathrm d\boldsymbol \lambda\tag{32}$$

&emsp;&emsp;对于单一网络单元 $i$ 的输出，公式 (30) 可以写作

$$h_i^{(l)}=\int\int\lambda_i^{(l,m)}\phi_m\left ( \boldsymbol w^l_i\boldsymbol h^{l-1} \right )p(\boldsymbol w_i^l|\mathcal D)p(\boldsymbol \lambda_i^{(l,m)}|\mathcal D)\mathrm d\boldsymbol w\mathrm d\boldsymbol \lambda\tag{33}$$

其中 $p(\boldsymbol w_i^l|\mathcal D),p(\boldsymbol \lambda_i^{(l,m)}|\mathcal D)$ 分别是激活函数系数以及网络参数的后验概率，这样可以按照贝叶斯神经网络中的变分法进行求解。这种高斯过程在深度学习网络中的应用比较常见，比如在 Transformer 中，我们就可以利用这种做法选定若干个激活函数如 ReLU, GELU, sigmoid, tanh 等，然后获得一个最佳的激活函数加权组合以提高网络性能。

## 3.3 支持向量机 Support Vector Machine

&emsp;&emsp;前⼀章中讨论的核方法在训练阶段需要对所有训练点进行两两计算核函数，由于时间复杂度过高在数据集较大时几乎是不可⾏的，并且在预测时也会花费过多的时间。本章我们会讨论具有稀疏 (sparse) 解的核算法，一种对新数据的预测只依赖于训练数据的⼀个⼦集上计算的核函数，这通常被称为稀疏核机 (sparse kernel machine)。本章重点讨论⽀持向量机 (support vector machine, SVM)，这就是一种稀疏核机，下面会通过一个例子解释它的名称以及稀疏性来源。这是一种很流行的算法，常常被用来解决分类和回归问题。⽀持向量机的⼀个重要性质是模型参数的确定对应于⼀个凸优化问题，因此许多局部解也是全局最优解。

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
