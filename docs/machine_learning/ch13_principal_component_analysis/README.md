# 13 主成分分析 Principal Component Analysis

&emsp;&emsp;主成分分析 (Principal Component Analysis, PCA) 是⼀种被⼴泛应⽤于维度降低、数据压缩、特征提取等领域的方法，顾名思义，它的主要作用就是能够提取出数据中的主要成分或信息，一般可以通过映射到其他空间来实现，因此具有以上应用，它也被称为 Karhunen-Loève 变换。

&emsp;&emsp;主成分分析有两种常用定义，但是其算法思路大致相同。首先，主成分分析可以被定义为数据在低维线性空间上的正交投影，这个线性空间被称为主⼦空间 (principal subspace)，使得投影数据的⽅差被最⼤化，便于区分。另一种定义是使平均投影代价最⼩的线性投影，平均投影代价是指数据点和它们的投影之间的平均平⽅距离。如下图所示，正交投影寻找⼀个低维空间作为主⼦平⾯，⽤紫⾊的线表示，使得红色数据点在⼦空间上的正交投影能够最⼤化绿色投影点的⽅差；另⼀个是基于投影误差的平⽅和的最⼩值，⽤蓝线表示。

<div align=center>
<img src="images/13_1_pca1.png" width="60%"/>
</div>

## 13.1 最大方差形式

&emsp;&emsp;考虑⼀组观测数据集 $\left \{ \boldsymbol x_n \right \}$，其中 $n=1,\dots,N$，$\boldsymbol x_n$ 是⼀个 $D$ 维空间中的向量。我们的⽬标是将数据投影到维度 $M<D$ 的空间中，同时最⼤化投影数据的⽅差。⾸先考虑在⼀维空间 $M=1$ 上的投影，使⽤ $D$ 维向量 $\boldsymbol \mu_1$ 定义这个空间的⽅向，由于我们只对向量的方向感兴趣，不失一般性，假定选择⼀个单位向量，从⽽ $\boldsymbol \mu_1^T\boldsymbol \mu_1=1$。这样，每个数据点 $\boldsymbol x_n$ 被投影到标量值 $\boldsymbol \mu_1^T\boldsymbol x_n$ 上，投影数据的均值是 $\boldsymbol \mu_1^T\bar{ \boldsymbol x}$，$\bar{ \boldsymbol x}$ 是样本集合的均值，形式为

$$\bar{ \boldsymbol x}=\frac{1}{N}\sum_{n=1}^N{ \boldsymbol x}_n\tag1$$

&emsp;&emsp;投影数据的⽅差为

$$\frac{1}{N}\sum_{n=1}^N\left \{ \boldsymbol \mu_1^T\boldsymbol x_n - \boldsymbol \mu_1^T\bar{\boldsymbol x} \right \}^2=\boldsymbol \mu_1^T\boldsymbol S\boldsymbol \mu_1\tag2$$
其中 $\boldsymbol S$ 是数据的协⽅差矩阵，定义为

$$\boldsymbol S=\frac{1}{N}\sum_{n=1}^N\left (\boldsymbol x_n - \bar{\boldsymbol x} \right )\left (\boldsymbol x_n - \bar{\boldsymbol x} \right )^T\tag3$$

&emsp;&emsp;现在关于 $\boldsymbol \mu_1$ 最⼤化投影⽅差 $\boldsymbol \mu_1^T\boldsymbol S\boldsymbol \mu_1$，选择单位向量也是为了避免出现 $\left \| \boldsymbol \mu_1\right \|\rightarrow\infty$，因此引入单位向量归一化条件作为限制，有了这个限制，就可以很自然地引入拉格朗日乘数法，记作 $\lambda_1$，然后对下式进⾏最⼤化

$$\boldsymbol \mu_1^T\boldsymbol S\boldsymbol \mu_1+\lambda_1(1-\boldsymbol \mu_1^T\boldsymbol \mu_1)\tag4$$

&emsp;&emsp;通过令它关于 $\boldsymbol \mu_1$ 的导数等于零，可以看到驻点满⾜

$$\boldsymbol S\boldsymbol \mu_1=\lambda_1\boldsymbol \mu_1\tag5$$

&emsp;&emsp;这表明 $\boldsymbol\mu_1$ 是 $\boldsymbol S$ 的⼀个特征向量，如果左乘 $\boldsymbol\mu_1^T$，根据限制条件，⽅差为

$$\boldsymbol \mu_1^T\boldsymbol S\boldsymbol \mu_1=\lambda_1\tag6$$

&emsp;&emsp;因此当 $\boldsymbol\mu_1$ 设置为与具有最⼤的特征值 $\lambda_1$ 的特征向量相等时，⽅差会达到最⼤值，这个特征向量被称为第⼀主成分。

&emsp;&emsp;对于额外的主成分，如果我们考虑 $M$ 维投影空间的⼀般情形，那么最⼤化投影数据⽅差的最优线性投影由数据协⽅差矩阵 $\boldsymbol S$ 的 $M$ 个特征向量 $\boldsymbol\mu_1,\dots,\boldsymbol\mu_M$ 定义，对应于 $M$ 个最⼤的特征值 $\lambda_1,\dots,\lambda_M$，根据特征值大小依次选择为主成分的方向。

&emsp;&emsp;主成分分析涉及到计算数据集的均值 $\bar{ \boldsymbol x}$ 和协⽅差矩阵 $\boldsymbol S$，然后寻找 $\boldsymbol S$ 对应于 $M$ 个最⼤特征值的 $M$ 个特征向量。计算⼀个 $D\times D$ 矩阵的特征向量分解的复杂度为 $O(D^3)$。如果将数据投影到前 $M$ 个主成分中，我们只需寻找前 $M$ 个特征值和特征向量，这可以使⽤更⾼效的⽅法，例如幂⽅法或 EM 算法。

## 13.2 最小误差形式

&emsp;&emsp;现在讨论主成分分析基于误差最⼩化投影的形式，引⼊ $D$ 维基向量的⼀个完整的单位正交集合 $\left \{ \boldsymbol \mu_i \right \}$，其中 $i=1,\dots,D$，并满⾜

$$\boldsymbol \mu_i ^T\boldsymbol \mu_j=\sigma_{ij}\tag7$$

&emsp;&emsp;每个数据点可以精确地表示为基向量的⼀个线性组合，即

$$\boldsymbol x_n=\sum_{i=1}^D\alpha_{ni}\boldsymbol \mu_i\tag8$$
其中，系数 $\alpha_{ni}$ 对于不同的数据点来说是不同的。这对应于将坐标系旋转到了⼀个由 $\left \{ \boldsymbol \mu_i \right \}$ 定义的新坐标系，原始 $D$ 个分量 $\left \{ x_{n1},\dots,x_{nD} \right \}$ 被替换为⼀个等价的集合 $\left \{ \alpha_{n1},\dots,\alpha_{nD} \right \}$。与 $\boldsymbol\mu_j$ 做内积，然后使⽤单位正交性质，有 $\alpha_{nj}=\boldsymbol x_n^T\boldsymbol \mu_j$，不失⼀般性，有

$$\boldsymbol x_n=\sum_{i=1}^D\left ( \boldsymbol x_n^T\boldsymbol\mu_i \right )\boldsymbol\mu_i\tag9$$

&emsp;&emsp;我们的⽬标是使⽤限定数量 $M<D$ 个变量的⼀种表示⽅法来近似数据点，这对应于低维⼦空间上的⼀个投影，$M$ 维线性⼦空间可以⽤前 $M$ 个基向量表示，因此可以⽤下式来近似每个数据点

$$\tilde{\boldsymbol x}_n=\sum_{i=1}^Mz_{ni}\boldsymbol \mu_i+\sum_{i=M+1}^Db_i\boldsymbol \mu_i\tag{10}$$
其中 $\left \{z_{ni}\right \}$ 依赖于特定数据点，⽽ $\left \{b_{i}\right \}$ 是常数，对所有数据点都相同。我们可以任意选择 $\left \{ \boldsymbol \mu_i \right \}$，$\left \{z_{ni}\right \}$ 和 $\left \{b_{i}\right \}$，从⽽最⼩化维度降低引⼊的失真。作为失真的度量，我们使⽤原始数据点与它的近似点 $\tilde{\boldsymbol x}_n$ 之间的平⽅距离，然后在数据集上取平均。因此最⼩化

$$E=\frac{1}{N}\sum_{n=1}^N\left \| \boldsymbol x_n-\tilde{\boldsymbol x}_n \right \|^2\tag{11}$$

&emsp;&emsp;⾸先考虑关于 $\left \{z_{ni}\right \}$ 的最⼩化，消去 $\tilde{\boldsymbol x}_n$，令它关于 $z_{nj}$ 的导数为零，然后使⽤单位正交条件，有

$$z_{nj}=\boldsymbol x_n^T\boldsymbol \mu_j\tag{12}$$

&emsp;&emsp;令 $E$ 关于 $\left \{b_{i}\right \}$ 的导数等于零，再次使⽤单位正交，有

$$b_j=\bar{\boldsymbol x}^T\boldsymbol \mu_j\tag{13}$$

&emsp;&emsp;消去公式 (10) 中的 $z_{ni}$ 和 $b_i$，使⽤⼀般的展开式 (9)，有

$$\boldsymbol x_n-\tilde{\boldsymbol x}_n=\sum_{i=M+1}^D\left \{ (\boldsymbol x_n-\bar{\boldsymbol x})^T\boldsymbol \mu_i \right \}\boldsymbol \mu_i\tag{14}$$

&emsp;&emsp;可以看到，从 $\boldsymbol x_n$ 到 $\tilde{\boldsymbol x}_n$ 的位移向量位于与主⼦空间垂直的空间中，因为它是 $\left \{ \boldsymbol \mu_i \right \}$ 的线性组合，其中 $i=M+1,\dots,D$，而主子空间的投影向量由前 $M$ 个基向量线性组合而成，在后面空间上的分量为 0。投影点⼀定位于主⼦空间内，最⼩误差由正交投影给出。于是可以得到失真度量 $E$ 的表达式，它是⼀个关于 $\left \{ \boldsymbol \mu_i \right \}$ 的函数，形式为

$$E=\frac{1}{N}\sum_{n=1}^N\sum_{i=M+1}^D\left ( \boldsymbol x_n^T\boldsymbol \mu_i-\bar{\boldsymbol x}^T\boldsymbol \mu_i \right )^2=\sum_{i=M+1}^D\boldsymbol \mu_i^T\boldsymbol S\boldsymbol \mu_i\tag{15}$$

&emsp;&emsp;最后是关于 $\left \{ \boldsymbol \mu_i \right \}$ 对 $E$ 最⼩化，同样需要单位正交条件限制 $\boldsymbol \mu_i$ 的绝对值大小，同之前一样，解可以表示为协⽅差矩阵的特征向量展开式。先考虑⼆维数据空间 $D=2$ 以及⼀维主⼦空间 $M=1$ 的情形，选择⼀个⽅向 $\boldsymbol\mu_2$ 来最⼩化 $E=\boldsymbol\mu_2^T\boldsymbol S\boldsymbol\mu_2$，同时满⾜限制条件 $\boldsymbol\mu_2^T\boldsymbol\mu_2=1$。使⽤拉格朗⽇乘数 $\lambda_2$ 引入这个限制，最⼩化

$$\tilde E=\boldsymbol\mu_2^T\boldsymbol S\boldsymbol\mu_2+\lambda_2\left ( 1-\boldsymbol\mu_2^T\boldsymbol\mu_2 \right )\tag{16}$$
令关于 $\boldsymbol\mu_2$ 的导数为零，有 $\boldsymbol S\boldsymbol\mu_2=\lambda_2\boldsymbol\mu_2$，从⽽ $\boldsymbol\mu_2$ 是 $\boldsymbol S$ 的⼀个特征向量，且特征值为 $\lambda_2$。因此任何特征向量都会定义失真度量的⼀个驻点，为了找到 $E$ 在最⼩值，将 $\boldsymbol\mu_2$ 的解代回失真度量中，得到 $E=\lambda_2$。于是通过将 $\boldsymbol\mu_2$ 选择为特征值较⼩的特征向量，就可以得到 $E$ 的最⼩值。因此，应该将主⼦空间与具有较⼤特征值的特征向量对齐，即为了最⼩化平均平⽅投影距离，应将主成分⼦空间选为穿过数据点的均值且与最⼤⽅差⽅向对齐。对于特征值相等的情形，任何主⽅向的选择都会得到同样的 $E$ 值。 对于任意的 $D$ 和 $M<D$，最⼩化 $E$ 的⼀般解都可以通过将 $\left \{ \boldsymbol \mu_i \right \}$ 选择为协⽅差矩阵特征向量的⽅式得到，即

$$\boldsymbol S\boldsymbol\mu_i=\lambda_i\boldsymbol\mu_i\tag{17}$$
其中 $i=1,\dots,D$，特征向量 $\left \{ \boldsymbol \mu_i \right \}$ 单位正交，失真度量的值为

$$E=\sum_{i=M+1}^D\lambda_i\tag{18}$$

&emsp;&emsp;这就是与主⼦空间正交的特征值的加和。通过将这些特征向量选择成 $D-M$ 个最⼩的特征值对应的特征向量，来得到 $E$ 的最⼩值，并且定义了主⼦空间的特征向量是对应于 $M$ 个最⼤特征值的特征向量。

&emsp;&emsp;虽然已经考虑了 $M<D$ 的情形，但主成分分析对于 $M=D$ 的情形仍然成⽴，这种情况下没有维度的降低，仅仅是将坐标轴旋转，与主成分对齐即可。

## 13.3 高维数据主成分分析

&emsp;&emsp;在主成分分析的⼀些应⽤中，数据点的数量⼩于数据空间的维度。例如将主成分分析应⽤于⼏百张图⽚组成的数据集，每个图⽚对应于⼏万维空间中的⼀个向量。在⼀个 $D$ 维空间中，$N$ 个数据点 $(N<D)$ 定义了⼀个线性⼦空间， 它的维度最多为 $N-1$，因此在使⽤主成分分析时，⼏乎没有 $M>N-1$ 的数据点。在主成分分析中，⾄少有 $D-N+1$ 个特征值为零，对应于沿着数据集⽅差为零⽅向的特征向量。此外，寻找 $D\times D$ 矩阵的特征向量的算法复杂度为 $O(D^3)$，因此对于诸如图像这种应⽤来说，直接应⽤主成分分析算法是不可⾏的。

&emsp;&emsp;可以这样解决这个问题，⾸先，将 $\boldsymbol X$ 定义为 $(N\times D)$ 维中⼼数据矩阵，它的第 $n$ ⾏为 $(\boldsymbol x_n-\bar{\boldsymbol x})^T$，这样协⽅差矩阵可以写成 $\boldsymbol S=N^{-1}\boldsymbol X^T\boldsymbol X$，对应特征向量⽅程变成了

$$\frac{1}{N}\boldsymbol X^T\boldsymbol X\boldsymbol \mu_i=\lambda_i\boldsymbol \mu_i\tag{19}$$

&emsp;&emsp;将两侧左乘 $\boldsymbol X$，可得

$$\frac{1}{N}\boldsymbol X\boldsymbol X^T(\boldsymbol X\boldsymbol \mu_i)=\lambda_i(\boldsymbol X\boldsymbol \mu_i)\tag{20}$$

&emsp;&emsp;定义 $\boldsymbol v_i=\boldsymbol X\boldsymbol \mu_i$，有

$$\frac{1}{N}\boldsymbol X\boldsymbol X^T\boldsymbol v_i=\lambda_i\boldsymbol v_i\tag{21}$$

&emsp;&emsp;它是 $N\times N$ 矩阵 $\frac{1}{N}\boldsymbol X\boldsymbol X^T$ 的⼀个特征向量⽅程。这个矩阵与原始协⽅差矩阵具有相同的 $N-1$ 个特征值，原始协⽅差矩阵本⾝有额外的 $D-N+1$ 个值为零的特征值，因此可以在低维空间中解决特征向量问题，计算复杂度为 $O(N^3)$ ⽽不是 $O(D^3)$。为了确定特征向量，将公式 (21) 两侧乘以 $\boldsymbol X^T$，可得

$$\left ( \frac{1}{N}\boldsymbol X^T\boldsymbol X \right )\left ( \boldsymbol X^T\boldsymbol v_i \right )=\lambda_i\left ( \boldsymbol X^T\boldsymbol v_i \right )\tag{22}$$

&emsp;&emsp;从中可以看到 $\left ( \boldsymbol X^T\boldsymbol v_i \right )$ 是 $\boldsymbol S$ 的⼀个特征向量，对应特征值为 $\lambda_i$，这些特征向量的长度未必等于 1。为了确定合适的归⼀化，我们使⽤⼀个常数来对 $\boldsymbol \mu_i \propto \boldsymbol X^T\boldsymbol v_i$ 进⾏标度，使得 $\left \| \boldsymbol \mu_i \right \|=1$。假设 $\boldsymbol v_i$ 的长度已经被归⼀化，那么有

$$\boldsymbol \mu_i =\frac{1}{(N\lambda_i)^{\frac{1}{2}}}\boldsymbol X^T\boldsymbol v_i\tag{23}$$

&emsp;&emsp;在高维空间应用主成分分析方法的思路就是先计算 $\boldsymbol X\boldsymbol X^T$，然后找到其特征值和特征向量，最后再计算原始数据空间的特征向量。