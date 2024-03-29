# 3 神经网络 Neural Networks

&emsp;&emsp;神经⽹络 (neural networks) 是人工智能三大学派中联结主义 (connectionism) 的主要研究对象，联结主义又称为仿生学派 (bionicsism)，是一门主张使用仿生学，尤其是人脑模型，进行人工智能研究的学派。最初使用电子装置从神经元突触等开始模仿人体神经系统结构功能，第三篇文章所讲的的感知器就是对神经系统的模拟，它最初就是使用光电管的矩阵阵列来实现的。

&emsp;&emsp;我们通常所讲的人工神经网络是一种深度前馈神经网络，也是一种多层感知器。后来随着计算机的发展以及深度学习算法的提出，神经网络的运算得到了极大提升，衍生出各种各样的结构，从最初的多层感知器 (MLP, multi-layer perceptron)，到循环神经网络 (recurrent neural networks)，卷积神经网络 (convolutional neural netowrks)，以及当前最热门的基于注意力机制 (Attention) 的 Transformer，在图像识别，自然语言处理等方向均取得巨大成果，在很多领域都成为主流研究方法。

&emsp;&emsp;神经网络的名称或多或少受到些神经科学的启发，神经网络的节点也因此被称作神经元 (neuron)，而激活函数也类似于突触结构的运作机理。但从本质上来看，虽然神经网络最初是一种使用数学和物理实现⽣物系统的信息处理的研究模型，实际上，除了结构有一些相似性，它和人体神经系统就没任何关系了，现代神经网络的研究更多受到数学和工程学科的指引，并且神经网络的目标并不是完美地给神经系统建模。应该说前馈神经网络是为了实现统计泛化而设计出的函数近似机，它偶尔从我们了解的大脑神经系统中提取灵感，但并不是人体神经系统的模型，并且在结构上，如果一味地模仿人体神经系统，反而会带来诸多限制。

&emsp;&emsp;一个单层神经网络就类似于前面讲过的单层感知器，从结构上讲，它是一种有向图 (directional graph)，输入输出都是节点，而权重则组成连接这些节点的有向弧。相比感知器，神经网络可以更灵活选择输出层的非线性单元，并不需要只做出 -1 和 1 这样的分类判断，多层感知器之所以要这样做，是因为受到硬件实现的限制只能采用这样的方式，而神经网络就可以提供各种非线性函数从而灵活地应用到回归，二分类，多分类等问题。

&emsp;&emsp;神经网络当下应用极其广泛，相关知识量巨大，且更偏向深度学习范畴，本文只是重点介绍一下神经网络相对前几篇线性回归分类的几个模型的一些改进，从深度前馈神经网络这一基本模型说起。

## 3.1 深度前馈神经网络 Deep Feed-forward Neural Networks

&emsp;&emsp;我们在前几篇文章讨论了不少线性回归和分类模型，用一个统一的方式表示，即

$$y(\boldsymbol x, \boldsymbol w)=f(\boldsymbol w^T\boldsymbol x)\tag{1}$$
其中 $f(\cdot)$ 在分类问题中是⼀个⾮线性激活函数，在回归问题中为恒等函数。在第一篇文章中我们提到过一种更一般的形式

$$y(\boldsymbol x, \boldsymbol w)=f(\boldsymbol w^T\phi(\boldsymbol x))\tag2$$
其中 $\phi(\cdot)$ 是基函数。我们在前面提到过，使用基函数对输入向量进行非线性变换，可以提取更丰富的特征从而更有效的建立模型。为了进一步提取特征，我们可以尝试搭建多层的神经网络，在每一层网络之后都可以选择性增加一个非线性单元。这与神经系统的组成也是相似的，人体神经网络就是由无数个神经元组成，神经元之间借助类似于非线性激活函数的突触结构单向传导神经冲动，而如果选用 ReLU 这样的激活函数的话，其实现功能甚至与突触都是类似的，我们在后文会提及。

&emsp;&emsp;在多层的神经网络中，我们将前一层的输出作为下一层的输入，每一层都是类似公式 (1) 的形式。这种神经网络被称为前馈 (feedforward) 神经网络，是因为信息从 $\boldsymbol x$ 的输入向量起始，流经并作用于 $f(\cdot)$ 所定义的函数，最终到达输出 $\boldsymbol y$，在模型输出和模型本身之间没有反馈 (feedback) 连接，从图模型角度看就是一个有向无环图，前馈神经网络用许多不同函数复合在一起来表示。

&emsp;&emsp;我们假设有一个三层的神经网络，如下图所示

<div align=center>
<img src="images/6_1_ffn1.png" width="55%"/>
</div>

&emsp;&emsp;每层与每层之间对数据进行一次映射，分别为 $f^{(1)},f^{(2)}$，具有权重 $\boldsymbol w^{(1)},\boldsymbol w^{(2)}$，输入为 $D$ 维向量 $\boldsymbol x$，输出为 $K$ 维向量 $\boldsymbol y$，它们连在一个链上就可以写作

$$\boldsymbol y=f(\boldsymbol x)=f^{(2)}({\boldsymbol w^{(2)}}^Tf^{(1)}({\boldsymbol w^{(1)}}^T\boldsymbol x))\tag3$$

&emsp;&emsp;链的全长或神经网络的层数称为模型的深度 (depth)，也正是因为这个术语才出现了 "深度学习" (deep learning) 这个名字，并且一定范围内，模型的性能总是随着层数增加而提升。可以看出，神经网络的第一层用于接收输入数据 $\boldsymbol x$，最后一层用于产生模型输出 $\boldsymbol y$ 与训练数据中的目标值相参考，而中间层并没有明确的输入和输出，因此把这些中间层也称作隐藏层 (hidden layer)，隐藏层的节点 $\boldsymbol z$ 被称为隐藏单元 (hidden unit)。

&emsp;&emsp;神经网络中隐藏层和输出层可以选择不同的非线性函数。通常输出层我们根据回归，二分类，多分类等任务选择恒等，logistic sigmoid 函数或 softmax 函数，隐藏层的⾮线性函数 $f^{(1)}$ 通常被选为 $S$ 形函数，例如 logistic sigmoid 函数或者双曲正切函数，与多层感知器相⽐，神经⽹络在隐含单元中使⽤连续⾮线性函数，⽽感知器使⽤阶梯函数这⼀⾮线性函数。这意味着神经⽹络函数关于神经⽹络参数是可微的，这个性质在神经⽹络的训练过程中起着重要的作⽤。

&emsp;&emsp;如果⽹络中的所有隐藏单元的激活函数都取线性函数，那么对于任何这种⽹络，我们总可以 找到⼀个等价的⽆隐藏单元的⽹络。这是由于连续的线性变换的组合本⾝仍旧是⼀个线性变换。然⽽，如果隐藏单元数量⼩于输⼊或输出单元的数量，那么⽹络能够产⽣的变换不是直接从输⼊到输出的线性变换，因为在隐藏单元的维度降低造成了信息丢失。这一性质也启发我们在设计网络时 $\boldsymbol z$ 的维度一定要大于 $\boldsymbol x$ 和 $\boldsymbol y$。

&emsp;&emsp;上文的神经⽹络结构是实际中最常⽤也是形式上最简单的⼀个，它很容易扩展，最常见的是增加隐藏层，或者在节点级别使⽤不同的激活函数进行非线性变换。另⼀个扩展是引⼊跨层 (skip-layer) 连接，每个跨层连接都对应⼀个可调节参数，即对于某些输入单元直接映射到输出。具有 sigmoid 隐藏单元的⽹络能够模拟跨层连接，因为 sigmoid 在 0 点附近可以近似为线性的，只要从输入到隐藏层使用足够小的权值，然后将隐藏层到输出的权值设得⾜够⼤进⾏补偿就可以实现。当然实际操作我们还是会选择显示地包含跨层连接。

&emsp;&emsp;神经网络的误差函数，输出单元的非线性函数，以及训练方式等与前面线性回归和分类基本类似，也是用最大似然法构造最小误差平方或者交叉熵等函数用于梯度下降，此处不做赘述。神经网络效用很大程度都是依赖中间的隐藏层的，它实际上是把输入变量中的低维数据映射到高维空间，然后提取更丰富的特征，在输出层在进行降维，这才是我们感兴趣的，研究神经网络的性质，就必须搞清楚这些隐藏单元的作用。

## 3.2 隐藏单元 Hidden Unit

&emsp;&emsp;在前馈神经网络中关于隐藏单元我们最感兴趣的无非是，如何选择隐藏层中隐藏单元的数量和类型。隐藏单元的设计是一个非常热门的研究领域，到现在依然没有明确的理论指导，因此隐藏单元的类型选择十分困难。我们往往基于经验和直觉来尝试一些单元，但不可能预先预测出哪种隐藏单元工作得最好，设计过程充满了试验和错误，先直觉认为某种隐藏单元可能表现良好，然后用它组成神经网络进行训练，最后用验证集来评估它的性能。这也是神经网络及其所有扩展的变体共存的问题，由于中间过程过于复杂，可解释性相比其他模型很差，所以神经网络往往都被看作黑盒模型。不过幸运的是，经过大量的实验验证，我们已经找到几个相当不错的隐藏单元选择。

#### 整流线性单元 Rectified Linear Unit

&emsp;&emsp;整流线性单元是隐藏单元一个效果相当不错的默认选择，它使用激活函数

$$h(x)=\max \left \{ 0,x \right \}tag4$$

&emsp;&emsp;线性单元和整流线性单元的唯一区别在于整流线性单元在其一半的定义域上输出为零，这也使得它们相对容易优化，只要整流线性单元处于激活状态，它的二阶导数几乎处处为 0，一阶导数处处为 1。相比于引入二阶效应的激活函数来说，它的梯度方向对学习更加有用。

&emsp;&emsp;然而整流线性单元在 $x=0$ 处不可微，这似乎使得它对基于梯度的学习算法无效。但在实践中，梯度下降对这些机器学习模型仍然表现得足够好，部分原因是神经网络训练算法通常不会达到误差函数的局部最小值，而是仅仅减小，因为我们不期望训练能够实际到达梯度为 0 的点，所以误差函数的最小值对应于梯度未定义的点是可接受的。

&emsp;&emsp;不可微的隐藏单元通常只在少数点上不可微，即使发生这种情况，$h(x)$ 在 $x=0$ 处的左导数为 0，右导数为 1。神经网络训练的软件实现通常返回左导数或右导数的其中一个，而不是报告导数未定义或产生一个错误。当一个函数被要求计算 $h(x)$ 时，底层值真正为 0 是不太可能的。它可能是被舍入为 0 的一个极小量 $\epsilon$ 。在实践中，我们可以放心地忽略整流线性单元的不可微性。

&emsp;&emsp;对于多分类问题，在输出层值最大的那个单元会被选做输出类别，输入层和隐藏层的单元代表的是特征，神经网络中的种种变换其实都是使提取选择特征然后使有用的特征最大化。整流线性单元，我们可以将其看做一个 “优胜劣汰” 的过程，相对明显的特征才能被保留，其余就会被筛掉，卷积神经网络的最大池化层操作也是类似的原理。整流线性单元的效用依赖于初始化和归一化，因为我们通常会选择在 0 附近进行初始化，而归一化后也可以将特征向量集中分布到 0 点附近，此时整流线性单元就可以很好地做筛选。

&emsp;&emsp;有很多整流线性单元的扩展存在，因为整流线性单元的一个缺陷是不能通过基于梯度的方法学习那些使它们激活为 0 的特征，所以它的各种扩展保证了能在各个位置都计算梯度。整流线性单元可以扩展为当 $x<0$ 时使用一个非零的斜率 $a:h(x)=\max\left \{ 0,x \right \}+a\min\left \{ 0,x \right \}$，关于 $a$ 的选择有三种方法。

&emsp;&emsp;1. 绝对值整流 (absolute value rectiﬁcation) 固定 a=-1 来得到 $g(x)=|x|$，它用于图像中的对象识别，对于寻找在输入照明极性反转下不变的特征是很有意义的；2. 渗漏整流线性单元 (Leaky ReLU) 将 a 固定成一个类似 0.001 的极小值；3. 参数化整流线性单元 (parametric ReLU) 将 a 作为学习的参数。

#### logistic sigmoid 与双曲正切函数

&emsp;&emsp;在引入整流线性单元之前，大多数神经网络使用 logistic sigmoid 激活函数 $g(x)=\sigma (x)$ 或双曲正切函数 $g(x)=\tanh (x)$，其中 $\tanh(x)=2\sigma(2x)-1$。

&emsp;&emsp;前文中 sigmoid 函数作为输出单元用来预测二分类问题取值为 1 的概率，与整流线性单元不同，sigmoid 函数在其大部分定义域内都饱和，当 x 取绝对值很大的值时，函数就会饱和到 0 或 1，并且仅仅当 x 接近 0 时它们才对输入强烈敏感，因为在 0 附近导数较大。sigmoid 单元的广泛饱和性会使得基于梯度的学习变得非常困难，所以现在很少将它们用作前馈网络中的隐藏单元，除非选择一个合适的误差函数来抵消 sigmoid 的饱和性时，它们才能作为输出单元可以与基于梯度的学习算法相兼容。

&emsp;&emsp;当必须使用 sigmoid 激活函数时，双曲正切激活函数通常要比 logistic sigmoid 表现更好。$\tanh(0)=0$ 而 $\sigma(0)=\frac{1}{2}$，所以 $\tanh(x)$ 在 0 附近更像是单位函数，训练深层神经网络 $y={\boldsymbol {w}^{(3)}}^T\tanh({\boldsymbol {w}^{(2)}}^T\tanh({\boldsymbol {w}^{(1)}}^T\boldsymbol x))$ 在所有激活都保持较小的时候，类似于训练 $y={\boldsymbol {w}^{(3)}}^T{\boldsymbol {w}^{(2)}}^T{\boldsymbol {w}^{(1)}}^T\boldsymbol x$，这使得训练 $\tanh(x)$ 网络更加容易。

## 3.3 反向传播 Back Propagation

&emsp;&emsp;当我们使用前馈神经网络接收输入 $\boldsymbol x$ 并产生输出 $\boldsymbol y$ 时，信息通过网络向前流动。输入 $\boldsymbol x$ 提供初始信息，然后传播到每一层的隐藏单元，最终产生输出 $\boldsymbol y$，这称之为前向传播 (forward propagation)。在训练过程中，前向传播可以持续向前直到产生一个标量误差函数 $E(\boldsymbol x)$。 反向传播 (back propagation) 算法则允许来自误差函数的信息通过网络向后流动，以便计算梯度。计算梯度的解析表达式是很直观的，但是数值化地求解在计算上的代价可能很大，反向传播算法可以在程序上非常简便地实现这个目标。

&emsp;&emsp;反向传播是用于计算梯度的方法，而使用梯度进行学习的则是梯度下降算法，两者共同作用，反向传播可以计算任何函数的导数。在机器学习算法中，我们最常需要的梯度是误差函数关于参数的梯度，即 $\nabla_{\boldsymbol w}E(\boldsymbol w)$。通过在网络中传播信息来计算导数的想法非常普遍，它还可以用于计算多输入输出函数 $f(\boldsymbol x)$ 的 Jacobian 矩阵。

#### 计算图

&emsp;&emsp;神经网络本质上是一种有向无环图，而为了更精确地描述反向传播算法，可以使用更精确的计算图 (computational graph) 语言。将计算形式化为图形的方法有很多，这里，我们使用图中的每一个节点来表示一个变量，变量可以是标量、向量、矩阵、张量、或者甚至是另一类型的变量。此外还需引入操作 (operation) 这一概念，操作是指一个或多个变量的简单函数，图形语言往往伴随着一组操作，可以通过将多个操作复合在一起来描述更为复杂的函数。如果变量 y 是变量 x 通过一个操作计算得到的，那么我们画一条从 $x$ 到 $y$ 的有向边。下图给出了一些计算图的示例，分别表示 (a) $z=xy$；(b) $\hat y=\sigma(\boldsymbol w^T\boldsymbol x+b)$；(c) $\boldsymbol H=\max\left \{ 0, \boldsymbol X\boldsymbol W+\boldsymbol b \right \}$；(d) $\hat y=\boldsymbol w^T\boldsymbol x$，$\mu^{(3)}=\lambda\sum_iw_i^2$。其中 dot 表示向量相乘，matmul 表示包含小批量的张量指定维度的矩阵乘法，(d) 表示对变量实施多个操作，代表技能预测 \hat y 也能用于权重衰减的惩罚项。


图片来自 Deep Learning. Figure 6.8.
#### 链式法则
&emsp;&emsp;微积分中的链式法则用于计算复合函数的导数，在反向传播中广泛使用。设 $x$ 是实数，$f$ 和 $g$ 是从实数映射到实数的函数。假设 $y=g(x)$ 且 $z=f(y)=f(g(x))$，那么链式法则是

$$\frac{\mathrm d z}{\mathrm d x}=\frac{\mathrm dz}{\mathrm dy} \frac{\mathrm dy}{\mathrm dx}\tag 5$$

&emsp;&emsp;如果扩展到向量，假设 $\boldsymbol x\in\mathbb R^m,\boldsymbol y\in\mathbb R^n$，$g$ 是从 $\mathbb R^m$ 到 $\mathbb R^n$ 的映射，$f$ 是从 $\mathbb R^n$ 到 $\mathbb R$ 的映射，并且有 $y=g(\boldsymbol x)$ 且 $z=f(\boldsymbol y)=f(g(\boldsymbol x))$， 则有

$$\frac{\mathrm d z}{\mathrm d x_i}=\sum_j\frac{\mathrm d z}{\mathrm d y_j}\frac{\mathrm d y_j}{\mathrm d x_i}\tag6$$

&emsp;&emsp;使用向量记法，可以等价地写成

$$\nabla_{\boldsymbol xz}=\left ( \frac{\partial \boldsymbol y}{\partial \boldsymbol x} \right )^T\nabla_{\boldsymbol yz}\tag7$$
这里 $\frac{\partial \boldsymbol y}{\partial \boldsymbol x}$ 是 $g$ 的 $n\times m$ 的 Jacobian 矩阵。反向传播算法可以应用于任意维度的张量，在运行反向传播之前，将每个张量变平为一个向量，计算一个向量值梯度，然后将该梯度重新构造成一个张量，这与使用向量的反向传播基本相同。

&emsp;&emsp;使用链式法则，我们可以直接写出某个标量关于计算图中任何产生该标量节点的梯度表达式，但在实际操作中不得不面临一些问题。

&emsp;&emsp;许多子表达式可能在整个梯度表达式中重复若干次，任何计算梯度的程序都需要选择是存储这些子表达式还是重新计算。某些情况下，计算两次相同的子表达式纯粹是浪费。在复杂图中，可能存在指数多的这种计算上的浪费，使得简单的链式法则不可实现。在其他情况下，计算两次相同的子表达式可能是以较高的运行时间为代价来减少内存开销的有效手段。在实际操作中，我们首先考虑的是降低时间复杂度的问题，一般会存储这些梯度数据。

#### 误差函数的反向传播应用

&emsp;&emsp;我们现在推导适⽤于一般前馈神经⽹络的反向传播算法，神经⽹络具有任意可微的⾮线性激活函数，以及⼀⼤类的误差函数。推导的结果将会使⽤⼀个简单的⽹络结构说明，这个神经⽹络有⼀个单层的 sigmoid 隐含单元以及平⽅和误差函数。我们就使用针对⼀组独⽴同分布的数据的最⼤似然⽅法定义的误差函数，由若⼲项的求和式组成，每⼀项对应于包含 N 个数据的训练集的某个数据点，即

$$E(\boldsymbol w)=\sum_{n=1}^NE_n(\boldsymbol w)\tag8$$

&emsp;&emsp;首先考虑计算 $\nabla E(\boldsymbol w)$，考虑⼀个简单的线性模型，其中输出 $y_k$ 是输⼊变量 $x_i$ 的线性组合，即

$$y_k=\sum_iw_{ki}x_i\tag9$$

对于⼀个特定的输⼊数据 $n$，误差函数的形式为

$$E_n=\frac{1}{2}\sum_k(y_{nk}-t_{nk})^2\tag{10}$$
这个误差函数关于权值 $w_{ji}$ 的梯度为

$$\frac{\partial E_n}{\partial w_{ji}} = (y_{nj}-t_{nj}) x_{ni}\tag{11}$$

它可以表示参数权值 $w_{ji}$ 的输出端相关联的误差 $y_{nj}-t_{nj}$ 和输⼊端相关联的变量 $x_{ni}$ 的乘积。在⼀般的前馈⽹络中，每个单元都会计算输⼊的⼀个加权和，形式为

$$a_j=\sum_iw_{ji}z_i\tag{12}$$
其中 $z_i$ 是激活函数的输出或原始输⼊，它通过权值 $w_{ji}$ 与下一层单元节点 $j$ 连接。再通过⾮线性激活函数 $h(\cdot)$ 变换后，得到单元 $j$ 的输出 $z_j$，形式为

$$z_j=h(a_j)\tag{13}$$
对于训练集⾥的每个数据点，都作为对应的输入向量输入到神经网络，然后通过反复应⽤公式 (12) 和 (13)，计算神经⽹络中所有隐藏单元和输出单元的输出，这个过程就是正向传播 (forward propagation)，它可以被看做⽹络中的⼀个向前流动的信息流。

&emsp;&emsp;现在考虑计算 $E_n$ 关于权值 $w_{ji}$ 的导数。此处讨论的仍是针对数据点 $n$ 的操作，神经网络的输入和输出都与 $n$ 有关，为了保持简洁，以下将 $n$ 全部省略掉。$E_n$ 只通过单元 $j$ 的输出 $a_j$ 对权值 $w_{ji}$ 产⽣依赖。因此，可以应⽤偏导数的链式法则， 得到

$$\frac{\partial E_n}{\partial w_{ji}} = \frac{\partial E_n}{\partial a_{j}} \frac{\partial a_j}{\partial w_{ji}} \tag{14}$$
令 $\delta_j\equiv\frac{\partial E_n}{\partial a_{j}}$，结合公式 (12)，可得

$$\frac{\partial E_n}{\partial w_{ji}} =\delta_j z_i\tag{15}$$

&emsp;&emsp;根据上式可知，我们要找的导数可以通过简单地将权值输出单元的 $\delta$ 值与权值输⼊端的 $z$ 值相乘得到。因此，计算导数只需要计算⽹络中每个隐藏结点和输出结点的 $\delta$ 值，然后应⽤链式法则即可。结合下图

<div align=center>
<img src="images/6_1_ffn2.png" width="40%"/>
</div>

&emsp;&emsp;此处声明一下，我们假设一个三层的神经网络，下标 $i$ 代表输入层，下标 $j$ 代表隐藏层，下标 $k$ 代表输出层。上图蓝色箭头表示正向传播，红色箭头表示反向传播。在输出层，有

$$\delta_k=\frac{\partial E_n}{\partial a_{k}} =y_k-t_k\tag{16}$$

&emsp;&emsp;为了计算隐藏单元的 $\delta$ 值，我们再次使⽤偏导数的链式法则

$$\delta_j\equiv\frac{\partial E_n}{\partial a_{j}}=\sum_k\frac{\partial E_n}{\partial a_{k}}\frac{\partial a_k}{\partial a_{j}}\tag{17}$$
其中求和式的作⽤对象是所有与单元 $j$ 相关联的单元 $k$。对于多层神经网络而言，$k$ 也可以是隐藏单元。这也说明 $a_j$ 的改变所造成的误差函数的改变与所有相关联的 $a_k$ 都有关。如果将公式 (12) 和 (13) 代入，我们就得到了下⾯的反向传播 (backpropagation) 公式

$$\delta_j={h}' (a_j)\sum_kw_{kj}\delta_k\tag{18}$$

&emsp;&emsp;这表明⼀个特定的隐藏单元的 $\delta$ 值可以通过将⽹络中更⾼层单元的 $\delta$ 进⾏反向传播来实现。由于我们已经知道输出单元的 $\delta$，因此通过递归地应⽤公式 (17)，我们可以计算前馈⽹络中所有隐藏单元的 $\delta$ 值，⽆论它的网络结构是什么样的。

至此，反向传播算法可以总结为下：

对于神经⽹络的⼀个输⼊向量 $\boldsymbol x_n$，使⽤公式 (12) 和 (13) 进⾏正向传播，找到所有隐藏单元和输出层的输出值。
使用公式 (16) 计算所有输出单元的 $\delta_k$。
使⽤公式 (18) 反向传播，获得神经⽹络中所有隐藏单元的 $\delta_j$。
使⽤公式 (15) 计算对权值参数 $w$ 导数。
在上⾯的推导中，我们隐式地假设⽹络中的每个隐藏单元都有相同的激活函数 $h(\cdot)$。我们也可以设置不同的单元有不同的激活函数，在反向传播算法中只需记录该激活函数的 $\delta$ 即可。

#### 反向传播的效率

&emsp;&emsp;反向传播的⼀个重要⽅⾯是它的计算效率，这在上文提到如果每次每次反向传播都计算一遍 $\delta$，时间开销则呈指数增长。我们考察误差函数导数的计算次数与⽹络中权值总数 $\boldsymbol W$ 的关系。假设计算⼀次误差函数需要 $O(\boldsymbol W)$ 次操作，其中 $\boldsymbol W$ 充分⼤。这是因为在非稀疏的神经网络中，权值数量通常特别大，正向传播的计算复杂度主要取决于公式 (12) 求和式的计算，⽽激活函数的计算就相对耗时较少，求和式的每一项都需要进行一次乘法和加法。

&emsp;&emsp;另⼀种计算误差函数导数反向传播的⽅法是使⽤有限差，让每个权值有⼀个扰动，然后使⽤下⾯的表达式来近似导数

$$\frac{\partial E_n}{\partial w_{ji}} =\frac{E_n(w_{ji}+\epsilon)-E_n(w_{ji})}{\epsilon}+O(\epsilon)\tag{19}$$
其中 $\epsilon\ll1$。程序上通过让 $\epsilon$ 变⼩，可以提升导数的近似精度，但如果 $\epsilon$ 过小就会造成下溢问题。通过使⽤对称的中⼼差 (central difference)，有限差⽅法的精度可以极⼤地提⾼。中⼼差的形式为

$$\frac{\partial E_n}{\partial w_{ji}} =\frac{E_n(w_{ji}+\epsilon)-E_n(w_{ji}-\epsilon)}{2\epsilon}+O(\epsilon^2)\tag{20}$$

&emsp;&emsp;很容易通过泰勒展开证明，但与前一种有限差方法相⽐，计算步骤数⼤约变成了⼆倍。

&emsp;&emsp;计算数值导数的⽅法的主要问题是，每次正向传播需要 $O(\boldsymbol W)$ 步，⽽神经⽹络中有 $\boldsymbol W$ 个权值，每个权值必须被单独地施加扰动，因此整体的时间复杂度为 $O(\boldsymbol W^2)$。然⽽，数值导数的⽅法在实际应⽤中有重要作⽤，因为将反向传播算法计算的导数与使用中⼼差计算的导数进⾏对⽐，可以有效地检查反向传播算法的执⾏正确性。实际应⽤中，训练⽹络时反向传播算法具有最⾼的精度和效率，也应该使⽤⼀些测试样例将结果与数值导数进⾏对⽐，检查执⾏的正确性。

&emsp;&emsp;关于神经网络更详细的研究及其各种变体均属于深度学习范畴，机器学习更侧重探索其中的推导和运行机理，因此本文只讲述了神经网络的发展历史，以及前馈神经网络这一最基础神经网络的结构和原理。前五章重点介绍了机器学习中分类和回归两种任务及其用到的一些模型，下一章将介绍一些机器学习的基础理论知识。


## 1.1 过拟合和正则化 Overfitting and Regularization

#### 过拟合 Overfitting

&emsp;&emsp;重新考察第一篇文章 1.1 节的多项式曲线拟合问题，采样一组由 $\sin 2\pi x$ 产生的数据点，输入 $x$ 是在区间 [0,1] 上均值分布的采样，然后对每个点的标签 t 加上高斯噪声，就得到了数据集。使用多项式函数

$$y(x, \boldsymbol{w})=w_0+w_1\cdot x + w_2\cdot x^2 + \cdot \cdot \cdot +w_K\cdot x^K=\sum_{j=0}^{K} w_j\cdot x^j\tag{1}$$

&emsp;&emsp;进行曲线拟合，我们分别选择多详述阶数 $M=0,1,3,9$。正常情况判断是否过拟合需要计算训练集和测试集的误差函数，此处为了方便起见，由于我们知道数据是从 $\sin⁡2\pi x$ 采样的，所以我们只需要观察拟合的曲线在区间 [0,1] 上与 $\sin⁡2\pi x$ 偏差即可。如下图是四个不同的阶数的曲线拟合结果

<div align=center>
<img src="./../images/3_1_fit1.png" width="70%"/>
</div>

&emsp;&emsp;我们注意到 $M=0,1$ 时多项式对于数据的拟合效果相当差。三阶多项式似乎给出了对 $\sin⁡2\pi x$ 的最好拟合。当我们达到更⾼阶的多项式 ($M=9$)，我们得到了对于训练数据的⼀个完美的拟合。 事实上， 多项式函数精确地通过了每⼀个数据点，误差函数降到了 0。 然⽽，拟合的曲线剧烈震荡，与 $\sin⁡2\pi x$ 相去甚远，发生了过拟合，如果在测试集上做预测，可想而知效果是很差的。我们可以观察训练集和测试集的均方和误差，或者更一般的使用根均方误差 $E_{RMS}=\sqrt{2E(\boldsymbol w^\star)/N}$ ，这样可以确保与目标变量 $t$ 使⽤相同规模的单位进⾏度量，下图展示了 $M$ 取 [0,9] 上所有整数时训练集与测试集根均方的情况。

<div align=center>
<img src="images/3_1_fit2.png" width="50%"/>
</div>

&emsp;&emsp;对于 $M=9$ 的情形，训练集的误差为 0，因为此时的多项式函数有 10 个⾃由度，对应于 10 个系数 $\left \{w_0, w_1,...,w_9 \right \}$，调节后使得模型与训练集中的 10 个数据点精确匹配。但是它在测试集上却突然变差，这可能看起来很⽭盾，因为给定阶数的多项式包含了所有低阶的多项式函数作为特殊情况。 $M=9$ 的多项式因此能够产⽣⾄少与 $M=3$ ⼀样的结果。同时由于生成数据的函数 $\sin⁡2\pi x$ 的幂级数展开包含所有阶数的项，所以我们直观地认为结果会随着 $M$ 的增⼤⽽单调地变好，然而这时候却发生了激烈的震荡，尤其是在区间两端，考虑到附加的噪声，我们可以直观的解释，随着 M 值的增大，多项式可以被更灵活地调参，但是过度调参后反而连⽬标值的随机噪声都拟合了。所以在实际应用中，越复杂的模型未必能够得出更好的结果，但是从以上分析可以看出，可调节的参数数量小于用于训练的数据点个数，结果还都不错，所以，增加训练数据点个数，也可以避免发生这种过拟合的情况。

&emsp;&emsp;可知，如果如果模型过于复杂或参数过于固定，就会把数据的噪声也考虑进去导致过拟合，所以我们解决思路有两种，一种是针对参数，一种是针对模型，过拟合是一个很常见的问题，深度学习中常用 dropout 方法丢弃某些神经单元，其目的是为了使用不同的神经网络，这样可以有效抑制过拟合；此外，高斯过程和贝叶斯神经网络分别通过增加模型的不确定性以及参数的不确定性增强泛化能力；下面将一种很常见的控制过拟合的方法。

#### 正则化 Regularization

&emsp;&emsp;没有免费午餐定理暗示我们必须在特定任务上设计性能良好的机器学习算法。当我们设计的算法特性与希望解决的学习问题相吻合，也就是恰好能适应当前的数据生成分布时，算法性能会更好。上一节多项式回归过拟合问题表明，我们可以通过增加或减少学习算法的可选函数来增加或减少模型，比如增加或减少多项式的次数。算法的效果受影响于选择的的函数数量以及函数的具体形式，即模型结构。在可选空间中，相比于某一个学习算法，我们可能更偏好另一个学习算法。这意味着两个函数都是符合条件的，但是我们更偏好其中一个。
比如我们把加入权重衰减 (weight decay) 作为一种控制过拟合的常见方法，就是给误差函数添加一个惩罚项或正则项 (regularization)，令函数偏好于其偏好于可以使正则项较小的权重。此时误差函数为

$$\tilde{E}(\boldsymbol w)=\frac{1}{2} \sum_{n=1}^{N}\left \{ y(x_n,\boldsymbol w)-t_n \right \}^2 +\frac{\lambda }{2}\sum_{j=1}^{M} |w_j|^q\tag 2$$ 
其中 $\lambda$ 是正则化系数，控制数据误差函数和正则化项的相对重要性和偏好性。对于正则化项的选择⽅法也称为权值衰减 (weight decay)，在梯度下降算法中，它倾向于让权值向零的⽅向衰减。在统计学中， $q=1$ 的情形被称为套索 (lasso)。它的性质为：如果 $\lambda$ 充分⼤，那么某些系数 $w_j$ 会变为零，从⽽产⽣了⼀个稀疏 (sparse) 模型，这个模型中对应的基函数或数据点不起作⽤。根据拉格朗日乘数法 (Lagrange Multipliers)，最小化公式 (2) 等价于在满⾜以下限制条件时最⼩化平⽅和误差函数。

$$\sum_{j=1}^{M} |w_j|^q\leq\eta\tag3$$
 
&emsp;&emsp;参数 $\lambda$ 要选择⼀个合适的值。稀疏性的来源可以从下图中看出来，在限制条件 (3) 下误差函数的最⼩值。书中对这部分解释实在太少，我大致猜想，蓝色表示未正则化的误差函数的范围，由于是二次型所以误差函数轮廓是个圆，其中圆心位置又目标和训练数据的均值决定。$q=1$ 的套索轮廓与误差函数轮廓相切的位置通常在坐标轴上，而 $q=2$ 的圆形轮廓与误差函数的切点在其他位置，因此 $q=1$ 时有更多最优参数会为 0，随着 $\lambda$  的增⼤，越来越多的参数会变为零。也许这也是一种解释，但类似第一篇 1.3 节所提及最大后验估计结果也带有一个正则化项来看，正则化项也是我们加入对参数的先验估计的一种隐性表达，无论 $q$ 取何值，总是希望 $w$ 向 0 的方向移动。
​
<div align=center>
<img src="images/3_1_fit3.png" width="60%"/>
</div>

#### 拉格朗日乘数法 Lagrange Multipliers

&emsp;&emsp;简单解释下拉格朗日乘数法，这是一个被⽤于寻找多元变量在⼀个或者多个限制条件下的驻点的方法。考虑一个 $N$ 维空间变量 ${\boldsymbol x_1,\boldsymbol x_2,...,\boldsymbol x_N}$ ，限制方程 $g(\boldsymbol x)\equiv 0$ 表示空间中一个曲面，那么在限制曲⾯上的任何点处，限制函数的梯度 $\nabla g(\boldsymbol x)$ 都正交于限制曲⾯，通过考虑⼀个位于限制曲⾯上的点 $\boldsymbol x$ 以及这个点附近同样位于曲⾯上的点 $\boldsymbol x+\Delta\boldsymbol x$，对后者进行一阶泰勒展开 $g(\boldsymbol x+\Delta\boldsymbol x)\simeq g(\boldsymbol x)+\Delta\boldsymbol x^T\nabla g(\boldsymbol x)$ ，由于在限制曲面上恒有 $g(\boldsymbol x+\Delta\boldsymbol x)= g(\boldsymbol x)$ 且 $\Delta \boldsymbol x$ 平行于曲面，就能发现 $\nabla g(\boldsymbol x)$ 正交于曲面。而对于 $f(\boldsymbol x)$，我们寻找限制曲⾯上的⼀个点 $\boldsymbol x^{\star}$ 使得 $f(\boldsymbol x)$ 最⼤，那么这个点也一定满足 $\nabla f(\boldsymbol x^\star)$ 正交于曲面，因为如果不满足的话，那我们一定可以沿着曲面短距离移动 $\boldsymbol x$ 使 $f(\boldsymbol x)$ 增加，这与已知条件相悖，因此 $\nabla g(\boldsymbol x)$  与 $\nabla f(\boldsymbol x)$  都正交于曲面且平行，因此存在一个常数 $\lambda$ 使得

$$\nabla f(\boldsymbol x)+\lambda\nabla g(\boldsymbol x)=0\tag4$$
 
&emsp;&emsp;通过积分可得对应的拉格朗日函数为

$$\mathcal L(\boldsymbol x,\lambda)=f(\boldsymbol x)+\lambda g(\boldsymbol x)\tag5$$
 
&emsp;&emsp;所以说最小化限制条件 g(x) 下的 f(x) 等价于最小化上述拉格朗日函数。
最后回到正题，正则化⽅法通过限制模型的复杂度，使得复杂的模型能够在有限⼤⼩的数据集上进⾏训练，⽽不会产⽣严重的过拟合。需要注意的是，正则化通过修改学习算法，旨在降低泛化误差而非训练误差。

#### 局部加权线性回归 Locally Weighted Regression

&emsp;&emsp;下面再介绍一种针对线性回归问题解决过拟合的方法，一种非参数学习方法，叫做局部加权回归 (locally weighted regression)。普通的线性回归属于参数学习算法 (parametric learning algorithm)；而局部加权线性回归属于非参数学习算法 (non-parametric learning algorithm)。参数学习方法在训练完成所有数据后得到一系列训练参数，在预测时用固定参数来测试。而非参数学习在预测新样本值时候每次都会重新训练数据得到新的参数值，每次得到的参数值也是不确定的。 

&emsp;&emsp;之前的线性回归误差函数形式为 $E(\boldsymbol{w})=\frac {1}{2}\sum_{n=1}^{N}[y(x_n,\boldsymbol{w})-t_n]^2$，而在局部加权回归中，误差函数为 $E(\boldsymbol{w})=\frac {1}{2}\sum_{n=1}^{N}v_n[y(x_n,\boldsymbol{w})-t_n]^2$，其中 $v_n$ 是权重，取自高斯采样

$$v_n=\exp (-\frac {(x_n-x)^2}{2\tau^2})\tag 6$$
其中 $x$ 是新预测的样本，参数 $\tau$ 控制权值变化速率，可知如果 $|x_n-x|\approx0$，那么 $v_n\approx1$；如果 $|x_n-x|\approx+\infty$，那么 $v_n\approx0$；所以，离预测样本数据 $x$ 较近的点权值大，离预测样本较远的点权值小。这种做法本质上是为了让线性回归模型不再依赖于整体数据的特征选择，让与预测点更加接近的局部训练数据在重新训练时的损失函数中占据主导地位，实际上这种方法同时有效解决了过拟合和欠拟合的问题，而我们前面提到的样条函数其实也是为了让相近的局部函数在映射到新的特征空间后能够更接近，都是用局部数据预测局部数据，因为它们原始特征更接近，同样的，这种做法必然付出代价。样条函数是我们需要选择大量不同的函数分区拟合，而局部加权线性回归则因为在预测每一个新的数据点时都需要重新训练，因此需要付出巨大的计算量。

#### 交叉验证 Cross Validation

&emsp;&emsp;在我们使⽤最⼩平⽅拟合多项式曲线的例⼦中可以看到，存在⼀个最优的多项式阶数给出最好的结果。多项式的阶数控制了模型的⾃由度，以及模型的复杂度。添加正则项后，正则化系数也控制了我们的模型复杂度。在实际应⽤中，我们需要确定这些参数的值，以期在新数据上能做出最好的预测。此外，我们可能还希望找到适当的可选模型算法，以便找到对于特定应⽤的最好模型。

&emsp;&emsp;此前由于过拟合，模型在训练集上的表现并不能应用于对未知数据的预测。如果数据量很⼤，那么模型选择很简单。我们使⽤⼀部分可得到的数据，可以训练出⼀系列模型的参数值。之后在独⽴数据上⽐较它们，选择预测表现最好的模型即可。因此除了训练集和测试集外，我们又引入验证集 (validation set)。训练过程的超参数总是倾向于过拟合的方向，而测试集通常用来估计训练收敛后最终的泛化误差，在实际中并不能参与到模型选择中，因此从训练数据中构建验证集，将训练数据分成两个不相交的子集，一个用于学习参数，另一个作为验证集，估计训练中的泛化误差，更新超参数。

&emsp;&emsp;将训练数据集分成固定的训练集和验证集后，若验证集的误差很小，可能是有问题的。因为一个小规模的数据集意味着平均测试误差估计的统计不确定性，使得很难判断算法在其他给定的任务上是否做得更好。通常数据集都是有限的。为了建⽴更好的模型，我们想使⽤尽可能多的可得到的数据进⾏训练，使用所有的样本估计平均测试误差。⼀种解决⽅法是使⽤交叉验证 (cross validation)，这种⽅法能够让可得到数据总量 K 的 K−1K ⽤于训练，同时使⽤所有的数据来评估表现。

&emsp;&emsp;交叉验证法可以描述为：随机将数据集 $S$ 平均划分为 $K$ 个不相交的子集 $S_1,\dots,S_K$ ，对于可选择的 $M$ 个模型 $\left \{ M_i \right \}$，将每个模型依次在 $K−1$ 个子集上训练，在剩余的一个子集 $S_k$ 上验证得到误差 $E(M_{ik})$，共进行 $K$ 次，使每个子集都能都作为一次验证集，将 $K$ 个误差作均值得到 $E(M_i)$，然后选择具有最小估计误差的模型 $M_i$，然后在整个训练集上重新训练，得出的结果即为最终模型。当数据相当稀疏的时候，考虑 $K=N$ 的情况很合适，其中 $N$ 是数据点的总数，这种技术叫做 “留⼀法” (leave-one-out)。
如下图所示的交叉验证⽅法，其中 $K=4$，然后，$K−1$ 组数据被⽤于训练⼀组模型，然后在剩余的⼀组上进⾏评估，图中用红色标出，之后，对 $K$ 轮运⾏结果的误差求均值。
​
<div align=center>
<img src="images/3_2_feature1.png" width="35%"/>
</div>