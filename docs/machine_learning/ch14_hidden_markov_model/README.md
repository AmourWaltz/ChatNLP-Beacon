# 14 隐马尔科夫模型 Hidden Markov Model

&emsp;&emsp;在前面的篇章中，我们更多考虑的是假设数据集⾥的数据独⽴同分布的情形，根据这个假设，似然函数可以表示为所有数据点的概率分布的乘积。然⽽，在实际应用中，独⽴同分布的假设可能不成⽴，典型的就是顺序数据，这些数据通常是时间序列变量，比如语音信号就是在连续时间框架下的声学特征，本文将从语⾳识别任务切入，引出隐马尔科夫模型 (Hidden Markov model)。

&emsp;&emsp;前文提过马尔科夫链模型 (Markov Chain Model)，使⽤概率乘积规则表示一段观测序列 $\left \{ \boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_N \right \}$ 的联合概率分布，即

$$p(\boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_N)=\prod_{n=1}^{N}p(\boldsymbol w_n|\boldsymbol w_{n-1},\dots,\boldsymbol w_1)\tag 1 $$

&emsp;&emsp;我们假设了公式 (2) 中任意 $\boldsymbol w_n$ 的取值只与最近⼀次的观测 $\boldsymbol w_{n-1}$ 有关，⽽独⽴于其他所有之前的观测，那么就得到了⼀阶马尔科夫链 (first-order Markov chain)，观测序列的联合概率分布为

$$p(\boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_N)=\prod_{n=1}^{N}p(\boldsymbol w_n|\boldsymbol w_{n-1})\tag2$$

&emsp;&emsp;n 元语言模型 (n-gram language model) 就是一个 n 阶马尔科夫链，符合这个假设的随机过程被称为马尔科夫过程。我们用 n 元语言模型很容易统计出训练文本中在已知 n-1 个词依次出现时第 n 个词出现的频率，将这个频率当做概率，n 元语言模型就转换为概率模型，继而可用于对结果进行概率评估，即打分(score)，然后排序选择出最有可能的识别结果。一般情况下使用四元语言模型即可，因为 n 元语言模型的存储空间是随着 n 呈指数增长的。
然而我们还需要一个更合适的模型来对公式 (1) 中的声学模型建模，这就需要用到隐马尔科夫模型 (Hidden Markov Model, HMM)。

## 14.1 隐马尔科夫模型 Hidden Markov Model

&emsp;&emsp;隐马尔科夫模型是马尔科夫模型的一个扩展，是最常见的计算 $P(\boldsymbol x_1,\boldsymbol x_2,\dots,\boldsymbol x_T|\boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_N)$ 的统计模型。声学模型包含两种不同且长度不一的信号，我们求解条件概率 $P(\boldsymbol x_1,\boldsymbol x_2,\dots,\boldsymbol x_T|\boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_N)$，其中声学信号 $\boldsymbol x_1,\boldsymbol x_2,\dots,\boldsymbol x_T$ 是已知可观测到的特征向量，因此我们称 $\boldsymbol x$ 为 HMM 的观测值 (observation)，然后再将文本序列变作与观测序列等长的潜在特定状态序列 (specified state sequence) $\left \{ \boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_N \right \}$；最后定义一个与文本序列等长的包含 $N$ 个隐藏状态 (hidden state) 的序列，隐藏状态数与观测值数目彼此不受约束，这就解决了声学和文本序列的起始对齐问题。我们无法观测到任意的隐藏状态 n ，因此无法通过状态序列 $\boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_N$ 推测条件概率，隐马尔科夫模型把这个从隐藏状态转移到观测值的概率称作转移概率 (transition propability)。这里也需要引入一些假设：

&emsp;&emsp;首先假设从一个隐藏状态转移到另一个状态只和当前时刻的状态有关，这就是马尔科夫假设 (Markov assumption)。其中转移概率是恒定的，并且与观测值和前一时刻的隐藏状态不相关；
其次还应假设观测值是相互独立的，这样当前观测就不会受前面的观测值影响，这被称为状态条件独立性假设 (state conditional independence assumption)。前两条假设只是为了简化模型，与实际语音信号属性相去甚远，因为一段正常语音是高度连续的，某一帧的发音很容易受前后字词转换所影响；
最后，我们对连续的声学信号做了离散化处理，由观测值，即声学特征向量表示，这里的离散化采样是以帧为单位提取，每一帧内的声学信号看做是静止 (stationary) 的，通常每一帧都会采样 25 ms 的数据，所以有很多声音信息未能很好的提取。
尽管有诸多限制，时至今日，HMM 仍是处理声学模型的最优方法。HMM 是一个有限状态机 (finite state machine)，可作如下描述：一个 HMM 包含 N 个隐藏状态，其中有 $N-2$ 个发射状态 (emitting states)，一个起始状态 (entry state) 和一个结束状态 (exit state)，还应注意执行完这个有限状态机的总时间步长为观测序列长度 T ；在时间步 t ，从当前隐藏状态 $\boldsymbol h=i$ 及其对应的特定状态 $\boldsymbol w_t$ 转移到下一状态 $\boldsymbol h=j$ 和 $\boldsymbol w_{t+1}$ 的概率为 $a_{ij}=a_{\boldsymbol w_t,\boldsymbol w_{t+1}}=P(\boldsymbol h=j|\boldsymbol h=i)$，同时也会有 $a_{ii}=a_{\boldsymbol w_t,\boldsymbol w_{t+1}}$ 的概率继续停留在当前状态；而观测值 $\boldsymbol x_t$ 也就以发射概率 $b_{\boldsymbol w_t}(\boldsymbol x_t)=b_j(\boldsymbol x_t)=P(\boldsymbol x_t|\boldsymbol h=i)$ 随之产生。一个隐马尔科夫模型，$\boldsymbol\lambda$，就可以表示为参数集合 $\boldsymbol\lambda=\left \{ N,\left \{ a_{ij} \right \},\left \{ b_j(\cdot) \right \} \right \}$。这里的 HMM 是一个生成模型，用来产生观测值，也就是语音信号 \boldsymbol X 的最大似然概率。由于语音信号都是时序信号，所以 HMM 是一个从左至右 (left-to-right) 的单向结构，下图给出了一个 HMM 的实例。

<div align=center>
<img src="images/14_1_mk2.png" width="60%"/>
</div>

&emsp;&emsp;上图是一个 HMM 图例，大圈状态 1 和 5 分别是起始状态和结束状态，状态 2，3，4 是发射状态，表示隐藏状态，这里加入起始和结束状态是为了标示一次完整的 HMM 过程。声学模型研究的是给定某个文本信号 $\boldsymbol w$ 产生语音 $\boldsymbol x$ 的概率，而输入是语音信号，正好作为 HMM 中的观测数据 (observed data)，对应上图中的黄色圆圈 1-7，整个 HMM 观测值和隐藏状态的联合概率为

$$p(\boldsymbol X, \boldsymbol W|\boldsymbol\lambda)=a_{\boldsymbol h_0,\boldsymbol h_{1}}\prod_{t=1}^{T}b_{\boldsymbol h_t}(\boldsymbol x_t)a_{\boldsymbol h_t,\boldsymbol h_{t+1}}\tag3$$
其中 $\boldsymbol h_0$ 和 $\boldsymbol h_{T+1}$ 分别代表起始状态 1 和结束状态 5。上述图例中的 HMM 的联合概率可表示为

$$p(\boldsymbol X, \boldsymbol W|\boldsymbol\lambda)=a_{12}b_2(\boldsymbol x_1)a_{22}b_2(\boldsymbol x_2)a_{22}b_2(\boldsymbol x_3)a_{23}b_3(\boldsymbol x_4)a_{33}b_3(\boldsymbol x_5)a_{34}b_4(\boldsymbol x_6)a_{44}b_4(\boldsymbol x_7)a_{45}\tag4$$

&emsp;&emsp;与直观的认知不同，声学模型的 HMM 输入是被隐藏状态产生的观测值，参数是转移概率和发射概率，输出才是隐藏状态，这是因为我们一开始就使用了贝叶斯公式的缘故，可以理解为，我们在获得一段语音数据作为观测值后，反推出究竟是什么样的一段文本才能匹配这段语音，所以 HMM 的观测值才是重点关注的对象。

&emsp;&emsp;HMM 的输出分布应尽可能接近从一个隐藏状态产生对应观测值的真实分布 (actual distribution of the data associated with a state)，并且尽可能在数学上可计算 (be mathematically and computationally tractable)，另外每一步发射概率 b 通常用概率密度表示，一般选择高斯分布，即

$$b_j(\boldsymbol w)=\mathcal N(\boldsymbol \mu_j,\boldsymbol \Sigma_j)\tag5$$

&emsp;&emsp;总结一下，HMM 用来对序列进行建模，从一个观测序列，推出对应的状态序列，也就是“由果找因”。这里的“因”一般是隐藏的，无法简单的看出来，所以叫隐藏状态。HMM 涉及的主要内容有，两组序列（隐藏状态和观测值），三种概率（初始状态概率，状态转移概率，发射概率）。我们需要解决三个基本问题：

    1. 在已知模型参数和一段观测序列样本的前提下，产生出这段观测序列的概率是多少；
    2. 同时也想知道每个观测值对应的最优隐藏状态序列；
    3. 最后是如何训练出模型最优参数。

&emsp;&emsp;先说观测序列 $\boldsymbol X=\left [ \boldsymbol x_1, \boldsymbol x_2,\dots, \boldsymbol x_T\right ]$ 的产生问题，与上图所表示的 HMM 不同，实际应用中，隐藏状态是完全不可见的，好在语音识别系统中，词表数量 $N$ 是提前定义好的，即每一个隐藏状态有多少种取值是已知的，所以大不了就用一种很暴力但直观的方法，对所有可能的隐藏状态序列产生出这段观测序列的概率进行加权求和。这里我们可以直接认为隐藏状态序列 $\boldsymbol W=[\boldsymbol w_1,\boldsymbol w_2,\dots,\boldsymbol w_T]$ 与观测序列等长，因为已假设每一个隐藏状态可以取任意单词，那么闭环停留到当前状态的过程也可展开为前向过程。然而这种方法计算量会很大，如果隐藏状态有 N 种取值可能，观测序列长度为 T ，那么计算复杂度为 $TN^T$，为了降低复杂度，很容易想到用空间换时间的思路，只要把某一种状态序列在某一处观测值的概率记录下来，下次处于同样的位置就不用重新计算，然后使用一些递归算法就可以计算，这里有前向和后向两种计算方式。简单说一下前向计算过程 (forward algorithm)， 已知参数为 \boldsymbol \lambda 的 HMM 模型，将 t 时刻隐藏状态为 $\boldsymbol w_{t,i}$ 得到观测序列 $\boldsymbol x_1, \boldsymbol x_2,\dots, \boldsymbol x_t$ 的概率记为 $\alpha_{t}(i)$，设初始状态的转移概率为 $\pi_i$，其中 $1<i<N$，令

$$\alpha_{1}(i)=\pi_ib_i(\boldsymbol x_1)\tag6$$

&emsp;&emsp;注意这里将参数 $\alpha$ 的下标改为与观测相同的时间步，因为语音识别的 HMM 我们已经假设某时刻的隐藏状态只与当前时刻的观测有关，而参数 $\alpha$ 括号内的 $i$ 表示所有可能的隐藏状态，通过第一步，递归的初始值就得到了，然后使用递归算法 (recursive algorithms)

$$\alpha_{t+1}(j)=\left [ \sum_{i=1}^N\alpha_t(i)a_{ij} \right ]b_j(\boldsymbol x_{t+1})=p(\boldsymbol x_{t+1}|\boldsymbol w_{t+1,j},\boldsymbol \lambda)p(\boldsymbol w_{t+1,j}|\boldsymbol \lambda)\tag7$$

&emsp;&emsp;这里状态转移概率 $a_{ij}$ 是从隐藏状态 $\boldsymbol w_{t,i}$ 转移至 $\boldsymbol w_{t+1, j}$ 的概率，其中 $1\leq t\leq T-1,1\leq j\leq N$，最终可得

$$P(\boldsymbol X|\boldsymbol \lambda)=\sum_{i=1}^T\alpha_T(i)\tag8$$

&emsp;&emsp;当所有观测状态输出完毕，对所有 $\alpha_T(i)$ 求和，共需计算 $T$ 步，每一步进行 $N\times N$ 次乘法运算，因此计算复杂度为 $N^2T$。

&emsp;&emsp;通过上述步骤，可以依次得到每个时刻的 $N$ 个状态的概率值，并最终得到已知 HMM $\boldsymbol \lambda$ 时产生观测序列的概率 $p(\boldsymbol X|\boldsymbol \lambda)$。另一种后向算法 (backward algorithm) 和前向算法类似，顾名思义区别是后向算法从最后一个观测值出现的概率反推得到整个观测序列的出现概率；在此不做赘述。

&emsp;&emsp;然后接着讨论第二个基本问题，即如何找到最优状态序列 (best state sequence)，这里的最优状态序列并不是指对一个未知单词的语音片段找到其对应似然概率最大的单词再组成状态文本序列，而是直接寻找似然概率最大的状态序列 (most likely state sequence)，两者区别在于前者仍是基于孤立词识别系统出发，这意味着我们要事先对语音进行切片划分，但是实际上我们更多处理的是连续语音 (continuous speech)。针对这个问题，我们可以使用维特比算法。

## 14.2 维特比算法 Viterbi Algorithm

&emsp;&emsp;维特比算法用来解决在给定观测序列下，找到最优的隐藏状态序列。首先引入一种网格 (lattice, trellis) 的形式表示 HMM。每一个时间步 t ，隐藏状态都有 N 种取值可能，并且任意取值都能转移到下一时刻的任意状态取值。假设一个有 4 个时间步，每个隐藏状态有 5 种取值，即一个观测序列长度为 4 词汇表大小为 5 的语音识别 HMM，可用下图直观表示

<div align=center>
<img src="images/14_2_mk1.png" width="60%"/>
</div>

&emsp;&emsp;上图可以看出我们把所有可能的状态序列表示出后其网络形状确实像网格 (trellis) 或篱笆 (lattice)。似然概率最大的序列就是网格中的最优完整路径 (best complete path)，可以假设是上图中的红色路。所有的状态序列或路径都可以根据路径转移概率，发射概率等乘积计算，概率最大的路径也就是最短路径，说起最短路径问题，相信大部分人都会第一时间想起动态规划 (dynamic algorithm) 算法，维特比算法就是是一种动态规划算法，解决 HMM 图中的最短路径问题，找到一种从起始到结束概率最大的隐藏状态序列。而维特比算法也是针对这种特殊有向图 (lattice) 提出的。

&emsp;&emsp;简单说下为什么引入维特比算法，其实上述举例可以看出，既然想找到最短路径，那么把所有路径枚举出来，然后找出最优的那条不就易如反掌吗？然而在实际应用中，对于词汇表大小 $N$ 通常都数以万计，假设 $N=60,000$，观测序列长度 $T=10$，那么可能的路径共有 $N^T=6\times10^{47}$，这是无法计算的。同样地，我们又回到空间换时间的思路，只是前向算法时为了得到状态序列的期望概率，这里考虑的是最短路径。下面讨论维特比算法如何在给定 HMM $\boldsymbol \lambda$ 和长度为 $T$ 的观测序列 $\boldsymbol X$ 的前提下找出最优隐藏状态序列路径 $\boldsymbol W^\star$。

$$\hat P(\boldsymbol X|\boldsymbol \lambda)=\max_{\boldsymbol W}\left \{ P(\boldsymbol X,\boldsymbol W|\boldsymbol \lambda) \right \}\tag {9}$$

&emsp;&emsp;维特比算法只考虑任意时间步的最短路径 (a simple recursion only takes into account the most likely state sequence through the model at any point)，假设在时刻 $t$ 进入状态 $\boldsymbol w_{t,i}$ 的概率为 $P(\boldsymbol x_1,\dots,\boldsymbol x_t,\boldsymbol w_{t,i}|\boldsymbol \lambda)$，在动态规划算法中，整个有向图的最短路径在时刻 $t$ 经过状态 $\boldsymbol w_{t,i}$，那么这条路径从起始状态到 $\boldsymbol w_{t,i}$ 的子路径，一定是从起始状态到 $\boldsymbol w_{t,i}$ 的最短路径，利用这个思想不断地做递归，就可以以较低的计算复杂度得到最短路径，设 $t$ 时刻进入状态 $\boldsymbol w_{t,j}$ 的最短路径

$$\delta_j(t)=\max_{\boldsymbol X^{(t-1)}} P(\boldsymbol x_1,\dots,\boldsymbol x_t,\boldsymbol w_{t,j}|\boldsymbol \lambda)\tag{10}$$
其中 $\boldsymbol X^{(t-1)}$ 是到 $t-1$ 时刻的所有子路径，同样引入递归的思路计算整个序列的最短路径，递归公式为

$$\delta_{j}(t)=\max_i\left \{ \left [ \delta_i(t-1)a_{ij} \right ]b_j(\boldsymbol x_{t})\right \} \tag{11}$$

&emsp;&emsp;由于 $\delta_i(t-1)$ 已经是状态序列 $\boldsymbol w_{1},\dots,\boldsymbol w_{t-1}$ 的最短路径，我们只需考虑从这条路径出发在 $t$ 转移到所有 $N$ 个状态的概率，就能得到 $t$ 时刻 $\boldsymbol w_{1},\dots,\boldsymbol w_{t}$ 的最短路径。可以发现，为了得到完整路径，还需要一个 $\theta_t(j)$ 记录得到 $\delta_{t+1}(j)$ 的最优状态，这就是空间换时间的关键 (store the local decisions made at each point in the Viterbi trellis and the traceback along the most likely path at the end of utterance)。初始化 $\delta_i(1)=\pi_ib_i(\boldsymbol x_1)$，记录 $\theta_i(1)=0$；然后调用递归公式 (12)，记录每一步的 $\theta_{j}(t+1)=\arg \max_i\left [ \delta_i(t)a_{ij} \right ]$，这一步称为回溯 (traceback)；最后就能得到最短路径 $\hat P(\boldsymbol X|\boldsymbol \lambda)=\max_i\left [ \delta_i(T) \right ]$。

&emsp;&emsp;维特比算法的核心就是在进行到 $t$ 时刻时，我们只记录到 $t$ 时刻所有 $N$ 个状态的最短路径，因此，无论最终路径是哪一条，我们所记录的 $t$ 时刻的路径中总有一条是其子路径。从复杂度的角度分析，在进行第一步 $t=1$ 时，我们计算起始状态到各个状态的距离，由于只要一步，所有这些路径就是到每个状态的最短路径，进行了 $N$ 次计算；然后第二步 $t=2$，我们计算从起始状态到第二步状态的最短路径，对于第二步任意状态 $i$ 的最短路径，一定会经过第一步所记录的最短路径，由此只需要进行 $N^2$ 次计算，以后每一步时刻 $t$ 计算最短路径都只需要考虑 $t$ 和 $t-1$ 时刻的状态，需要 $N^2$ 次计算，最终计算完所有 $T$ 个观测值，只需要 $T\cdot N^2$ 次计算，搜索时间也就随时间步长 $T$ 线性增长。

&emsp;&emsp;如果不断进行概率乘积，在运算时会导致数值下溢 (the direct calculation of likelihood will cause arithmetic underflow)，因此实际操作时，我们会用将其转换为对数函数的形式

$$\log \delta_j(t)=\max_{1\leq k\leq N}\left [ \log (\delta_k(t-1))+\log (a_{kj}) \right ]+\log (b_j(\boldsymbol x_t))\tag{12}$$

&emsp;&emsp;维特比算法是一种非常高效的算法，在语音识别中，输入都是按照从左至右的流的方式进行的，只要处理每个状态的时间不比讲话慢，不论输入多长，语音识别永远都是实时的。当然这一切还是建立在 HMM 的假设基础上，HMM 和维特比算法也就构成了语音识别的基础框架。

