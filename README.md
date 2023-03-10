# ML&NLP Beacons
</br>

&emsp;&emsp;本项目是针对机器学习和自然语言处理所做的学习笔记和专栏文章。笔者是一名在读博士生，研究方向为预训练模型和对话系统，平时热衷写作分享，在接触 Datawhale 后萌生了将学习笔记整理为开源项目的想法，期望能找到更多志同道合的伙伴交流进步。机器学习部分包括“圣经”《Pattern Recognition and Machine Learning》和深度学习花书《Deep Learning》的学习笔记，整理了书中精华以及对科研更重要的内容作深入理解，而自然语言处理则是子方向的小综述，帮助其他方向的同学快速了解。

&emsp;&emsp;在科研过程中，面对浩渺如烟的知识云海，我深感“吾生而有涯，而知也无涯，以有涯随无涯，殆矣”，将项目取名为 ML&NLP Beacons，旨在从茫茫学海中过滤出最精华最实用的部分，如灯塔般指引学海中的一叶扁舟，以期让有限的研究生学习事半功倍，由浅入深，快速了解上手后再图精益求精，不失为一条高效的学习路径。

&emsp;&emsp;欢迎对本项目以及机器学习和自然语言处理感兴趣的同学加入，可以对现有章节进行斧正交流，也可以编写自己感兴趣的内容，欢迎联系微信：NightWalzer。

&emsp;&emsp;(注：该项目原是笔者在知乎写作 Bishop《Pattern Recognition and Machine Learning》的笔记专栏，接触 Datawhale 后尝试扩展的开源项目，但不久前恰好看到一份 PRML 笔记教程的开源资料，原来微博早有人在 2015 年就组织了 PRML 的读书会，做了和笔者想法相同的学习笔记，并在不久前竣工后将相关中文译本，官方代码，课程视频，笔记教程悉数发布，[链接在此](https://mp.weixin.qq.com/s/NQRU_y9SaRXlB53zvgeGjg)。笔者阅读后大受震撼，确实是非常优秀实用的学习资料，完整详细，深入浅出，非常感谢贡献者们的杰出工作，于我而言再重复前人工作不过画蛇添足，因此将原项目更新。)

## 参考目录

<!-- * **<font size=4>前言 Preface</font>**
    * [项目初衷 Project Intention](./preface/intention.md) -->

* **<font size=4>机器学习 Machine Learning</font>**
    * [1 线性模型 Linear Regression](./linear_model/README.md)
        * [1.1 线性基函数模型 Linear Basis Function Model](./linear_model/README.md)
        * [1.2 线性回归模型 Maximum Likelihoood Estimation](./linear_model/README.md)
            * [最大似然估计，最小均方差，解析法](./linear_model/README.md)
        * [1.3 线性分类模型 Linear Discriminate Analysis](./linear_model/README.md)
            * [Fisher 判别器，线性判别模型，感知器算法](./linear_model/README.md)
        * [1.4 判别式与生成式模型 Discrminate and Generative Model](./linear_model/README.md)
        * [1.5 广义线性模型 Generalized Linear Model](./linear_model/README.md)
    * [2 概率分布 Probability Distribution](./probability_distribution/README.md)
        * [2.1 贝叶斯概率 Bayes Probability](./probability_distribution/README.md)
        * [2.2 Beta分布 Beta Distribution](./probability_distribution/README.md)
        * [2.3 狄利克雷分布 Dirichlet Distribution](./probability_distribution/README.md)
        * [2.4 高斯分布 Gaussian Distribution](./probability_distribution/README.md)
    * [3 核方法 Kernel Method](./kernel_method/README.md)
        * [3.1 核函数 Kernel Function](./kernel_method/README.md)
        * [3.2 高斯过程 Gaussian Distribution](./kernel_method/README.md)
        * [3.3 支持向量机 Support Vector Machine](./kernel_method/README.md)
    * [4 期望最大化 Expectation Maximum](./expectation_maximum/README.md)
        * [4.1 K均值聚类 K-means Clustering](./expectation_maximum/README.md)
        * [4.2 EM算法 EM Algorithm](./expectation_maximum/README.md)
    * [5 变分法 Variational Inference](./variational_inference/README.md)
        * [5.1 变分法的物理推导 Inference of Variational Method](./variational_inference/README.md)
        * [5.2 变分近似推断 Variational Approximation](./variational_inference/README.md)
    * [6 采样方法 Sampling Method](./monte_carlo_sampling/README.md)
        * [6.1 蒙特卡罗采样 Monte Carlo Sampling](./monte_carlo_sampling/README.md)
  
* **<font size=4>深度学习 Deep Learning</font>**
    * [1 学习理论 Learning Theory](./learning_theory/README.md)
        * [1.1 过拟合与正则化 Overfitting](./learning_theory/README.md)
        * [1.2 模型特征选择 Model Feature Selection](./learning_theory/README.md)
        * [1.3 偏差与方差 Bias and Variance](./learning_theory/README.md)
    * [2 信息论 Information Theory](./information_theory/README.md)
      * [2.1 信息熵 Information Entropy](./information_theory/README.md)
      * [2.2 熵的物理意义 Entropy in Physics](./information_theory/README.md)
      * [2.3 相对熵和互信息 Relative Entropy and Mutual Information](./information_theory/README.md)
    * [3 神经网络 Neural Networks](./neural_networks/README.md)
      * [3.1 深度前馈神经网络 Deep Feed-forward Neural Networks](./neural_networks/README.md)
      * [3.2 隐藏单元 Hidden Unit](./neural_networks/README.md)
      * [3.3 反向传播 Back Propagation](./neural_networks/README.md)
    * [4 模型优化 Model Optimization](./optimization/README.md)
        * [4.1 梯度下降法 Gradient Desent](./optimization/README.md) </br>
    * [5 大规模深度学习 Large-scale Deep Learning](./large_scale_dl/README.md)

* **<font size=4>自然语言处理 Natural Language Processing</font>**
    * [1 预训练模型 Pretrained Models](./pretrained_model/README.md)
    * [2 对话系统 Dialogue Systems](./dialogue_system/README.md)
        * [2.1 基于知识的对话大模型](./dialogue_system/README.md)
        * [2.2 对话大模型中的事实错误](./dialogue_system/README.md)
    * [3 对话大模型实现 Dialogue Models](./dialogue_model/README.md)
        * [3.1 基于 GPT2 搭建对话生成模型](./dialogue_model/README.md)
        * [3.2 基于 BERT 的对话生成评估](./dialogue_model/README.md)

## 参考资料

</br>

<div align=center>
<img src="images/mldl.png" width="55%"/>
</div>

> **① Pattern Recognition and Machine Learning &emsp; 作者：Christopher M. Bishop** <br />
> **② Deep Learning &emsp; 作者：Ian Goodfellow, Yoshua Bengio, Aaron Courville**   <br />

<!-- ## 主要贡献者
[@薛博阳-Nocturne](https://github.com/Relph1119)   -->

## 贡献名单

</br>

<table align="center" style="width:90%;">
<thead>
  <tr>
    <th>成员</th>
    <th>简介</th>
    <th>主页</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><span style="font-weight:normal;font-style:normal;text-decoration:none"><a href="https://amourwaltz.github.io">薛博阳</a></span></td>
    <td><span style="font-weight:normal;font-style:normal;text-decoration:none">项目负责人，香港中文大学博士在读 </span></td>
    <td><span style="font-weight:normal;font-style:normal;text-decoration:none"><a href="https://github.com/AmourWaltz">Github</a>, <a href="https://www.zhihu.com/people/yi-ran-chao-shi-dai">知乎</a> </span></td>
  </tr>
</tbody>
</table>

&emsp;&emsp;由于项目规划工程量较大，时间线预计较久，我平日精力能力有限，非常期望能遇到对此项目以及 PRML, DL, NLP 感兴趣或者正在学习的伙伴，希望你能参与部分章节的编辑，或对完成的内容进行检查斧正，或对项目提出合理建议，有意者欢迎联系。

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="images/qrcode.jpeg" width = "180" height = "180">
</div>

&emsp;&emsp;Datawhale，一个专注于AI领域的学习圈子。初衷是 for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。