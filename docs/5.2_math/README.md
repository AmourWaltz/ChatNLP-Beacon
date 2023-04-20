# 数学推导 Mathematical Derivation

### 最大熵的证明过程

> 证明：当所有 $p(x_i)$ 都相等且 $p(x_i)=\frac{1}{M}$ 时，熵 $H(x)$ 取得最⼤值，其中 $M$ 是状态 $x_i$ 的总数，此时对应熵值为 $H=\ln M$。

&emsp;&emsp;根据 Jensen 不等式，对于凸函数 $f(x)$ 有

$$f\left (\sum_{i=1}^M\lambda_ix_i\right )\leq\sum_{i=1}^M\lambda_if(x_i)\tag1$$

> 注：公式 (1) 的推导可参见下方 **Jensen 不等式证明凸函数公式**。
 
其中 $\sum_{i=1}^M \lambda_i = 1$，如果把 $\lambda_i$ 看成离散变量 $x = x_i$ 时的概率分布，那么公式 (1) 就可记作

$$f(\mathbb E[x])\leq\mathbb E[f(x)]\tag{2}$$

&emsp;&emsp;更一般地，我们推广到连续变量，公式 (2) 就变为

$$f\left ( \int xp(x)\mathrm dx \right )\leq \int f(x)p(x)\mathrm dx\tag{3}$$

&emsp;&emsp;再回到公式 $H[p]=-\sum_{i}p(x_i)\ln p(x_i)$，此时的 $f(x)=\ln (x)$ ，而 $\ln (x)$ 为凹函数，只需将公式 (1) 不等号反转即可，即

$$H(x)=-\sum_{i=1}^Mp(x_i)\ln p(x_i)=\sum_{i=1}^M p(x_i)\ln \frac{1}{p(x_i)} \leq\ln \left ( \sum_{i=1}^M p(x_i)\frac{1}{p(x_i)}\right ) =\ln M \tag{4}$$

由于 $\ln \left ( \sum_{i=1}^M p(x_i)\right )=0$，可得 $H=\ln M$ 时，$p(x_i)=\frac{1}{M}$。

### Jensen 不等式证明凸函数公式

&emsp;&emsp;Jensen 不等式证明是凸函数 (convex function) 的一般公式：一个函数 $f(x)$ 在 $x=a$ 到 $x=b$ 之间任意点 $x$ 都可以写成 $\lambda a+(1-\lambda)b$ 的形式，其中 $0\leq \lambda \leq 1$，那么函数图像 $(a,f(a))$ 和 $(b,f(b))$ 之间的连线上对应的点就可以写成 $\lambda f(a)+(1-\lambda)f(b)$，而函数 $f(x)$ 上的对应值为 $f(\lambda a+(1-\lambda)b)$。如果 $f(x)$ 为凸函数，就具有性质

$$f(\lambda a+(1-\lambda)b)\leq\lambda f(a)+(1-\lambda)f(b)\tag5$$

&emsp;&emsp;这也等价于要求函数的⼆阶导数处处为正，特别地，如果仅在 $\lambda=0$ 和 $\lambda =1$ 处取等号，就称函数 $f(x)$ 为严格凸函数。反之，如果函数具有相反的性质，我们就称其为凹函数 (concave function)。


$$f\left (\sum_{i=1}^M\lambda_ix_i\right )\leq\sum_{i=1}^M\lambda_if(x_i)\tag6$$

&emsp;&emsp;我们可以使用数学归纳法来证明。假设 $M=1$ ，根据公式 (6) 很容易证明，然后令 $M=M+1$，证明

$$\begin{align} f\left (\sum_{i=1}^{M+1}\lambda_ix_i \right)&=f\left (\lambda_{M+1}x_{M+1}+\sum_{i=1}^M\lambda_ix_i\right )\\ &=f\left (\lambda_{M+1}x_{M+1}+(1-\lambda_{M+1})\sum_{i=1}^M\frac{\lambda_i}{1-\lambda_{M+1}}x_i\right )\\ &\leq\lambda_{M+1}f(x_{M+1})+(1-\lambda_{M+1})\sum_{i=1}^M\frac{\lambda_i}{1-\lambda_{M+1}}f(x_i)\\ &= \sum_{i=1}^{M+1}\lambda_if(x_i) \end{align}\tag{7}$$
其中从第二步到第三步就应用了公式 (5)。因为已知 $M=1$ 的情况成立，应用公式 (7) 很容易得出公式 (6)。

#### 高斯分布的熵值计算

&emsp;&emsp;对于给定的协⽅差，具有最⼤熵的多元概率分布是⾼斯分布。概率分布 $p(\boldsymbol x)$ 的熵为

$$H[\boldsymbol x]=-\int p(\boldsymbol x)\ln p(\boldsymbol x)\mathrm d\boldsymbol x\tag{8}$$

&emsp;&emsp;针对 $p(\boldsymbol x)$ 最大化 $H[\boldsymbol x]$，其中 $p(\boldsymbol x)$ 满足：1. 可归⼀化；2. 具有均值 $\boldsymbol \mu$；3. 具有协⽅差 $\boldsymbol \Sigma$。

&emsp;&emsp;我们使用拉格朗日乘数法来引入限制条件，需要注意我们需要三个不同的拉格朗日乘数器，首先引入一个单变量 $\lambda$ 针对归一化条件，然后是一个 $D$ 维向量 $\boldsymbol m$ 针对均值条件，以及一个 $D\times D$ 维的矩阵 $\boldsymbol L$ 针对协方差限制，这样我们有

$$\begin{align} \tilde{\boldsymbol H}[p]=&-\int p(\boldsymbol x)\ln p(\boldsymbol x)\mathrm d\boldsymbol x+\lambda\left (\int p(\boldsymbol x)\mathrm d\boldsymbol x-1\right )+\boldsymbol m^T\left (\int p(\boldsymbol x)\boldsymbol x\mathrm d\boldsymbol x - \boldsymbol \mu\right )\\&+\mathrm {Tr}\left \{ \left (\boldsymbol L\int p(\boldsymbol x)(\boldsymbol x-\boldsymbol \mu)(\boldsymbol x-\boldsymbol \mu)^T\mathrm d\boldsymbol x-\boldsymbol \Sigma\right )\right \} \end{align}\tag{9}$$

&emsp;&emsp;将公式 (9) 中所有积分项写在一起，定义被积函数 $F(\boldsymbol x) 为p(\boldsymbol x)\ln p(\boldsymbol x)+\lambda\left ( p(\boldsymbol x)-1\right )+\boldsymbol m^T\left (p(\boldsymbol x)\boldsymbol x - \boldsymbol \mu\right )+\mathrm {Tr} \left \{\boldsymbol L\left (p(\boldsymbol x)(\boldsymbol x-\boldsymbol \mu)(\boldsymbol x-\boldsymbol \mu)^T-\boldsymbol \Sigma\right )\right \}$，该函数被称为 $p(\boldsymbol x)$ 的泛函数，根据泛函数的导数定义，我们令 $\frac{\partial F(\boldsymbol x)}{\partial p(\boldsymbol x)} = 0$，可得

$$\frac{\partial F(\boldsymbol x)}{\partial p(\boldsymbol x)} = -1-\ln p(\boldsymbol x)+\lambda+\boldsymbol m^T\boldsymbol x+\mathrm{Tr}\left \{ \boldsymbol L(\boldsymbol x-\boldsymbol \mu)(\boldsymbol x - \boldsymbol \mu)^T \right \}\tag{10}$$
$$\begin{align} p(\boldsymbol x)&=\exp\left \{ \lambda-1 + \boldsymbol m^T\boldsymbol x+(\boldsymbol x-\boldsymbol \mu)^T\boldsymbol L(\boldsymbol x - \boldsymbol \mu) \right \}\\ &=\exp\left \{ \lambda-1 +\left(\boldsymbol x-\boldsymbol \mu+\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m\right )^T\boldsymbol L\left (\boldsymbol x - \boldsymbol \mu+\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m\right )+ \boldsymbol \mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m\right \} \end{align}\tag{11}$$

&emsp;&emsp;令 $\boldsymbol y=\boldsymbol x-\boldsymbol \mu+\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m$，然后代入归一化和均值的约束条件，可得以下两个公式

$$\int \exp\left \{ \lambda-1+\boldsymbol y^T\boldsymbol L\boldsymbol y+\boldsymbol\mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m \right \}\left ( \boldsymbol y+\boldsymbol \mu-\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m \right )\mathrm d\boldsymbol y=\boldsymbol \mu\tag{12}$$ 
$$\int \exp\left \{ \lambda-1+\boldsymbol y^T\boldsymbol L\boldsymbol y+\boldsymbol\mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m \right \}\mathrm d\boldsymbol y=1\tag{13}$$
其中 $\exp\left \{ \lambda-1+\boldsymbol y^T\boldsymbol L\boldsymbol y+\boldsymbol\mu^T\boldsymbol m-\frac{1}{4}\boldsymbol m^T\boldsymbol L^{-1}\boldsymbol m \right \}\boldsymbol y$ 为奇函数，公式 (11) 代入 (10) 可得 $-\frac{1}{2}\boldsymbol L^{-1}\boldsymbol m=\boldsymbol 0$，进一步可得 $\boldsymbol m=\boldsymbol 0$。证明一元高斯分布的熵值最大化时我们利用已知条件 $\int \exp(-x^2)\mathrm dx=\sqrt{\pi}$，多元高斯分布的推导需要利用矩阵的指数函数的性质，依次按照一元高斯分布的方式计算矩阵每一个元素，然后再求和，就可以得出 $\boldsymbol L=-\frac{1}{2}\boldsymbol \Sigma$，并且有 $\lambda-1=\ln\left \{ \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{|\boldsymbol \Sigma|^{\frac{1}{2}}} \right \}$。全部代入，就可以得出当概率分布 $p(\boldsymbol x)$ 的熵最大时，其函数形式满足高斯分布。

&emsp;&emsp;求⾼斯分布的熵，可以得到 $H[x]=\frac{1}{2}\left \{ 1+\ln (2\pi\sigma^2) \right \}$。