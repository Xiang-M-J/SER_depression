## 领域适应理论

领域迁移的经典理论由Ben-David在2010年给出[1]，该理论给出了模型在目标域上的误差的上界，上界由四项的和组成: 源域上的误差+目标域和源域的距离+模型复杂度+理想模型的误差。上界的推导基于二分类问题，一个域包含一个概率分布$D$和一个分类函数$f$，$f$将$D$中的数据映射到正确的类别$f:X->[0,1]$，源域记作$<D_S, f_S>$，目标域记作$<D_T, f_T>$。

一个假设$h$是一个分类器$h: X\rightarrow[0,1]$。$h$在源域上的误差就是$h$在多大程度与$f$不同，$\epsilon_S(h)=\epsilon_S(h,f_S)=E_{x\sim D_s}[|h(x)-f_x(x)|]$，$h$在源域上抽样的训练数据上的经验误差记作$\hat\epsilon_S(h)$，同理目标域的经验误差记作$\hat\epsilon_T(h)$

接下来，这里需要使用衡量两个概率分布间差异的variation divergence。对于两个概率分布$D$和$D'$ ，他们之间的variation divergence定义为：
$$
d_1(D, D')=2sup_{B\in\mathbb{B}}|Pr_{D}[B]-Pr_{D'}[B]|
$$
这是在所有的可能事件$B\in\mathbb{B}$中，两个概率分布$D$和$D'$可以分配给同一事件的概率的最大可能差异的2倍。有了variation divergence，对于一个假设 ℎ ， ℎ 在源域上的误差(source error) $\epsilon_S(h)$和目标域上的误差(target error)$\epsilon_T(h)$就可以被下式联系起来。
$$
\epsilon_T(h)<=\epsilon_S(h)+d_1(D_S, D_T)+min\{E_{D_S}[|f_S(x)-f_T(x)|], E_{D_T}[|f_S(x)-f_T(x)|]\}
$$
证明过程如下：
$$
\epsilon_T(h)=\epsilon_T(h)+\epsilon_S(h)-\epsilon_S(h)+\epsilon(h,f_T)-\epsilon(h,f_T)\\
\le\epsilon_S(h)+|\epsilon_S(h,f_T)-\epsilon_T(h,f_T)|+|\epsilon_T(h,f_T)-\epsilon_S(h,f_T)|\\
\le\epsilon_S(h)+E_{D_S}[|f_s(x)-f_T(x)|]+|\epsilon_T(h,f_T)-\epsilon_S(h,f_T)|\\
\le\epsilon_S(h)+E_{D_S}[|f_s(x)-f_T(x)|]+\int|\phi_S(x)-\phi_T(x)||h(x)-f_T(x)|dx\\
\le\epsilon_S(h)+E_{D_S}[|f_S(x)-f_T(x)|]+d_1(D_S, D_T)
$$
...更多内容在https://zhuanlan.zhihu.com/p/336501367



## MMD

MMD（最大均值差异）是域适应中最常用的一种的损失函数，主要用于度量两个不同但相关的分布的距离。两个分布的距离定义为：
$$
MMD(X,Y)=||\frac{1}{n}\sum_{i=1}^n\phi(x_i)-\frac{1}{n}\sum_{j=1}^m\phi(y_i)||_H^2
$$
其中H表示这个距离是由$\phi()$将数据映射到再生希尔伯特空间（映射到再生希尔伯特空间是可以将两点之间的距离用两个点的内积进行表示）中进行度量

MMD的求解：
$$
MMD(X,Y)=||\frac{1}{n^2}\sum_i^n\sum_{i'}^n\phi(x_i)\phi(x_i')-\frac{2}{nm}\sum_i^n\sum_j^m\phi(x_i)\phi(y_j)+\frac{1}{m^2}\sum_j^m\sum_{j'}^m\phi(y_j)\phi(y_j')||_H
$$
引入SVM的核函数概念，SVM推导后的公式为：
$$
\mathop{min}_{\alpha}\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^m\alpha_i\alpha_jy_iy_jx_i^Tx_j-\sum_{i=1}^m\alpha_i\\
s.t.\;\;\sum_{i=1}^m\alpha_iy_i=0\;\; 0\le\alpha_i\le C, i=1,2,3,..,n
$$
将核函数代入可得
$$
\mathop{min}_{\alpha}\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^m\alpha_i\alpha_jy_iy_j\phi(x_i)^T\phi(x_j)-\sum_{i=1}^m\alpha_i\\
s.t.\;\;\sum_{i=1}^m\alpha_iy_i=0\;\; 0\le\alpha_i\le C, i=1,2,3,..,n
$$
由于核函数会将特征映射到更高的维度上，所以$\phi(x_i)^T\phi(x_j)$这种矩阵乘会增加计算量，所以定义核函数需要具备如下特点：
$$
K(x_i,x_j)=K(x_i^Tx_j)=\Phi(x_i)^T\Phi(x_j)
$$
即K对$x_i^Tx_j$的结果进行计算等价于映射之后的结果再进行点乘操作，这样可以减小运算复杂度。常用的有线性核函数（即无核函数）、多项式核函数、高斯核、sigmoid核函数

将$\phi(x_i)\phi(x_i')$联系核函数，在实际实现时，又会为了方便计算引入了一个核矩阵方便计算
$$
\left[
\begin{matrix}
K_{s,s} & K_{t,s}  \\
K_{s,t} & K_{t,t}   \\
\end{matrix}
\right]
\tag{2}
$$
以及M矩阵：
$$
\begin{equation}
M_{i,j}
=\left\{
	\begin{array}{ll}
		\frac{1}{n^2}\;\;\; x_i,x_j\in D_s \\
		\frac{1}{m^2}\;\;\; x_i,x_j\in D_t \\
		-\frac{1}{nm}\;\;otherwise
	\end{array}\right.
\end{equation}
$$
所以MMD可以表示为
$$
MMD(X,Y)=||\frac{1}{n^2}\sum_i^n\sum_{i'}^nk(x_i,x_i')-\frac{2}{nm}\sum_i^n\sum_j^mk(x_i,y_j)+\frac{1}{m^2}\sum_j^m\sum_{j'}^mk(y_j,y_j')||_H \\
=tr(\left[
\begin{matrix}
K_{s,s} & K_{s,t}  \\
K_{t,s} & K_{t,t}   \\
\end{matrix}
\right]
\tag{2}M)
$$
在大多数论文中采用高斯核函数$k(u,v)=e^{-\frac{||u-v||^2}{\sigma}}$



## Wasserstein Distance(推土机距离)

### 推土机距离

-  如果我们将分布想象为两个有一定存土量的土堆，每个土堆维度为 N，那么 EMD 就是将一个土堆转换为另一个土堆所需的最小总工作量。工作量的定义是单位泥土 的总量乘以它移动的距离。两个离散的土堆分布记作 Pr 和 Pθ  ，以以下两个任意的分布为例。  

[EMD(earth mover's distances)距离 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/145739750)

假设存在两个分布$P_r(x)$和$P_{\theta}(y)$，它们之间的推土机距离可以表示为
$$
W[P_r,P_{\theta}]=\frac{inf}{\gamma\in\prod[P_r,P_{\theta}]}\int_x\int_y\gamma(x,y)d(x,y)dxdy
$$
$inf$表示下界，即最小的距离，要从所有的运输方案$\gamma$中找到使$\int_x\int_y\gamma(x,y)d(x,y)dxdy$最小的方案$\gamma(x,y)$。

$d(x,y)$是移动土堆的成本函数，可以用$||x-y||_1 \;\;||x-y||_2\;\;||x-y||_2^2$

矩阵表示：实际就是解决$\int_x\int_y\gamma(x,y)d(x,y)dxdy$的最小值，其中$d(x,y)$确定，最优化需要满足以下优化条件：$\int\gamma(x,y)dy=P_r(x), \int\gamma(x,y)dx=P_{\theta}(y)$，且$\gamma(x,y)\ge0$，因为搬走的土堆需要大于等于0。
$$
\int_x\int_y\gamma(x,y)d(x,y)dxdy=\sum_i\sum_j\gamma(x_i,y_i)d(x_i,y_j)=\Gamma\cdot D=<\Gamma,D> \\
\Gamma = [\gamma(x_1,y_1), \gamma(x_1,y_2),...,\gamma(x_1, y_n),\gamma(x_2,y_1), ...,\gamma(x_2, y_n),...,\gamma(x_n,y_1),...,\gamma(x_n,y_n)]^T \\
D=[d(x_1,y_1), d(x_1,y_2),...,d(x_1, y_n),d(x_2,y_1), ...,d(x_2, y_n),...,d(x_n,y_1),...,d(x_n,y_n)]^T
$$
约束条件可以写成
$$
\int\gamma(x,y)dy=\sum_j\gamma(x,y_j)=P_r(x)=[P_r(x_1),P_r(x_2),...,P_r(x_n)]^T \\
\int\gamma(x,y)dx=\sum_i\gamma(x_i,y)=P_{\theta}(y)=[P_{\theta}(y_1),P_{\theta}(y_2),...,P_{\theta}(y_n)]^T
$$
将$P_r(x)$和$P_{\theta}(y)$两个向量拼接成一个长向量$b$，两个约束条件可以写成$A\Gamma=b$
$$
b=[P_r(x_1),P_r(x_2),...,P_r(x_n)|P_{\theta}(y_1),P_{\theta}(y_2),...,P_{\theta}(y_n)]^T\\
\Gamma=[\gamma(x_1,y_1), \gamma(x_1,y_2),...,\gamma(x_1, y_n),\gamma(x_2,y_1), ...,\gamma(x_2, y_n),...,\gamma(x_n,y_1),...,\gamma(x_n,y_n)]^T
$$
对于$A\Gamma=b$，$\Gamma$的前n项$\gamma(x_1,y_1), \gamma(x_1,y_2),...,\gamma(x_1, y_n)$求和为$P_r(x_1)$，所以$A$矩阵可以表示为：

![img](https://pic2.zhimg.com/80/v2-964acff07b0bcbb7df485903e8029d59_720w.webp)

$A=[2n, n^2],\;\;\Gamma=[n^2,1],\;\;b=[2n,1]$，$n$为$P_r$或$P_{\theta}$的维度

所以最后的优化问题可以写成$\frac{min}{\Gamma}\{<\Gamma,D>|A\Gamma=b,\Gamma\ge0\}$
