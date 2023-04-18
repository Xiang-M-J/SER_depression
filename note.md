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
其中H表示这个距离是由$\phi()$将数据映射到再生希尔伯特空间中进行度量

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
由于核函数会将特征映射到更高的维度上，所以$\phi(x_i)^T\phi(x_j)$这种矩阵乘会增加计算量，损益定义核函数需要具备如下特点：
$$
K(x_i,x_j)=K(x_i^Tx_j)=\Phi(x_i)^T\Phi(x_j)
$$
即K对$x_i^Tx_j$的结果进行计算等价于映射之后的结果再进行点乘操作，这样可以减小运算复杂度。常用的有线性核函数（即无核函数）、多项式核函数、高斯核、sigmoid核函数

将$\phi(x_i)\phi(x_i')$联系核函数，所以MMD可以表示为：
$$
MMD(X,Y)=||\frac{1}{n^2}\sum_i^n\sum_{i'}^nk(x_i,x_i')-\frac{2}{nm}\sum_i^n\sum_j^mk(x_i,y_j)+\frac{1}{m^2}\sum_j^m\sum_{j'}^mk(y_j,y_j')||_H
$$
在大多数论文中采用高斯核函数$k(u,v)=e^{-\frac{||u-v||^2}{\sigma}}$

在实际实现时，为了方便计算引入了一个核矩阵方便计算
$$
\left[
\begin{matrix}
K_{s,s} & K_{t,s}  \\
K_{s,t} & K_{t,t}   \\
\end{matrix}
\right]
\tag{2}
$$
以及L矩阵：
$$
\begin{equation}
I_{i,j}
=\left\{
	\begin{array}{ll}
		\frac{1}{n^2}\;\;\; x_i,x_j\in D_s \\
		\frac{1}{m^2}\;\;\; x_i,x_j\in D_t \\
		-\frac{1}{nm}\;\;otherwise
	\end{array}\right.
\end{equation}
$$
