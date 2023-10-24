# Lattheat by LatticeGPU.jl
This code is an independent copy of [`su2-higgs`](https://igit.ific.uv.es/gtelo/latticegpu.jl) branch of [LatticeGPU](https://igit.ific.uv.es/alramos/latticegpu.jl), aimed to simulate ZQCD. To know what ZQCD is, look at the paper [here](https://arxiv.org/abs/0801.1566v2).  
  
## Action
We are simulating a 3d theory with an $SU(2)$ gauge field $U_{ij}$  and 1 adjoint scalar field called Z, parameterized as
$$Z = \frac{\Sigma}{2} + i \frac{\Pi_a}{2}\sigma_a \in SU(2)$$
The ZQCD lattice action is 
$$S = S_{w}^{d=3} + S_Z + V(\Sigma,\Pi)$$
where 
$$S_Z = \bigg(\frac{4}{\beta}\bigg)\biggl(\sum_{n,i} [\Sigma^2(n) - \Sigma(n)\Sigma(n+\hat i)] + 2\sum_{n,i}      \text{tr}[\Pi(n)^2 - \Pi(n)U_i(n)\Pi(n+\hat i)U_i^\dagger(n)]  \biggr)$$
and 
$$V = \biggl(\frac{4}{\beta}\biggr)^3\sum_n[\hat b_1\Sigma^2 + \hat b_2\Pi_a^2 + \hat c_1  \Sigma^4 + \hat c_2 (\Pi_a^2)^2 + \hat c_3  \Sigma^2\Pi_a^2]$$


## HMC
For the HMC we need to compute forces.
### Forces
We collect here the forces used in the HMC. For $\Sigma$:
$$\frac{\partial}{\partial \Sigma_x} (S_Z+V)= \biggl(\frac{4}{\beta}\biggr)\bigg[6\Sigma_x - \sum_{\hat i}(\Sigma_{x+\hat i}+\Sigma_{x-\hat i}) +  \biggl(\frac{4}{\beta}\biggr)^2(2b_1\Sigma_x + 4c_1 \Sigma_x^3 + 2 c_3 \Sigma_x \Pi_a^2) \bigg]$$
for $\Pi$:
$$\frac{\partial S_Z}{\partial \Pi_a(x)} = 2 \Big(\frac{4}{\beta}\Big)\text{tr} \biggl[i\sigma_a\Bigl(2\Pi_x - U_i(x)\Pi_{x+\hat i}U_i^\dagger(x)+U_i^\dagger(x-\hat i)\Pi_{x-\hat i}U_i(x-\hat i) \Bigr)\biggr]$$
$$\frac{\partial V}{\partial \Pi_a(x)} = -16c_2 \Big(\frac{4}{\beta}\Big)^3\biggl(\frac{b_2+c_3\Sigma_x^2}{4c_2}+2\det\Pi_x\biggr)\text{tr}(i\sigma_a \Pi_x)$$
and for gauge fields
$$\frac{\partial S_Z}{\partial U_i(x)} = \frac{8}{\beta}\text{tr}\biggl[i\sigma_a\Bigl(U_i(x)\Pi_{x+\hat{i}}U_i^\dagger(x)\Pi_x - U_i^\dagger(x)\Pi_x U_i(x)\Pi_{x+\hat i}\Bigr)\biggr]$$
### Some useful formulas
#### General $SU(2)$ things
An algebra matrix, as $\Pi$ satisfy
$$\Pi^2 = -\det(\Pi)\mathbf 1_2$$
$\mathbf 1_2$ being the $2\times 2$ identity matrix. Moreover, the norm $\Pi_a^2\equiv \Pi_1^2+\Pi_2^2+\Pi_3^2$ satisfy
$$\det\Pi=\frac{\Pi_a^2}{4}$$
and
$$\text{tr}\Pi^2 = -\frac{\Pi_a^2}{2} = -2\det \Pi$$


#### Group and algebra derivatives
The derivative of a function of a group element with respect to the element of the group can be calculated operatively by
$$\frac{\text{d}f[U]}{\text{d}U} = -\frac{1}{i} \frac{\text{d}}{\text{d}s} f\big[e^{-isX}U_i(x)\big] \biggl|_{s=0}$$
with 
$$e^{-isX}U_\mu(x) = e^{-is\delta(x-y)\delta_{ij}\sigma_a}U_i(x)$$
while for an element of the algebra 
$$\frac{\text{d}f[\Pi]}{\text{d}\Pi} = \frac{\text{d}}{\text{d}s} f\bigl[sX+M\bigr]$$
with $X=i\delta(x-y)\sigma_a$.

##### Derivative of the determinant of an algebra matrix
We can use $\Pi^2 = -\det(\Pi)\mathbf 1_2$ to easily demonstrate that
$$\det(sX+M) = \det\Pi\det\biggl(\mathbf 1 +\frac{s}{-\det\Pi}X\Pi\biggr)$$
and we can use the fact that $\frac{\text{d}}{\text{d}t}\det(\mathbf 1 + tM)=\text{tr}(M)$ to conclude that
$$\frac{\text{d}}{\text{d}s}\det(sX + M) = -\text{tr}(XM)$$

