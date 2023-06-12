# Graudate Texts in Mathematics - Applications of Lie Groups to Differential Equations
 
## Chapter 1 - Introduction to Lie Groups
> "*...Once we have freed outselves of this dependence on coordinates, it is a small step to the general definition of a smooth manifold.*" - Olver, pg. 3

We want to understand what a Lie Group is, given the simple definition that it is a Group that is also a Manifold.

To begin, we are working towards understanding smooth manifolds as a means to move away from defining transformations applied on objects in terms of local coordinates. 

To do this, let's start with a definition.

### **Definition 1.1** - **$M$-Dimensional Manifold**
An **$m$-dimensional manifold** is a set **$M$**, together with a countable collection of subsets **$U_{\alpha} \subset M$**, called ***coordinate charts***, and one-to-one functions **$\chi_\alpha \colon U_\alpha \mapsto V_\alpha$** onto connected open subsets **$V_{\alpha}\subset \mathbb{R}^m$**, called ***local coordinate maps***, which satisfy the following properties:

*a)* The ***coordinate charts*** *cover* **$M$**:
$$\bigcup_{\alpha} U_{\alpha} = M$$

*b)* On the overlap of any pair of coordinate charts, $U_{\alpha}\cap U_{\beta}$, the composite map
$$
\chi_{\beta}\circ \chi_{\alpha}^{-1}\colon \chi_{\alpha}(
    U_{\alpha}\cap U_{\beta}
) \mapsto \chi_{\beta}(
    U_{\alpha}\cap U_{\beta}
)
$$

is a smooth (***infinitely differentiable***) function.

*c)* If $x \in U_{\alpha}$ and $\tilde x \in U_{\beta}$ are distinct points of **$M$**, then there exist open subsets $W\subset V_{\alpha}$, $\tilde W \subset V_{\beta}$ with $\chi_{\alpha}(x)\in W$, $\chi_{\beta}(\tilde x)\in \tilde W$, satisfying
$$
\chi_{\alpha}^{-1}(W)\cap\chi_{\beta}^{-1}(\tilde W) = \emptyset
$$

> #### ***Notes***
>
> *The local coordinate charts $\chi_{\alpha}\colon U_{\alpha} \mapsto V_{\alpha}$ endow the manifold $M$ with the structure of a topological space. Namely, we require that for each open subset $W\subset V_{\alpha}\subset\mathbb{R}^{m}$, $\chi_{\alpha}^{-1}(W)$ be an open subset of $M$. These sets form a *basis* for the topology on $M$, so that $U \subset M$ is open if and only if for each $x \in U$ there is a neighborhood of $x$ of the above form contained in $U$; i.e., $x \in \chi_{\alpha}^{-1}(W) \subset U$, where $\chi_{\alpha}\colon U_{\alpha} \mapsto V_{\alpha}$ is a coordinate chart containing $x$, and $W$ is an open subset of $V_{\alpha}$. In terms of this topology, the third requirement in the definition of a manifold is just a statement of the Hausdorff separation axiom. The degree of differentiability of the overlap functions $\chi_{\beta} \circ \chi_{\alpha}^{-1}$ determines the degree of smoothness of the manifold.*



### **Example 1.2**
The simplest $m$-dimensional manifold is just Euclidean space $\R^{m}$ itself.

