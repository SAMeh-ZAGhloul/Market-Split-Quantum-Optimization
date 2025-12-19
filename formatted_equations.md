# Market Split Problem - Mathematical Formulations

## Feasibility Variant (fMSP)

**Problem Statement:**
Find a binary vector $x \in \{0, 1\}^n$ satisfying:

$$\sum_{j=1}^n a_{ij}x_j = d_i, \quad \text{for } i = 1, \dots, m$$

**Parameters:**
- $a_{ij}$: demand of retailer $j$ for product $i$
- $d_i$: target allocation for product $i$
- $x_j \in \{0, 1\}$: binary decision variable indicating whether retailer $j$ is selected

---

## Optimization Variant (OPT)

**Problem Statement:**
Minimize total slack when perfect split is impossible:

$$\min \sum_{i=1}^m |s_i|$$

**Subject to:**

$$\sum_{j=1}^n a_{ij}x_j + s_i = d_i, \quad i = 1, \dots, m$$

$$x_j \in \{0, 1\}, \quad s_i \in \mathbb{Z}$$

**Parameters:**
- $x_j \in \{0, 1\}$: binary decision variable indicating whether retailer $j$ is selected
- $s_i \in \mathbb{Z}$: slack variable for product $i$ (integer-valued)
- $|s_i|$: absolute value representing the deviation from target allocation

---

## Summary

The **fMSP** variant seeks a perfect allocation where each product's demand is exactly met by the selected retailers. The **OPT** variant allows for imperfect allocations by introducing slack variables, minimizing the total deviation from the target allocations.
