# Example: the suprisingly sensetivity of initial guess in `sklearn.linear_model.MultiTaskElasticNet` 

## Motivations
- For a task of drawing a **ElasticNet path for a multi-task linear regression problem** that is supposed to suit well with `MultiTaskElasticNet`, we found that the *path* from **`sklearn.linear_model.MultiTaskElasticNet`** in Sklearn is not consistent with that from **`sklearn.linear_model.enet_path`**
- In addition, we found something weird, if we feed the **alpha** in the ElastNet one by one versus feed a whole array of **alpha**, the **sklearn.linear_model.enet_path** gives different results  
- A first glance of the [source code](https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/linear_model/coordinate_descent.py#L1629) clearly shows that both `MultiTaskElasticNet` and `enet_path` are calling the same **`Cython`** function for doing the heavy lifting in ElasticNet in the [cd_fast](https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/linear_model/cd_fast.pyx) source code. 
- Even if we use the same arguments in the interface of both `MultiTaskElasticNet` and `enet_path`, the results are different, **which is not acceptable!** 

## Problem setup

I have 16 moderately correlated features, to predict 2 tasks. I want a the solution to be found via ElasticNet, i.e., l1+l2, path to search for a sparse yet also a unique solution for strongly correlated features space. Data is included in the source code `test.py`.

## What we find
- Going deep into the source code, Alex Sun found that in the specific block of [**enet_path**](https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/linear_model/coordinate_descent.py#L455), the for loop block is exactly reusing the coefficient from last alpha, under a sorted descending $\alpha$ array from large to small value. 
- Before running the algorithm written in `Cython` in `cd_fast`, besides regular checking just as any other well-written library, `enet_path` **forces to sort the alpha arrary**, if there is any, from large to small, and then run them one by one with each one reusing previous final coefficient as **initial guess**! This is what really different from the `MultiTaskElasticNet` or `ElasticNet` itself in `Sklearn`! 

## Try it out yourself!
To show this, type
1. `python test.py`
2. And then check out all the `png` files.

## Result and Conclusions
1. `enet_path` gives you the better result (the one gives smaller total loss) with sparsity while a blind calling from MultiTaskElasticNet cannot!
2. With a careful modification to force it to reuse previous final solution (and previous solution must come from a larger alpha), as noted by Alex Sun, one can improve the result from `MultiTaskElasticNet` to make it the same performance as `enet_path`. 

## Acknowledgment
I thank Alex Sun for pointing out the issues in the source code that the difference lies in whether or not reusing previous solutions. 