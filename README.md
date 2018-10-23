# ElasticNet in Sklearn does not result in consistent result with enet_path

To show this, type
`python test.py`

Note that using the same arguments in both ElasticNet and MultiElasticNet, the result differs from enet_path.

**It turns out that `enet_path` give you the right result with sparsity while ElasticNet cannot!**
