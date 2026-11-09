## Gradient Descent Demo

To create the data, I am using np.random.default_rng() to ensure replicable results. 
```
seed = np.random.default_rng(11)
weight = seed.random(10)
height = weight + seed.normal(0.5, 0.1, 10)
```
