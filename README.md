# Gradient Descent Demo

To create the data, I am using np.random.default_rng() to ensure replicable results. 
```
seed = np.random.default_rng(11)
weight = seed.random(10)
height = weight + seed.normal(0.5, 0.1, 10)
```
10 samples are created for weight with rng.random(). Then, normally distributed values are added to the weight values to create height values.  

## Defining the variables for gradient descent:
```
#Define datapoints
x,y = weight, height

#Starting value of slope, intercept, learning rate
m = 0.0
c = 0.0
lr = 0.01
```

**Define history array so results can be shown**
```
history = []
```
