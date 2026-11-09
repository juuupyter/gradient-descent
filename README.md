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

#Define history array so results can be shown

history = []
```
## Main training loop
```
for i in range(1001):

    #Formula for residual: y - (m * x + c) 
    residuals = y - (m * x + c)

    #Formula for sum of squared residual: Sigma(y - (m * x + c))²
    #In this case we are using mean squared error instead, as np.sum does not take into account the amount of data
    #(len(x,y)), which would mean the learning rate would need adjusted. 
    mse = np.mean(residuals**2)
```

Given our MSE formula: $\sum (yᵢ - (mxᵢ + c))² $, we can derive with respect to the intercept (c):
$\sum 2 * (yᵢ - (mxᵢ + c)) * u $. <br />
Treating everything but $c$ in the equation $u$ as a constant, we get $c = -1$. <br />
This gives us our formula for the optimum gradient of c: $\sum -2 * (yᵢ - (mxᵢ + c))$
```
    gradient_c = -2 * np.mean(residuals)
```
We can do the same with respect to the slope (m) -> $\sum 2 * (y - (mx + c) * u $. <br /> 
Then, treating everything but $m$ in $u$ as a constant, we get that $m = -x$. (Since $m$ was dependent on $x$ for its value.) <br />
This gives us our formula for the optimum gradient of m: $\sum -2xᵢ * (yᵢ - (mxᵢ + c))$
```
    gradient_m = -2 * np.mean(x * residuals)
```
### Performing descent
    #Redfine m and c per step to perform descent. 
    m = m - lr * gradient_m
    c = c - lr * gradient_c

    #Record variables every 10 iterations
    if i % 10 == 0:
        history.append((i,m,c,mse))
2-norm ($$\sqrt{\sum |xᵢ|²}$$) of the two gradients. This gives us a vector for the error surface which can stop the loop when sufficiently small.
```
    if np.linalg.norm([gradient_m, gradient_c]) < 1e-5:
        break
```

## Plotting the results
