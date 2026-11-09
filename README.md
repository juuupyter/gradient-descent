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

Given our MSE formula: $\sum (yᵢ - (mxᵢ + c))² $, we can derive the equation with respect to the intercept (c):
$\sum (yᵢ - (mxᵢ + c))² = \sum 2 * (yᵢ - (mxᵢ + c)) * u $. Treating everything but c in the equation u as a constant, we get c = -1. 
This gives us our formula for the optimum gradient of c: $\sum -2 * (yᵢ - (mxᵢ + c))$
```
    gradient_c = -2 * np.mean(residuals)

    #Do the same with respect to the slope (m) -> Sigma(y - (m * x + c))² = Sigma 2 * (y - (m  * x + c)) * -x
    # -> Sigma -2xi * (yi - (m * xi +c))                                              (0)- (-1)* x +(0)  = -x
    gradient_m = -2 * np.mean(x * residuals)

    #Redfine m and c per step to perform descent. 
    m = m - lr * gradient_m
    c = c - lr * gradient_c

    #Record variables every 100 iterations
    if i % 10 == 0:
        history.append((i,m,c,mse))
```

    #2-norm of the two gradients (sqrt sum of squares). Measure the error surface and only stop when the surface is flat for BOTH variables. 
    if np.linalg.norm([gradient_m, gradient_c]) < 1e-5:
        break
