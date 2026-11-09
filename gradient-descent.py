import matplotlib.pyplot as plt 
import numpy as np

#datapoints
seed = np.random.default_rng(11)
weight = seed.random(10)
height = weight + seed.normal(0.5, 0.1, 10)

#use least squares for an estimate of the slope
#y = mx+c
m = np.polyfit(weight,height,1)[0]
x,y = weight, height
#starting value of intercept, learning rate 
c = 0.0
lr = 0.01

for i in range(1001):

    #Formula for residual: y - (m * x + c) 
    residuals = y - (m * x + c)

    #Formula for sum of squared residual: Sigma(y - (m * x + c))²
    #In this case we are using mean squared error instead, as np.sum does not take into account the amount of data (len(x,y)), which would mean the learning rate would need adjusted. 
    mse = np.mean(residuals**2)

    #We can derive this equation with respect to the intercept (c) -> Sigma(y - (m * x + c))² = Sigma 2 * (y - (m * x + c) * -1 
    # -> Sigma -2 * (yi - (m * xi + c))                                                                   (0)- (0) (0) (1) = -1
    gradient_c = -2 * np.mean(residuals)

    #Do the same with respect to the slope (m) -> Sigma(y - (m * x + c))² = Sigma 2 * (y - (m  * x + c)) * -x
    # -> Sigma -2xi * (yi - (m * xi +c))                                              (0)- (-1)* x +(0)  = -x
    gradient_m = -2 * np.mean(x * residuals)

    #Redfine m and c per step to perform descent. 
    m = m - lr * gradient_m
    c = c - lr * gradient_c

    #2-norm of the two gradients (sqrt sum of squares). Measure the error surface and only stop when the surface is flat for BOTH variables. 
    if np.linalg.norm([gradient_m, gradient_c]) < 1e-5:
        break

print(f"\nGradient descent intercept: {c:.4f}")
print(f"Closed-form intercept:      {np.mean(y) - m * np.mean(x):.4f}")


