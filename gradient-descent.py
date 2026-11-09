import matplotlib.pyplot as plt 
import numpy as np

#Create datapoints
seed = np.random.default_rng(11)
weight = seed.random(10)
height = weight + seed.normal(0.5, 0.1, 10)

#Define datapoints
x,y = weight, height

#Starting value of slope, intercept, learning rate
m = 0.0
c = 0.0
lr = 0.01

#Define history array so results can be shown
history = []

#Main loop
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

    #Record variables every 100 iterations
    if i % 10 == 0:
        history.append((i,m,c,mse))

    #2-norm of the two gradients (sqrt sum of squares). Measure the error surface and only stop when the surface is flat for BOTH variables. 
    if np.linalg.norm([gradient_m, gradient_c]) < 1e-5:
        break

#Compare results between the optimal value (least squares) and the values gradient descent achieved. 
best_m, best_c = np.polyfit(x, y, 1)
print(f"Gradient descent: m={m:.4f}, c={c:.4f}")
print(f"np.polyfit:       m={best_m:.4f}, c={best_c:.4f}")

#Plotting

#Create a figure with 2 subplots for showing the line update and error over time. 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#Use a diverging scale to show improvement, use linspace to create a color entry for each element in len(history)
colors = plt.cm.RdYlGn(np.linspace(0,1,len(history)))
#Create 100 samples between x.min - x.max to plot lines with. 
x_line = np.linspace(x.min(), x.max(), 100)
#Plot datapoints, use zorder 3 to ensure points display above lines. 
ax1.scatter(x, y, color="black", zorder=3, label="data")

#For each iteration, m, and c value in history, create lines and pair with color from colors, ignoring mse value. 
for (it, m_i, c_i, _), col in zip(history, colors):
    #Given m_i and c_i, calculate heights for the horizontal values given by x_line, plotting it along the linspace. 
    #If it is 0 or the last iter value in history, add it to the legend, otherwise discard.
    ax1.plot(x_line, m_i * x_line + c_i, color=col, alpha=0.8, label=f"iter {it}" if it in (0, history[-1][0]) else None)

ax1.plot(x_line, best_m * x_line + best_c, "r--", label="polyfit")
ax1.set_xlabel("weight")
ax1.set_ylabel("height")
ax1.set_title("Fitted line every 10 iterations")
ax1.legend()

plt.tight_layout()
plt.show()




