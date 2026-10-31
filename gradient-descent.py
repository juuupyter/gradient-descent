import matplotlib.pyplot as plt 
import numpy as np

#datapoints
seed = np.random.default_rng(11)
height = seed.random(10) * 100
weight = height / 2 + seed.normal(1,0.1,10)


#line predictions
m = 0.4
c = 5
predicted = m * height + c

# Residuals: actual - predicted
residuals = weight - predicted
print(height, weight)

# Plot data, line, and residual segments
plt.scatter(height, weight, label='Data')

x_line = np.linspace(0, 100, 100)
plt.plot(x_line, m * x_line + c, color='red', label=f'y = {m}x + {c}')

# Draw vertical lines showing each residual
plt.vlines(height, predicted, weight, colors='gray', linestyles='dashed')

plt.xlabel('height')
plt.ylabel('weight')
plt.legend()
plt.show()