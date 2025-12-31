import matplotlib.pyplot as plt
import numpy as np

# True quadratic function
def quadratic(x):
    return 3 * x**2 - 5 * x + 7

x_vals = [-2.0, -1.9, -1.8, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0]
y_vals = [23.2, 22.1, 21.6, 20.9, 20.2, 19.1, 18.3, 17.3, 16.7, 15.6, 14.8, 13.9, 12.9, 12.3, 11.2, 10.7, 10.1, 9.2, 8.9, 8.4, 8.0, 7.7, 7.4, 7.5, 7.6, 7.8, 8.4, 8.6, 8.9, 9.4, 9.9, 10.4, 11.2, 11.8, 12.2]

# Generate smooth x values for the function
x_smooth = np.linspace(-2, 2, 100)
y_smooth = quadratic(x_smooth)

# plots the true function
plt.plot(x_smooth, y_smooth, label='True quadratic: y = 3x² - 5x + 7', color='blue')

# plots predicted points
plt.scatter(x_vals, y_vals, label='Predicted points', color='red', marker='o')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Function vs Predicted Points')
plt.legend()
plt.grid(True)

# Shows the plot
plt.show()
