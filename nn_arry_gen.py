import numpy as np

#quad variables

a = 3.0
b = -5.0
c = 7.0

#range

x_min = -10.0
x_max =  10.0
num_samples = 21   

# sample real vals

x_real = np.linspace(x_min, x_max, num_samples)
y_real = a*x_real**2 + b*x_real + c

# normalzation
# Inputs → [-1, +1]
x_norm = x_real / max(abs(x_min), abs(x_max))

# Outputs → [0, 1]
y_min = y_real.min()
y_max = y_real.max()
y_norm = (y_real - y_min) / (y_max - y_min)

# fixed point (Q8.8 so scale = 256)

SCALE = 256

x_fixed = np.round(x_norm * SCALE).astype(int)
y_fixed = np.round(y_norm * SCALE).astype(int)

# prints the c array vals

print("const int16_t training_inputs[] = {")
print(", ".join(map(str, x_fixed)))
print("};\n")

print("const int16_t training_outputs[] = {")
print(", ".join(map(str, y_fixed)))
print("};\n")

print(f"// y_real range: [{y_min:.3f}, {y_max:.3f}]")
print(f"// normalization: y_norm = (y - {y_min:.3f}) / ({y_max - y_min:.3f})")
