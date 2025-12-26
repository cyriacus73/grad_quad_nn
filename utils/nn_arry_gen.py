import numpy as np

# Quadratic variables
a = 3.0
b = -5.0
c = 7.0

# Range (reduced to -2 to +2 for better fit with limited neurons)
x_min = -2.0
x_max = 2.0
num_samples = 35

# Sample real values
x_real = np.linspace(x_min, x_max, num_samples)
y_real = a * x_real**2 + b * x_real + c

# Normalization
# Inputs → [-1, +1] (scaled to Q8.8: -256 to +256)
x_norm = x_real / max(abs(x_min), abs(x_max))

# Outputs → [0, 1] (scaled to Q8.8: 0 to 256)
y_min = y_real.min()
y_max = y_real.max()
y_norm = (y_real - y_min) / (y_max - y_min)

# Fixed point (Q8.8 so scale = 256)
SCALE = 256
x_fixed = np.round(x_norm * SCALE).astype(int)
y_fixed = np.round(y_norm * SCALE).astype(int)

# Print C array values
print("const int16_t training_inputs[] = {")
print(", ".join(map(str, x_fixed)))
print("};\n")

print("const int16_t training_outputs[] = {")
print(", ".join(map(str, y_fixed)))
print("};\n")

print(f"// y_real range: [{y_min:.3f}, {y_max:.3f}]")
print(f"// normalization: y_norm = (y - {y_min:.3f}) / ({y_max - y_min:.3f})")
print(f"// de-normalization: y_real = y_norm * {y_max - y_min:.3f} + {y_min:.3f}")