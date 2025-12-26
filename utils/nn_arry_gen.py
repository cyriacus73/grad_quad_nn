import numpy as np

# Generate training data for y = 3x² - 5x + 7
a, b, c = 3.0, -5.0, 7.0

# Training range x ∈ [-2, 2], 35 samples
x_min, x_max = -2.0, 2.0
num_samples = 35

# Sample quadratic function
x_real = np.linspace(x_min, x_max, num_samples)
y_real = a * x_real**2 + b * x_real + c

# Normalize inputs: [-2,2] → [-1,1]
x_norm = x_real / max(abs(x_min), abs(x_max))

# Normalize outputs: [y_min,y_max] → [0,1]
y_min, y_max = y_real.min(), y_real.max()
y_norm = (y_real - y_min) / (y_max - y_min)

# Convert to Q8.8 fixed-point
SCALE = 256
x_fixed = np.round(x_norm * SCALE).astype(int)
y_fixed = np.round(y_norm * SCALE).astype(int)

# Output C arrays for PIC code
print("const int16_t training_inputs[] = {")
print(", ".join(map(str, x_fixed)))
print("};\n")

print("const int16_t training_outputs[] = {")
print(", ".join(map(str, y_fixed)))
print("};\n")

# Denormalization constants
print(f"// y_real range: [{y_min:.3f}, {y_max:.3f}]")
print(f"// y_norm = (y - {y_min:.3f}) / ({y_max - y_min:.3f})")
print(f"// y_real = y_norm * {y_max - y_min:.3f} + {y_min:.3f}")