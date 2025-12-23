import math

# config

LUT_SIZE = 256           # Number of entries in the table
X_MIN = -8.0             # Lower bound of sigmoid input
X_MAX =  8.0             # Upper bound of sigmoid input
Q_SCALE = 256            # Q8.8 scaling factor (1.0 -> 256)

# sigmoid function

def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))

# LUT gen

lut = []

for i in range(LUT_SIZE):
    # Map index -> real input value
    # i = 0      -> X_MIN
    # i = 255    -> X_MAX
    x = X_MIN + (X_MAX - X_MIN) * i / (LUT_SIZE - 1)

    # Compute sigmoid
    y = sigmoid(x)

    # Convert to Q8.8 fixed-point
    y_q88 = int(round(y * Q_SCALE))

    # Clamp to valid range [0, 256]
    y_q88 = max(0, min(Q_SCALE, y_q88))

    lut.append(y_q88)

#prints in c format

print("const int16_t sigmoid_lut_256[256] = {")
for i in range(0, LUT_SIZE, 16):
    row = ", ".join(f"{v:3d}" for v in lut[i:i+16])
    print("   ", row, ",")
print("};")
