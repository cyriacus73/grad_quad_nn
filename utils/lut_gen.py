import math

# generates sigmoid LUT for Q8.8 fixed-point neural network

##configs
LUT_SIZE = 256
X_MIN, X_MAX = -4.0, 4.0  # Sigmoid input range
Q_SCALE = 256             # Q8.8 scale factor

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

lut = []

# index i maps to sigmoid(X_MIN + i * (X_MAX-X_MIN)/255)
for i in range(LUT_SIZE):
    z = X_MIN + (X_MAX - X_MIN) * i / (LUT_SIZE - 1)  # Map index to z ∈ [-4,4]
    y = sigmoid(z)                                     # σ(z) ∈ [0,1]
    y_q88 = int(round(y * Q_SCALE))                    # Convert to Q8.8
    y_q88 = max(0, min(Q_SCALE, y_q88))                # Clamp to [0,256]
    lut.append(y_q88)

# Output C array for PIC code
print("const int16_t sigmoid_lut_256[256] = {")
for i in range(0, LUT_SIZE, 16):
    row = ", ".join(f"{v:3d}" for v in lut[i:i+16])
    print("   ", row, ",")
print("};")
