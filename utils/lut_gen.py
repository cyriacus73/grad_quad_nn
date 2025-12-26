import math

LUT_SIZE = 256
X_MIN = -4.0              
X_MAX =  4.0              
Q_SCALE = 256             # Q8.8

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

lut = []

for i in range(LUT_SIZE):
    # Map index -> real z in [-4, +4]
    z = X_MIN + (X_MAX - X_MIN) * i / (LUT_SIZE - 1)

    y = sigmoid(z)

    y_q88 = int(round(y * Q_SCALE))
    y_q88 = max(0, min(Q_SCALE, y_q88))

    lut.append(y_q88)

print("const int16_t sigmoid_lut_256[256] = {")
for i in range(0, LUT_SIZE, 16):
    row = ", ".join(f"{v:3d}" for v in lut[i:i+16])
    print("   ", row, ",")
print("};")
