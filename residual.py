x = [1.0, 3.0, 5.0, 7.0]
reconstruction = [0.0] * len(x)
levels = [1, 2, 4]
codebooks = [[0.0, 4.0, 8.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]

for positions, codebook in zip(levels, codebooks):
    width = len(x) // positions
    residual = [a - b for a, b in zip(x, reconstruction)]
    pooled = [sum(residual[i:i + width]) / width
              for i in range(0, len(x), width)]
    values = [min(codebook, key=lambda entry: (v - entry) ** 2)
              for v in pooled]
    correction = [v for v in values for _ in range(width)]
    reconstruction = [a + b for a, b in zip(reconstruction, correction)]
    print(positions, reconstruction)
