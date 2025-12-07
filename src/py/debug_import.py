import sys
print(sys.path)
try:
    import cuda
    print("cuda imported:", cuda)
except ImportError as e:
    print("cuda import failed:", e)

try:
    import cuda.tile
    print("cuda.tile imported:", cuda.tile)
except ImportError as e:
    print("cuda.tile import failed:", e)
