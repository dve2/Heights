import numpy as np
import matplotlib.pyplot as plt


def read_bruker_binary(file, offset, shape=(128, 128), dtype=np.int16):
    with open(file, 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=np.prod(shape))
    return data.reshape(shape)


raw_height = read_bruker_binary("tests/data/NEPC1Y2.005", 8192)
print(f"raw_height min: {raw_height.min()}  max: {raw_height.max()}")

height = read_bruker_binary("tests/data/NEPC1Y2.005", 8192) * 16.1335 / 32768  # in nm
#phase = read_bruker_binary("tests/data/NEPC1Y2.005", 40960) * 87.5061 / 32768  # in deg

plt.figure(figsize=(8, 6))
plt.imshow(height)
plt.colorbar(label='Height (nm)')
plt.title("AFM Height Topography")
plt.savefig("heights.png")

print(height)