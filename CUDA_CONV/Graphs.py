# Kaustav Vats (2016048)

import matplotlib.pyplot as plt

# In[]
x = [64, 128, 256, 512, 1024, 2048]
y = [
    [17.48, 83.5008, 167.7, 249.205, 266.049, 279.15],
    [42.4391, 151.097, 222.049, 260.524, 272.326, 275.302],
    [67.2689, 189.466, 245.54, 265.623, 264.458, 260.811],
    [333.551, 216.654, 266.094, 308.191, 318.506, 328.045],
    [121.262, 238.01, 271.682, 337.681, 420.93, 467.673]
]
size = [3, 5, 7, 9, 11]
# kernel_3 = 
# kernel_5 = 
# kernel_7 = 
# kernel_9 = 
# kernel_11 = 

# In[]
color = ['b', 'g', 'r', 'm', 'y']
plt.figure()
for i in range(5):
    plt.plot(x, y[i][:], color[i], label="Speedup Curve Kernel size %d"%size[i])
plt.ylabel("Speedups")
plt.xlabel("Image Size")
plt.title("Convolution Speedup Curve for different Kernel Size")
plt.legend(loc='lower right')
plt.show()
# plt.savefig("speedup_curve.png")

#%%
