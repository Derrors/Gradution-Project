import matplotlib.pyplot as plt
import numpy as np

a = [0.677,0.534,0.711,0.705,0.726,0.741,0.789,0.766,0.843,0.875,0.886,0.891,0.899,.908,0.912]
b = [(i+1) for i in range(15)]
a = np.array(a)
b = np.array(b)

plt.figure(0)
plt.plot(b,a,'or-')
plt.xlabel("epochs")
plt.ylabel("Acc")
plt.show()