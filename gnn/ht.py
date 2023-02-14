import matplotlib.pyplot as plt
from ht_cpp.build.ht import *
from mcbm_dataset import *
# a = ht([1,2,3,5,6,9], [4,5,6,1,2,3], [7,8,9,5,6,4])
# print(a)
num_samples = 10
reload = True
r = 7
max_num_neighbors = 8
data = MyDataset(   dataset="train", N = num_samples, reload=reload, \
                    radius = r, max_num_neighbors = max_num_neighbors)

pos = data[3].pos.detach().cpu().numpy()
t = data[3].t.detach().cpu().numpy()

print(data[1])

ht = HoughTransform(pos[:, 0], pos[:, 1], t[:,0])
print("HT points: ", len(ht.result[0]))
# plot the result[0] and result[1] in a scatter plot and result[2] in a bar plot side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.scatter(pos[:, 0], pos[:, 1], t[:,0])
ax1.set_xlim(0.0, 31.0)
ax1.set_ylim(0.0, 71.0)
ax2.scatter(ht.result[0], ht.result[1], c='r')
ax2.set_xlim(0.0, 31.0)
ax2.set_ylim(0.0, 71.0)
ax3.bar(range(len(ht.result[2])), ht.result[2])
plt.show()





