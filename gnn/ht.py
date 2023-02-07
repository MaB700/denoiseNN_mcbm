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

pos = data[1].pos.detach().cpu().numpy()
t = data[1].t.detach().cpu().numpy()

print(data[1])

ht = HoughTransform(pos[:, 0], pos[:, 1], t[:,0])
print("HT points: ", len(ht.result[0]))
# print(ht.indices.shape)



