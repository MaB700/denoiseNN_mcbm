import numpy as np
import math
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
import uproot

def CreateGraphDataset_quadrant(path, n, dist = 7):
    data = uproot.open(path)
    graphs = []
    for batch in tqdm(data.iterate(["time", "tar"], step_size=1000, entry_stop=None if n==0 else n)):
        time_batch = np.array(batch["time"])
        tar_batch = np.array(batch["tar"])
        graphs += [make_graph_quadrant(i, time_batch, tar_batch, dist) for i in range(len(time_batch))]

    graphs = [T.ToUndirected()(g) for g in graphs]
    # graphs = [T.RemoveIsolatedNodes()(g) for g in graphs]
    return graphs

def make_graph_quadrant(index, time, tar, dist):
    n = np.count_nonzero(time[index, :] > 0.0001)
    hit_indices = np.nonzero(time[index, :] > 0.0001)
    hits = time[index, :][hit_indices]
    y_pos, x_pos = np.divmod(hit_indices, 32)

    # Nodes
    x = np.zeros((n, 3))
    x[:, 0] = hits.astype('float32') # time [0,1]
    x[:, 1] = (x_pos.astype('float32'))/31.0 # x_coord [0,1]
    x[:, 2] = (y_pos.astype('float32'))/71.0 # y_coord [0,1]
    
    # Edges
    start_index = [] #np.empty((0), dtype=np.int_)
    end_index = [] #np.empty((0), dtype=np.int_)

    # TODO: kNN, instead of 1 neighbor (random or grid ordered) per quadrant
    k = 1

    for i in range(n): # source node
        indices_1 = []
        distances_1 = []
        indices_2 = []
        distances_2 = []
        indices_3 = []
        distances_3 = []
        indices_4 = []
        distances_4 = []
        for j in range(n): # target node
            if i == j : continue
            if (abs(x[i, 0] - x[j, 0]) > 0.20): continue # outside 5ns window
            if abs(x_pos[0, i] - x_pos[0, j]) > dist or abs(y_pos[0, i] - y_pos[0, j]) > dist: continue # outside dist x dist window   
            d2 = (x[j, 1] - x[i, 1])**2 + (x[j, 2] - x[i, 2])**2 # TODO: replace with x_pos, y_pos
            if (x_pos[0, j] >= x_pos[0, i]) and (y_pos[0, j] > y_pos[0, i]): # first quadrant
                indices_1.append(j)
                distances_1.append(d2)
            elif (x_pos[0, j] > x_pos[0, i]) and (y_pos[0, j] <= y_pos[0, i]): # second quadrant
                indices_2.append(j)
                distances_2.append(d2)
            elif (x_pos[0, j] <= x_pos[0, i]) and (y_pos[0, j] < y_pos[0, i]): # third quadrant
                indices_3.append(j)
                distances_3.append(d2)
            elif (x_pos[0, j] < x_pos[0, i]) and (y_pos[0, j] >= y_pos[0, i]): # fourth quadrant
                indices_4.append(j)
                distances_4.append(d2)
            else:
                print("Error no quadrant: ", x_pos[0, j], x_pos[0, i], y_pos[0, j], y_pos[0, i])

        if len(indices_1) > k:
            indices_1 = [x for _,x in sorted(zip(distances_1,indices_1))]
            for idx in indices_1[:k]:
                start_index.append(i)
                end_index.append(idx)
        else :
            for idx in indices_1:
                start_index.append(i)
                end_index.append(idx)
        
        if len(indices_2) > k:
            indices_2 = [x for _,x in sorted(zip(distances_2,indices_2))]
            for idx in indices_2[:k]:
                start_index.append(i)
                end_index.append(idx)
        else :
            for idx in indices_2:
                start_index.append(i)
                end_index.append(idx)

        if len(indices_3) > k:
            indices_3 = [x for _,x in sorted(zip(distances_3,indices_3))]
            for idx in indices_3[:k]:
                start_index.append(i)
                end_index.append(idx)
        else :
            for idx in indices_3:
                start_index.append(i)
                end_index.append(idx)

        if len(indices_4) > k:
            indices_4 = [x for _,x in sorted(zip(distances_4,indices_4))]
            for idx in indices_4[:k]:
                start_index.append(i)
                end_index.append(idx)
        else :
            for idx in indices_4:
                start_index.append(i)
                end_index.append(idx)
            
            

            




    # for i in range(n): # source node
    #     hit_in_quadrant = [False, False, False, False]
    #     # hits_in_quadrant_knn = [0, 0, 0, 0]
    #     for j in range(n): # target node
    #         if i == j : continue
    #         if (abs(x[i, 0] - x[j, 0]) > 0.20): continue # outside 5ns window
    #         if abs(x_pos[0, i] - x_pos[0, j]) > dist or abs(y_pos[0, i] - y_pos[0, j]) > dist: continue # outside dist x dist window   
    #         if (x_pos[0, j] >= x_pos[0, i]) and (y_pos[0, j] > y_pos[0, i]): # first quadrant
    #             if hit_in_quadrant[0]: 
    #                 continue
    #             else:
    #                 hit_in_quadrant[0] = True
    #         elif (x_pos[0, j] > x_pos[0, i]) and (y_pos[0, j] <= y_pos[0, i]): # second quadrant
    #             if hit_in_quadrant[1]: 
    #                 continue
    #             else:
    #                 hit_in_quadrant[1] = True
    #         elif (x_pos[0, j] <= x_pos[0, i]) and (y_pos[0, j] < y_pos[0, i]): # third quadrant
    #             if hit_in_quadrant[2]: 
    #                 continue
    #             else:
    #                 hit_in_quadrant[2] = True
    #         elif (x_pos[0, j] < x_pos[0, i]) and (y_pos[0, j] >= y_pos[0, i]): # fourth quadrant
    #             if hit_in_quadrant[3]: 
    #                 continue
    #             else:
    #                 hit_in_quadrant[3] = True
    #         else:
    #             print("Error no quadrant: ", x_pos[0, j], x_pos[0, i], y_pos[0, j], y_pos[0, i])

    #         start_index.append(i)
    #         end_index.append(j)

    
    
    
    # for i in range(n):
    #     for j in range(n):
    #         if i == j :
    #             continue
    #         if abs(x_pos[0, i]-x_pos[0, j]) <= dist and abs(y_pos[0, i]-y_pos[0, j]) <= dist :
    #             if -0.20 < (x[i, 0] - x[j, 0]) < 0.20 :
    #                 start_index.append(i)
    #                 end_index.append(j)
    
    start_index = np.asarray(start_index)
    end_index = np.asarray(end_index)
    edge_index = np.row_stack((start_index, end_index))
    edge_index = torch.from_numpy(edge_index).long()

    # Edge features
    # edge_features = np.zeros((edge_index.shape[1], 2))
    # if n > 1 :
    #     for i in range(edge_index.shape[1]):
    #         x0 = x[start_index[i], 1]
    #         y0 = x[start_index[i], 2]
    #         x1 = x[end_index[i], 1]
    #         y1 = x[end_index[i], 2]
    #         edge_features[i, 0] = math.sqrt(((x1-x0)*31.0)**2 + ((y1-y0)*71)**2)/(dist*1.41422)
    #         edge_features[i, 1] = abs(x[start_index[i], 0] - x[end_index[i], 0])
        
    # edge_features = torch.from_numpy(edge_features).float()

    # Labels
    y = np.zeros((n, 1))
    y[:, 0] = tar[index, :][hit_indices]


    return Data(x=torch.from_numpy(x).float(), edge_index=edge_index, y=torch.from_numpy(y).float())