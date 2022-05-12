import ast
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn


with open('data/17885.pdb', 'r') as pdb:
    matrix= []
    while 1:
        temp = pdb.readline()
        temp = temp.split()
        if temp[0] == 'END':
            break
        try:
            temp_list = [ast.literal_eval(i) for i in temp[6:9]]
            matrix.append(temp_list)
        except Exception as e:
            print('TER')


cnt_dict = {'(hbond:mc_mc)': 0, '(hbond:sc_sc)': 1, '(cnt:mc_sc)': 2, '(cnt:mc_mc)': 3, '(hbond:mc_sc)': 4, '(combi:all_all)': 5, '(cnt:sc_sc)': 6}
cnt_num = len(cnt_dict)
edges_dict = {}

with open('data/17885.intsc', 'r') as intsc:
    # cnt_list = []

    while 1:
        temp = intsc.readline()
        temp = temp.split()
        if not temp:
            break
        try:
            s = int(temp[0].split(':')[1])
            t = int(temp[2].split(':')[1])
        except IndexError:
            continue

        cnt_name = temp[1]

        # print(temp)
        # cnt = temp[1]
        # cnt_list.append(cnt)

        key_name = str([s, t])

        if key_name not in edges_dict.keys():

            edges_dict[key_name] = [0 for i in range(2 * cnt_num)]

            index = cnt_dict[cnt_name]

            edges_dict[key_name][index] = ast.literal_eval(temp[-1])

        else:
            index = cnt_dict[cnt_name]

            edges_dict[key_name][index] = ast.literal_eval(temp[-1])

    # print(set(cnt_list))


with open('data/17885.nrint', 'r') as nrint:
    while 1:
        temp = nrint.readline()
        temp = temp.split()
        if not temp:
            break

        try:
            s = int(temp[0].split(':')[1])
            t = int(temp[2].split(':')[1])
            pass
        except IndexError:
            continue

        cnt_name = temp[1]

        key_name = str([s, t])

        if key_name not in edges_dict.keys():

            raise IndexError(f'Edge not match, new edge appear: {key_name}')
        else:
            index = cnt_dict[cnt_name]

            edges_dict[key_name][index + cnt_num] = ast.literal_eval(temp[-1])

node_num = len(matrix)-1
nodes_set = tfgnn.NodeSet.from_fields(
    sizes=[node_num],
    features={'node_features': np.asarray(matrix[0:-1])}
)

adjacency = np.asarray([ast.literal_eval(i) for i in edges_dict.keys()])
edge_num = len(edges_dict)
edges_set = tfgnn.EdgeSet.from_fields(
    sizes=[edge_num],
    features={'edge_features': np.asarray([i for i in edges_dict.values()])},
    adjacency=tfgnn.Adjacency.from_indices(
        source=('source', adjacency[:, 0]),
        target=('target', adjacency[:, 1])
    )
)

graph_tensor = tfgnn.GraphTensor.from_pieces(
    edge_sets={'edges': edges_set},
    node_sets={'nodes': nodes_set}
)

tensor_spec = graph_tensor.spec

inputs = tf.keras.layers.Input(type_spec=tensor_spec)

gnn = tfgnn.keras.ConvGNNBuilder(
    lambda edge: tfgnn.keras.layers.SimpleConvolution(tf.keras.layers.Dense(16)),

    lambda node: tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(16)),
)



pass