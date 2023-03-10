import numpy as np
import pandas as pd
import gc
import os
import datetime
# pd.to_datetime(int(data.iloc[2, 3]), unit='s')

def main():
    dataset_name = 'epinion'
    number_of_timestamps = 11
    sliding_window_size = 5


    folder_path = "data/" + dataset_name
    if os.path.exists(folder_path):
        print("Folder path \"" + folder_path + "\" exists")
        pass
    else:
        os.makedirs(folder_path)

    data_edges = pd.read_csv('data/' + dataset_name + '_edges.csv')
    if 'Unnamed: 0' in data_edges: data_edges.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in data_edges: data_edges.pop('Unnamed: 0.1')

    data_nodes = pd.read_csv('data/' + dataset_name + '_nodes.csv')
    if 'Unnamed: 0' in data_nodes: data_nodes.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in data_nodes: data_nodes.pop('Unnamed: 0.1')


    for i in range(sliding_window_size, number_of_timestamps):
        print(i)
        if i == sliding_window_size:
            data_edges_temp = data_edges.loc[data_edges['timestamp'] < i+1].loc[data_edges['timestamp']>(i-sliding_window_size)]
        else:
            data_edges_temp = data_edges.loc[data_edges['timestamp'] < i + 1].loc[
                data_edges['timestamp'] > (i-1)]
        nodes_list = []
        for j in range(2):
            nodes_list = nodes_list + data_edges_temp[data_edges_temp.columns[j]].tolist()
        nodes_set = set(nodes_list)
        data_nodes_temp = data_nodes[data_nodes[data_nodes.columns[0]].isin(nodes_set)]
        # nodes_not_in = data_nodes[~data_nodes[data_nodes.columns[0]].isin(nodes_set)][data_nodes.columns[0]].to_list()
        # nodes_not_in_indexes = []
        # for j in range(data_edges_temp.shape[0]):
        #     if (data_edges_temp[data_edges_temp.columns[0]].iloc[j] in nodes_not_in) or (data_edges_temp[data_edges_temp.columns[1]].iloc[j] in nodes_not_in):
        #         nodes_not_in_indexes.append(j)
        # data_edges_temp = data_edges_temp.drop(nodes_not_in_indexes, axis=0)

        del nodes_list
        del nodes_set
        # del nodes_not_in
        # del nodes_not_in_indexes
        gc.collect()
        data_edges_temp = data_edges_temp.iloc[:, : 2]
        nodes_list = data_nodes_temp[data_nodes_temp.columns[0]].to_list()
        # for j in range(2):
        #     for k in range(data_edges_temp.shape[0]):
        #         if data_edges_temp[data_edges_temp.columns[j]].iloc[k] in nodes_list:
        #             pass
        #         else:
        #             print("True", j, k, data_edges_temp[data_edges_temp.columns[j]].iloc[k])
        data_edges_temp.to_csv("data/" + dataset_name + "/" + str(i+1 - sliding_window_size) + "_edges.csv", index=False)
        data_nodes_temp.to_csv("data/" + dataset_name + "/" + str(i + 1 - sliding_window_size) + "_nodes.csv", index=False)



    # df = pd.DataFrame(columns=['source', 'target', 'timestamp'])
    # df_nodes = pd.DataFrame(columns=['0'])
    #
    # node_list = []
    # for i in range(data.shape[0]):



if __name__ == "__main__":
    main()