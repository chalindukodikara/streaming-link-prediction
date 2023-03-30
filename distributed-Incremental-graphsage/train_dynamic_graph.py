import numpy as np
import pandas as pd
import gc
import os
import datetime
# pd.to_datetime(int(data.iloc[2, 3]), unit='s')

def main():
    name = 'elliptic'
    for i in range(1, 25):
        data_edges = pd.read_csv('data/'+name+'/'+str(i)+'_edges.csv')
        data_nodes = pd.read_csv('data/' + name + '/' + str(i) + '_nodes.csv')



if __name__ == "__main__":
    main()