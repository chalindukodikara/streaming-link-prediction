import socket
import pickle
import select
import sys
import numpy as np
import pandas as pd
import logging
from timeit import default_timer as timer
import time
import gc
import argparse
import os
# import warnings
# warnings.filterwarnings("ignore")

######## Our parameters ################
parser = argparse.ArgumentParser('Client')
parser.add_argument('--path_weights', type=str, default='./local_weights/', help='Weights path to be saved')
parser.add_argument('--path_nodes', type=str, default='./data/', help='Nodes path')
parser.add_argument('--path_edges', type=str, default='./data/', help='Edges Path')
parser.add_argument('--ip', type=str, default='localhost', help='IP')
parser.add_argument('--port', type=int, default=5000, help='PORT')

######## Frequently configured #######
parser.add_argument('--dataset_name', type=str, default='wikipedia', help='Dataset name')
parser.add_argument('--graph_id', type=int, default=1, help='Graph ID')
parser.add_argument('--partition_id', type=int, default=0, help='Partition ID')
parser.add_argument('--partition_size', type=int, default=1, help='Partition size')
parser.add_argument('--partition_algorithm', type=str, default='hash', help='Partition algorithm')
parser.add_argument('--training_epochs', type=int, default=10, help='Initial Training: number of epochs')
parser.add_argument('--epochs', type=int, default=4, help='Streaming data training for batches: number of epochs')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

WEIGHTS_PATH = args.path_weights
NODES_PATH = args.path_nodes
EDGES_PATH = args.path_edges
IP = args.ip
PORT = args.port
DATASET_NAME = args.dataset_name
GRAPH_ID = args.graph_id
PARTITION_ID = args.partition_id
PARTITION_SIZE = args.partition_size
PARTITION_ALGORITHM = args.partition_algorithm
TRAINING_EPOCHS = args.training_epochs
EPOCHS = args.epochs
######## Our parameters ################

######## Setup logger ################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('logs/test/test_{}.log'.format(str(time.strftime('%l:%M%p on %b %d, %Y')))),
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    from models.supervised import Model

    if IP == 'localhost':
        IP = socket.gethostname()


    logging.warning('####################################### New Training Session: Client %s #######################################', PARTITION_ID)
    logging.info('Client started, graph name %s, graph ID %s, partition ID %s, training epochs %s, epochs %s', DATASET_NAME, GRAPH_ID, PARTITION_ID, TRAINING_EPOCHS, EPOCHS)

