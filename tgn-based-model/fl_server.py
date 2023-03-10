import socket
import pickle
import select
import time
import numpy as np
import pandas as pd
import sys
import logging
from timeit import default_timer as timer
import gc
import os
import math

####
import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from models.tgn.evaluation.evaluation import eval_edge_prediction
from models.tgn.model.tgn import TGN
from models.tgn.utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from models.tgn.utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)
###

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

############
### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
############
# arg_names = [
#     'path_weights',
#     'path_nodes',
#     'path_edges',
#     'graph_id',
#     'partition_id',
#     'num_clients',
#     'num_rounds',
#     'IP',
#     'PORT',
#     'name',
#     'transfer_learning',
#     'num_timestamps'
#     ]
#
# args = dict(zip(arg_names, sys.argv[1:]))
args = dict()
args['path_weights'] = './weights/'
args['path_nodes'] = './data/'
args['path_edges'] = './data/'
args['graph_id'] = '4'
args['partition_id'] = '0'
args['num_clients'] = '1'
args['initial_num_rounds'] = '6'
args['normal_num_rounds'] = '2'
args['IP'] = 'localhost'
args['PORT'] = '5000'
args['name'] = 'elliptic'
args['transfer_learning'] = True
args['num_timestamps'] = '6'
class Server:

    def __init__(self, MODEL, INITIAL_ROUNDS, NORMAL_ROUNDS, weights_path, graph_id, MAX_CONN = 2, IP= socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10,iteration_id=0, transfer_learning=False, NUM_TIMESTAMPS=0):

        # Parameters
        self.HEADER_LENGTH =  HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT
        self.MAX_CONN = MAX_CONN
        self.ROUNDS = INITIAL_ROUNDS
        self.NORMAL_ROUNDS = NORMAL_ROUNDS

        self.weights_path = weights_path
        self.graph_id = graph_id

        self.global_modlel_ready = False

        self.weights = []
        self.partition_sizes = []
        self.training_cycles = 0

        self.stop_flag = False

        # List of sockets for select.select()
        self.sockets_list = []
        self.clients = {}
        self.client_ids = {}

        # Craete server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.IP, self.PORT))
        self.server_socket.listen(self.MAX_CONN)

        self.sockets_list.append(self.server_socket)

        # Variables for dynamic graphs training
        self.iteration_id = iteration_id
        self.transfer_learning = transfer_learning
        self.NUM_TIMESTAMPS = NUM_TIMESTAMPS
        self.all_timestamps_finished = False

        # Global model
        self.GLOBAL_WEIGHTS = MODEL.state_dict()

    def send_iteration_id(self, client_socket):
        data = {"ITERATION_ID": self.iteration_id}
        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data

        client_socket.sendall(data)

    def update_model(self, new_weights, num_examples):
        self.partition_sizes.append(num_examples)
        self.weights.append(num_examples * new_weights)
        # self.weights.append(new_weights)

        if len(self.weights) == self.MAX_CONN:

            #avg_weight = np.mean(self.weights, axis=0)
            avg_weight = sum(self.weights) / sum(self.partition_sizes)
            self.weights = []

            self.partition_sizes = []

            #self.GLOBAL_MODEL.set_weights(new_weights)
            self.GLOBAL_WEIGHTS = avg_weight

            self.training_cycles += 1

            # weights file name : global_weights_graphid.npy
            weights_path = self.weights_path + 'weights_' + 'graphID:' + self.graph_id + "_V" + str(self.training_cycles) + ".npy"
            np.save(weights_path, avg_weight)

            for soc in self.sockets_list[1:]:
                self.send_model(soc)
            
            logging.info("___________________________________________________Iteration id %s, Training round %s done ______________________________________________________", self.iteration_id, self.training_cycles)
        

    def send_model(self, client_socket):

        if self.ROUNDS == self.training_cycles:
            self.stop_flag = True

        if self.NUM_TIMESTAMPS == self.iteration_id:
            self.all_timestamps_finished = True


        weights = np.array(self.GLOBAL_WEIGHTS)

        data = {"STOP_FLAG": self.stop_flag, "WEIGHTS": weights, "ITERATION_FLAG": self.all_timestamps_finished}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data

        client_socket.sendall(data)

        logging.info('Sent global model to client-%s at %s:%s', self.client_ids[client_socket], *self.clients[client_socket])


    def receive(self, client_socket):
        try:
            
            message_header = client_socket.recv(self.HEADER_LENGTH)

            if not len(message_header):
                logging.error('Client-%s closed connection at %s:%s',self.client_ids[client_socket], *self.clients[client_socket])
                return False

            message_length = int(message_header.decode('utf-8').strip())

            #full_msg = client_socket.recv(message_length)

            full_msg = b''
            while True:
                msg = client_socket.recv(message_length)

                full_msg += msg

                if len(full_msg) == message_length:
                    break
            
            return pickle.loads(full_msg)

        except Exception as e:
            logging.error('Client-%s closed connection at %s:%s',self.client_ids[client_socket], *self.clients[client_socket])
            return False


    def run(self):
        while not self.all_timestamps_finished:
            while not self.stop_flag:

                read_sockets, write_sockets, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)

                for notified_socket in read_sockets:

                    if notified_socket == self.server_socket:

                        client_socket, client_address = self.server_socket.accept()
                        self.sockets_list.append(client_socket)
                        self.clients[client_socket] = client_address
                        self.client_ids[client_socket] = "new"

                        logging.info('Accepted new connection at %s:%s', *client_address)

                        self.send_model(client_socket)

                    else:

                        message = self.receive(notified_socket)

                        if message is False:
                            self.sockets_list.remove(notified_socket)
                            del self.clients[notified_socket]
                            continue
                        else:
                            client_id = message['CLIENT_ID']
                            weights = message['WEIGHTS']
                            num_examples = message["NUM_EXAMPLES"]
                            self.client_ids[notified_socket] = client_id

                        logging.info('Recieved model from client-%s at %s:%s',client_id, *self.clients[notified_socket])
                        self.update_model(weights, int(num_examples))

                for notified_socket in exception_sockets:
                    self.sockets_list.remove(notified_socket)
                    del self.clients[notified_socket]

            # Save weights to use for transfer learning
            weights_path = "./global_weights/" + 'weights_' + 'graphID:' + self.graph_id + "_V" + str(self.iteration_id) + ".npy"
            np.save(weights_path, self.GLOBAL_WEIGHTS)

            self.iteration_id += 1
            self.weights = []
            self.partition_sizes = []
            self.training_cycles = 0
            self.stop_flag = False
            self.ROUNDS = self.NORMAL_ROUNDS

            # List of sockets for select.select()
            # self.sockets_list = []
            # self.clients = {}
            # self.client_ids = {}
            clients_connected = 0
            # while clients_connected != self.MAX_CONN:


if __name__ == "__main__":

    # from models.supervised import Model

    # Create weights folder
    folder_path = "weights"
    if os.path.exists(folder_path):
        logging.info("Folder path \"" + folder_path + "\" exists")
        pass
    else:
        logging.info("Weights folder created")
        os.makedirs(folder_path)

    # Create global weights folder
    folder_path = "global_weights"
    if os.path.exists(folder_path):
        logging.info("Folder path \"" + folder_path + "\" exists")
        pass
    else:
        logging.info("Weights folder created")
        os.makedirs(folder_path)



    logging.warning('####################################### New Training Session #######################################')
    logging.info('Server started , graph ID %s, number of clients %s, number of rounds %s, transfer learning %s, number of timestamps %s',args['graph_id'],args['num_clients'], args['initial_num_rounds'], args['transfer_learning'], args['num_timestamps'])

    if 'IP' not in args.keys() or args['IP'] == 'localhost':
        args['IP'] = socket.gethostname()

    if 'PORT' not in args.keys():
        args['PORT'] = 5000
    # for timestamp_id in range(1, 25):
    #     logging.warning('###################### New Interation (%s) started #############################', str(timestamp_id))

    # Model initialization to send weights
    #####################################################################################
    ### Extract data for training, validation and testing
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data = get_data(DATA,
                                  different_new_nodes_between_val_and_test=args.different_new_nodes,
                                  randomize_features=args.randomize_features)

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                          seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                           new_node_test_data.destinations,
                                           seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    # Initialize Model
    model = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep)
    #####################################################################################



    logging.info('Model initialized')

    server = Server(model, INITIAL_ROUNDS=int(args['initial_num_rounds']), NORMAL_ROUNDS=int(args['normal_num_rounds']), weights_path=args['path_weights'],graph_id=args['graph_id'],MAX_CONN=int(args['num_clients']),IP=args['IP'],PORT=int(args['PORT']),iteration_id=1,transfer_learning=args['transfer_learning'], NUM_TIMESTAMPS=int(args['num_timestamps']))

    del model
    del node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data
    del train_ngh_finder, full_ngh_finder, train_rand_sampler, val_rand_sampler, nn_val_rand_sampler, test_rand_sampler, nn_test_rand_sampler
    del device_string, device
    del mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
    gc.collect()

    logging.info('Federated training started!')

    start = timer()
    server.run()
    end = timer()

    elapsed_time = end - start
    logging.info('Federated training done!')
    logging.info('Training report : Elapsed time %s seconds, graph ID %s, number of clients %s, number of rounds %s',elapsed_time,args['graph_id'],args['num_clients'],args['normal_num_rounds'])
