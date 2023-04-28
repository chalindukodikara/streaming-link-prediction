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
import argparse
import os
import glob
import warnings
warnings.filterwarnings("ignore")


######## Our parameters ################
parser = argparse.ArgumentParser('Server')
parser.add_argument('--path_weights', type=str, default='./local_weights/', help='Weights path to be saved')
parser.add_argument('--path_nodes', type=str, default='./data/', help='Nodes path')
parser.add_argument('--path_edges', type=str, default='./data/', help='Edges Path')
parser.add_argument('--ip', type=str, default='localhost', help='IP')
parser.add_argument('--port', type=int, default=5000, help='PORT')
parser.add_argument('--graph_id', type=int, default=1, help='Graph ID')
parser.add_argument('--partition_id', type=int, default=0, help='Partition ID')

######## Frequently configured #######
parser.add_argument('--dataset_name', type=str, default='facebook', help='Dataset name')
parser.add_argument('--partition_size', type=int, default=2, help='Partition size')
parser.add_argument('--num_clients', type=int, default=2, help='Number of clients')
parser.add_argument('--partition_algorithm', type=str, default='hash', help='Partition algorithm')
parser.add_argument('--training_rounds', type=int, default=6, help='Initial Training: number of rounds')
parser.add_argument('--rounds', type=int, default=3, help='Streaming data testing for batches: number of rounds')


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
TRAINING_ROUNDS = args.training_rounds
ROUNDS = args.rounds
NUM_CLIENTS = args.num_clients
######## Our parameters ################

######## Setup logger ################
# create data folder with the dataset name
folder_path_logs = "logs"
if os.path.exists(folder_path_logs):
    pass
else:
    os.makedirs(folder_path_logs)

folder_path_process = folder_path_logs + "/server"
if os.path.exists(folder_path_process):
    pass
else:
    os.makedirs(folder_path_process)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('logs/server/{}_{}_{}_partition_{}.log'.format(str(time.strftime('%m %d %H:%M:%S # %l:%M%p on %b %d, %Y')), DATASET_NAME, PARTITION_ALGORITHM, PARTITION_SIZE)),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create weights folder
folder_path = "local_weights"
if os.path.exists(folder_path):
    pass
else:
    os.makedirs(folder_path)

# Create global weights folder
folder_path = "global_weights"
if os.path.exists(folder_path):
    pass
else:
    os.makedirs(folder_path)
############################################
# Find the minimum batch number in partitions
timestamps = []
for partition in range(PARTITION_SIZE):
    path = 'data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(partition)
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    timestamps.append(int(max(paths, key=os.path.getctime).split('/')[-1].split('_')[0]))
NUM_TIMESTAMPS = min(timestamps)

class Server:

    def __init__(self, MODEL, training_rounds, rounds, weights_path, graph_id, MAX_CONN = 2, IP= socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10, iteration_number=0, transfer_learning=False, num_timestamps=0):

        # Parameters
        self.HEADER_LENGTH = HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT
        self.MAX_CONN = MAX_CONN
        self.training_rounds = training_rounds
        self.rounds = rounds

        self.weights_path = weights_path
        self.graph_id = graph_id

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
        self.iteration_number = iteration_number
        self.transfer_learning = transfer_learning
        self.num_timestamps = num_timestamps
        self.all_timestamps_finished = False

        # Global model
        self.GLOBAL_WEIGHTS = MODEL.get_weights()

    def update_model(self, new_weights, num_examples):
        self.partition_sizes.append(num_examples)
        self.weights.append(num_examples * new_weights)

        # if len(self.weights) == self.MAX_CONN:
        if (len(self.weights) % self.MAX_CONN) == 0 and len(self.weights) != 0:
            avg_weight = sum(self.weights) / sum(self.partition_sizes)
            self.weights = []

            self.partition_sizes = []

            #self.GLOBAL_MODEL.set_weights(new_weights)
            self.GLOBAL_WEIGHTS = avg_weight

            self.training_cycles += 1

            # weights file name : global_weights_graphid.npy
            weights_path = self.weights_path + 'weights_' + 'graphID:' + str(self.graph_id) + "_V" + str(self.training_cycles) + ".npy"
            np.save(weights_path, avg_weight)

            for soc in self.sockets_list[1:]:
                self.send_model(soc)
            if self.iteration_number != 0:
                logging.info(
                    "____________________________________ Batch %s: round %s finished ____________________________________",
                    self.iteration_number, self.training_cycles)
            elif self.iteration_number == 0:
                logging.info("____________________________________ Initial training: round %s finished ____________________________________", self.training_cycles)


    def send_model(self, client_socket):

        if self.training_rounds == self.training_cycles:
            self.stop_flag = True

        if self.num_timestamps == self.iteration_number:
            self.all_timestamps_finished = True


        weights = np.array(self.GLOBAL_WEIGHTS)
        if self.iteration_number != 0:
            data = {"STOP_FLAG": self.stop_flag, "WEIGHTS": weights, "ITERATION_FLAG": self.all_timestamps_finished}
        else:
            data = {"STOP_FLAG": self.stop_flag, "WEIGHTS": weights, "ITERATION_FLAG": self.all_timestamps_finished, "NUM_TIMESTAMPS": self.num_timestamps}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data

        client_socket.sendall(data)
        if self.iteration_number != 0:
            logging.info('Batch %s, round %s: aggregated global model sent to client-%s at %s:%s',
                         self.iteration_number, self.training_cycles, self.client_ids[client_socket], *self.clients[client_socket])
        elif self.iteration_number == 0 and self.training_cycles != 0:
            logging.info('Initial training round %s: aggregated global model sent to client-%s at %s:%s', self.training_cycles, self.client_ids[client_socket], *self.clients[client_socket])
        elif self.iteration_number == 0 and self.training_cycles == 0:
            logging.info('Randomly initialized global model sent to client-%s at %s:%s', self.client_ids[client_socket], *self.clients[client_socket])


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
        while self.iteration_number == 0:
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

                        if self.iteration_number != 0:
                            logging.info('Batch %s: recieved model from client-%s at %s:%s', self.iteration_number, client_id, *self.clients[notified_socket])
                        elif self.iteration_number == 0:
                            logging.info('Initial training: recieved model from client-%s at %s:%s', client_id, *self.clients[notified_socket])
                        self.update_model(weights, int(num_examples))

                for notified_socket in exception_sockets:
                    self.sockets_list.remove(notified_socket)
                    del self.clients[notified_socket]
            if self.iteration_number == 0:
                logging.info("#################################### Initial Trained final model sent to clients ####################################")
            else:
                logging.info("#################################### Batch %s: sent the final model to clients ####################################", self.iteration_number)
            # Save weights to use for transfer learning
            weights_path = "./global_weights/" + 'weights_' + 'dataset:' + str(DATASET_NAME) + "_V" + str(self.iteration_number) + ".npy"
            np.save(weights_path, self.GLOBAL_WEIGHTS)

            self.iteration_number += 1
            self.weights = []
            self.partition_sizes = []
            self.training_cycles = 0
            self.stop_flag = False
            # set normal number of rounds to the training rounds
            self.training_rounds = self.rounds

        num_results = 0

        accuracy_list = []
        accuracy_std_list = []
        accuracy_99th_list = []
        accuracy_90th_list = []

        recall_list = []
        recall_std_list = []
        recall_99th_list = []
        recall_90th_list = []

        auc_list = []
        auc_std_list = []
        auc_99th_list = []
        auc_90th_list = []

        f1_list = []
        f1_std_list = []
        f1_99th_list = []
        f1_90th_list = []

        precision_list = []
        precision_std_list = []
        precision_99th_list = []
        precision_90th_list = []

        mean_time_list = []
        mean_time_std_list = []
        mean_time_99th_list = []
        mean_time_90th_list = []

        total_time_list = []

        while num_results != int(NUM_CLIENTS):
            read_sockets, write_sockets, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)

            for notified_socket in read_sockets:

                message = self.receive(notified_socket)

                if message is False:
                    self.sockets_list.remove(notified_socket)
                    del self.clients[notified_socket]
                    continue
                else:
                    accuracy, accuracy_std, accuracy_99th, accuracy_90th = message['ACCURACY'], message['ACCURACY_STD'], \
                                                                           message['ACCURACY_99TH'], message[
                                                                               'ACCURACY_90TH']
                    recall, recall_std, recall_99th, recall_90th = message['RECALL'], message['RECALL_STD'], message[
                        'RECALL_99TH'], message['RECALL_90TH']
                    auc, auc_std, auc_99th, auc_90th = message['AUC'], message['AUC_STD'], message['AUC_99TH'], message[
                        'AUC_90TH']
                    f1, f1_std, f1_99th, f1_90th = message['F1'], message['F1_STD'], message['F1_99TH'], message[
                        'F1_90TH']
                    precision, precision_std, precision_99th, precision_90th = message['PRECISION'], message[
                        'PRECISION_STD'], message['PRECISION_99TH'], message['PRECISION_90TH']
                    mean_time, mean_time_std, mean_time_99th, mean_time_90th = message['MEAN_TIME'], message[
                        'MEAN_TIME_STD'], message['MEAN_TIME_99TH'], message['MEAN_TIME_90TH']
                    total_time = message['TOTAL_TIME']

                    accuracy_list.append(float(accuracy))
                    accuracy_std_list.append(float(accuracy_std))
                    accuracy_99th_list.append(float(accuracy_99th))
                    accuracy_90th_list.append(float(accuracy_90th))

                    recall_list.append(float(recall))
                    recall_std_list.append(float(recall_std))
                    recall_99th_list.append(float(recall_99th))
                    recall_90th_list.append(float(recall_90th))

                    auc_list.append(float(auc))
                    auc_std_list.append(float(auc_std))
                    auc_99th_list.append(float(auc_99th))
                    auc_90th_list.append(float(auc_90th))

                    f1_list.append(float(f1))
                    f1_std_list.append(float(f1_std))
                    f1_99th_list.append(float(f1_99th))
                    f1_90th_list.append(float(f1_90th))

                    precision_list.append(float(precision))
                    precision_std_list.append(float(precision_std))
                    precision_99th_list.append(float(precision_99th))
                    precision_90th_list.append(float(precision_90th))

                    mean_time_list.append(float(mean_time))
                    mean_time_std_list.append(float(mean_time_std))
                    mean_time_99th_list.append(float(mean_time_99th))
                    mean_time_90th_list.append(float(mean_time_90th))

                    total_time_list.append(float(total_time))

                    self.client_ids[notified_socket] = client_id

                    num_results += 1

            for notified_socket in exception_sockets:
                self.sockets_list.remove(notified_socket)
                del self.clients[notified_socket]

        logging.info(
            "______________________________________________________________________________________________________ Final Mean Values of all Clients ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")

        logging.info(
            'Result report : Accuracy - %s (%s), Recall - %s (%s), AUC - %s (%s), F1 - %s (%s), Precision - %s (%s), Mean time for a batch - %s (%s) seconds, Total Time - %s (%s)',
            str(round(np.mean(accuracy_list), 4)), str(round(np.mean(accuracy_std_list), 4)),
            str(round(np.mean(recall_list), 4)), str(round(np.mean(recall_std_list), 4)),
            str(round(np.mean(auc_list), 4)), str(round(np.mean(auc_std_list), 4)),
            str(round(np.mean(f1_list), 4)), str(round(np.mean(f1_std_list), 4)),
            str(round(np.mean(precision_list), 4)), str(round(np.mean(precision_std_list), 4)),
            str(round(np.mean(mean_time_list), 4)), str(round(np.mean(mean_time_std_list), 4)),
            str(round(np.mean(total_time_list), 4)), str(round(np.std(total_time_list), 4))
        )
        logging.info(
            'Result report : Accuracy 99th - 90th (%s, %s), Recall 99th - 90th (%s, %s), AUC 99th - 90th (%s, %s), F1 99th - 90th (%s, %s), Precision 99th - 90th (%s, %s), Mean time for a batch 99th - 90th (%s, %s)',
            str(round(np.mean(accuracy_99th_list), 4)), str(round(np.mean(accuracy_90th_list), 4)),
            str(round(np.mean(recall_99th_list), 4)), str(round(np.mean(recall_90th_list), 4)),
            str(round(np.mean(auc_99th_list), 4)), str(round(np.mean(auc_90th_list), 4)),
            str(round(np.mean(f1_99th_list), 4)), str(round(np.mean(f1_90th_list), 4)),
            str(round(np.mean(precision_99th_list), 4)), str(round(np.mean(precision_90th_list), 4)),
            str(round(np.mean(mean_time_99th_list), 4)), str(round(np.mean(mean_time_90th_list), 4))
        )
        logging.info(
            "______________________________________________________________________________________________________ Final Mean Values of all Clients ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")

        logging.info('Total time list: ' + str(total_time_list))
        logging.info('Accuracy list: ' + str(accuracy_list))
        logging.info('Recall list: ' + str(recall_list))
        logging.info('AUC list: ' + str(auc_list))
        logging.info('F1 list: ' + str(f1_list))
        logging.info('Precision list: ' + str(precision_list))

if __name__ == "__main__":

    logging.warning('####################################### New Training Session #######################################')
    logging.info('Server started , graph ID %s, number of clients %s, number of rounds %s, number of timestamps %s', GRAPH_ID, NUM_CLIENTS, TRAINING_ROUNDS, NUM_TIMESTAMPS)

    if IP == 'localhost':
        IP = socket.gethostname()

    edges = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(0) + '_training_batch_edges.csv')
    nodes = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(0) + '_training_batch_nodes.csv', index_col=0)

    from models.supervised import Model

    model = Model(nodes, edges)
    model.initialize()

    server = Server(model, training_rounds=TRAINING_ROUNDS, rounds=ROUNDS, weights_path=WEIGHTS_PATH, graph_id=GRAPH_ID, MAX_CONN=NUM_CLIENTS, IP=IP, PORT=PORT, iteration_number=0, transfer_learning=True, num_timestamps=NUM_TIMESTAMPS)

    del nodes
    del edges
    del model
    del paths
    del files
    gc.collect()

    logging.info('Distributed training for streaming graphs started!')

    start = timer()
    server.run()
    end = timer()

    elapsed_time = end - start

    logging.info(
        "______________________________________________________________________________________________________ Final Values ______________________________________________________________________________________________________")
    logging.info(
        "##########################################################################################################################################################################################################################")

    logging.info('Distributed training done!')
    logging.info('Training report : Total elapsed time %s seconds, graph ID %s, number of clients %s, training rounds %s, rounds %s, number of timestamps %s', round(elapsed_time, 0), GRAPH_ID, NUM_CLIENTS, TRAINING_ROUNDS, ROUNDS, NUM_TIMESTAMPS)
