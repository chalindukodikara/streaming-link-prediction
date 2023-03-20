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
from models.supervised import Model
import argparse
import os

######## Our parameters ################
parser = argparse.ArgumentParser('Client')
parser.add_argument('--path_weights', type=str, default='./weights/', help='Weights path to be saved')
parser.add_argument('--path_nodes', type=str, default='./data/', help='Nodes path')
parser.add_argument('--path_edges', type=str, default='./data/', help='Edges Path')
parser.add_argument('--ip', type=str, default='localhost', help='IP')
parser.add_argument('--port', type=int, default=5000, help='PORT')
parser.add_argument('--dataset_name', type=str, default='tg', help='Dataset name')
parser.add_argument('--graph_id', type=int, default=1, help='Graph ID')
parser.add_argument('--partition_id', type=int, default=0, help='Partition ID')
parser.add_argument('--training_epochs', type=int, default=5, help='Initial Training: number of epochs')
parser.add_argument('--epochs', type=int, default=2, help='Streaming data training for batches: number of epochs')

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
TRAINING_EPOCHS = args.training_epochs
EPOCHS = args.epochs
######## Our parameters ################

######## Setup logger ################
# create data folder with the dataset name
folder_path_logs = "logs"
if os.path.exists(folder_path_logs):
    pass
else:
    os.makedirs(folder_path_logs)

folder_path_process = folder_path_logs + "/client"
if os.path.exists(folder_path_process):
    pass
else:
    os.makedirs(folder_path_process)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('logs/client/client_{}.log'.format(PARTITION_ID)),
        logging.StreamHandler(sys.stdout)
    ]
)
############################################
class Client:

    def __init__(self, MODEL, graph_params, weights_path, dataset_name, graph_id, partition_id, training_epochs=30, epochs = 2, IP = socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10, iteration_id=1):

        self.HEADER_LENGTH = HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT

        self.weights_path = weights_path
        self.graph_id = graph_id
        self.partition_id = partition_id

        # Initial training is bit large
        self.training_epochs = training_epochs
        self.epochs = epochs
        self.epochs = training_epochs

        self.graph_params = graph_params

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.MODEL = MODEL
        self.STOP_FLAG = False
        self.rounds = 0
        self.iteration_id = iteration_id
        self.ITERATION_FLAG = False
        self.dataset_name = dataset_name
        self.GLOBAL_WEIGHTS = None

        connected = False
        while not connected:
            try:
                self.client_socket.connect((IP, PORT))
            except ConnectionRefusedError:
                time.sleep(5)
            else:
                logging.info('(Iteration id %s) Connected to the server', self.iteration_id)
                connected = True

    def send_model(self):

        # svae model weights
        # weights file name : weights_graphid_workerid.npy
        weights_path = self.weights_path + 'weights_' + self.graph_id + '_' + self.partition_id + ".npy"
        
        #np.save(weights_path,self.MODEL.get_weights())

        weights = np.array(self.MODEL.get_weights())

        data = {"CLIENT_ID": self.partition_id,"WEIGHTS":weights,"NUM_EXAMPLES":self.graph_params[0]}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data
        self.client_socket.sendall(data)


    def receive(self):
        try:

            message_header = self.client_socket.recv(self.HEADER_LENGTH)
            if not len(message_header):
                return False

            message_length = int(message_header.decode('utf-8').strip())

            full_msg = b''
            while True:
                msg = self.client_socket.recv(message_length)

                full_msg += msg

                if len(full_msg) == message_length:
                    break
            # logging.info(full_msg)
            data = pickle.loads(full_msg)
            # logging.info(data)
            self.STOP_FLAG = data["STOP_FLAG"]
            self.ITERATION_FLAG = data["ITERATION_FLAG"]

            return data["WEIGHTS"]

        except Exception as e:
            print(e)


    def fetch_model(self):
        data = self.receive()
        # logging.info("data", data, type(data))
        self.MODEL.set_weights(data)
        self.GLOBAL_WEIGHTS = data

    def train(self):
        self.MODEL.fit(epochs = self.epochs)

    def run(self):
        while not self.ITERATION_FLAG:
            while not self.STOP_FLAG:
                if self.iteration_id > 1 and self.rounds == 0:
                    self.MODEL.set_weights(self.GLOBAL_WEIGHTS)
                    print('set global weights :', self.iteration_id)
                    pass
                else:
                    read_sockets, _, exception_sockets = select.select([self.client_socket], [], [self.client_socket])

                    for soc in read_sockets:
                        self.fetch_model()


                if self.STOP_FLAG:
                    eval = self.MODEL.evaluate()

                    try:
                        f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
                        f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])
                    except ZeroDivisionError as e:
                        f1_train = "undefined"
                        f1_test = "undefined"

                    logging.info('_____________________________________________________ (Iteration id %s) Final model evalution ____________________________________________________________', self.iteration_id)
                    logging.info('(Iteration id %s) Final model (v%s) fetched', self.iteration_id, self.rounds)
                    logging.info('(Iteration id %s) Training set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_id, [0][0], eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                    logging.info('(Iteration id %s) Testing set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',  self.iteration_id, [1][0], eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])
                else:

                    self.rounds += 1
                    logging.info('_____________________________________________________ (Iteration id %s) Training Round ____________________________________________________________',self.rounds, self.iteration_id)
                    logging.info('(Iteration id %s) Global model v%s fetched', self.iteration_id, self.rounds - 1)

                    eval = self.MODEL.evaluate()

                    try:
                        f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
                        f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])
                    except ZeroDivisionError as e:
                        f1_train = "undefined"
                        f1_test = "undefined"

                    logging.info('(Iteration id %s) Global model v%s - Training set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_id, self.rounds - 1, eval[0][0], eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                    logging.info('(Iteration id %s) Global model v%s - Testing set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_id, self.rounds - 1,  eval[1][0], eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])


                    logging.info('(Iteration id %s) Training started', self.iteration_id)
                    self.train()
                    logging.info('(Iteration id %s) Training done', self.iteration_id)

                    # eval = self.MODEL.evaluate()

                    # f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
                    # f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])
                    # logging.info('After Round %s - Local model - Training set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',self.rounds, eval[0][0], eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                    # logging.info('After Round %s - Local model - Testing set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',self.rounds, eval[1][0], eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])

                    logging.info('(Iteration id %s) Sent local model to the server', self.iteration_id)
                    self.send_model()

            self.STOP_FLAG = False
            self.rounds = 0
            self.iteration_id += 1
            self.epochs = self.epochs
            # self.client_socket.close()

            edges = pd.read_csv('data/' + self.dataset_name + '/' + str(self.iteration_id) + '_edges.csv')
            nodes = pd.read_csv('data/' + self.dataset_name + '/' + str(self.iteration_id) + '_nodes.csv', index_col=0)

            logging.info('(Iteration id %s) Model initialized ', str(self.iteration_id))
            self.MODEL = Model(nodes, edges)
            num_train_ex, num_test_ex = self.MODEL.initialize()

            self.graph_params = (num_train_ex, num_test_ex)

            del nodes
            del edges
            gc.collect()




if __name__ == "__main__":

    if IP == 'localhost':
        IP = socket.gethostname()


    logging.warning('####################################### New Training Session #######################################')
    logging.info('Client started, graph ID %s, partition ID %s, epochs %s',args['graph_id'],args['partition_id'],args['initial_epochs'])


    edges = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_ID) + '/' + str(0) + '_training_batch_edges.csv')
    nodes = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_ID) + '/' + str(0) + '_training_batch_nodes.csv', index_col=0)

    logging.info('Model initialized for training')
    model = Model(nodes, edges)
    num_train_ex, num_test_ex = model.initialize()

    graph_params = (num_train_ex, num_test_ex)

    logging.info('(Iteration id %s) Number of training examples - %s, Number of testing examples %s', str(1), num_train_ex,num_test_ex)
    client = Client(model, graph_params, weights_path=WEIGHTS_PATH, dataset_name=DATASET_NAME, graph_id=GRAPH_ID, partition_id=PARTITION_ID, training_epochs = TRAINING_EPOCHS, epochs = EPOCHS ,IP=IP,PORT=PORT, iteration_id=1)

    del nodes
    del edges
    gc.collect()

    logging.info('(Iteration id %s) Distributed training started!', str(1))

    start = timer()
    client.run()
    end = timer()

    elapsed_time = end -start
    logging.info('Distributed training done!')
    logging.info('Training report : Elapsed time %s seconds, graph ID %s, partition ID %s, epochs %s', elapsed_time,args['graph_id'],args['partition_id'],args['epochs'])

    
