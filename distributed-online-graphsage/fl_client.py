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
import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument('--partition_size', type=int, default=2, help='Partition size')
parser.add_argument('--partition_algorithm', type=str, default='hash', help='Partition algorithm')
parser.add_argument('--training_epochs', type=int, default=3, help='Initial Training: number of epochs')
parser.add_argument('--epochs', type=int, default=3, help='Streaming data training for batches: number of epochs')

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
        logging.FileHandler('logs/client/{}_{}_{}_partition_{}_client_{}.log'.format(str(time.strftime('%m %d %H:%M:%S # %l:%M%p on %b %d, %Y')), DATASET_NAME, PARTITION_ALGORITHM, PARTITION_SIZE, PARTITION_ID)),
        logging.StreamHandler(sys.stdout)
    ]
)
############################################
class Client:

    def __init__(self, MODEL, graph_params, weights_path, dataset_name, graph_id, partition_id, training_epochs=30, epochs = 2, IP = socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10, iteration_number=1):

        self.HEADER_LENGTH = HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT

        self.weights_path = weights_path
        self.graph_id = graph_id
        self.partition_id = partition_id

        # Initial training is bit large
        self.testing_epochs = epochs
        self.epochs = training_epochs

        self.graph_params = graph_params

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.MODEL = MODEL
        self.STOP_FLAG = False
        self.rounds = 0
        self.iteration_number = iteration_number
        self.ITERATION_FLAG = False
        self.dataset_name = dataset_name
        self.GLOBAL_WEIGHTS = None
        self.all_test_metric_values = [[], [], [], [], [], []]
        self.NUM_TIMESTAMPS = 0

        connected = False
        while not connected:
            try:
                self.client_socket.connect((IP, PORT))
            except ConnectionRefusedError:
                time.sleep(5)
            else:
                logging.info('Connected to the server')
                connected = True

    def send_model(self):

        # svae model weights
        # weights file name : weights_graphid_workerid.npy
        weights_path = self.weights_path + 'weights_' + str(self.dataset_name) + '_client:' + str(self.partition_id) + "_R" + str(self.rounds) + ".npy"
        
        np.save(weights_path, self.MODEL.get_weights())

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
            if self.iteration_number == 0:
                self.NUM_TIMESTAMPS = data["NUM_TIMESTAMPS"]

            return data["WEIGHTS"]

        except Exception as e:
            print(e)


    def fetch_model(self):
        data = self.receive()
        # logging.info("data", data, type(data))
        logging.info('------------------------- Received aggregated global model from the server -------------------------')
        self.MODEL.set_weights(data)
        self.GLOBAL_WEIGHTS = data

    def train(self):
        return self.MODEL.fit(epochs=self.epochs)

    def run(self):
        start_time = timer()
        while not self.ITERATION_FLAG:
            while not self.STOP_FLAG:
                if self.iteration_number > 0 and self.rounds == 0:
                    self.MODEL.set_weights(self.GLOBAL_WEIGHTS)
                    if self.iteration_number == 1:
                        logging.info('################################## Next batch processing started: transfer learning is ON ##################################')
                else:
                    read_sockets, _, exception_sockets = select.select([self.client_socket], [], [self.client_socket])

                    for soc in read_sockets:
                        self.fetch_model()

                if self.iteration_number == 0 and self.rounds == 0:
                    training_start_time = timer()
                    logging.info('################################## Initial model training started ##################################')
                elif self.rounds == 0:
                    testing_start_time = timer()

                if self.STOP_FLAG:
                    if self.iteration_number == 0:
                        training_end_time = timer()
                    else:
                        testing_end_time = timer()


                    if self.iteration_number == 0:
                        logging.info(
                            '################ Initial trained model: Final global model evalution after %s rounds ################', self.rounds)

                        eval = self.MODEL.evaluate()

                        try:
                            f1_train = round((2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4]), 2)
                            f1_test = round((2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4]), 2)
                        except ZeroDivisionError as e:
                            f1_train = "undefined"
                            f1_test = "undefined"

                        logging.info(
                            'Initially trained model: Training set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s, training time - %s seconds',
                            round(eval[0][0], 2), round(eval[0][1], 4), round(eval[0][2], 4), round(eval[0][3], 4), f1_train, round(eval[0][4], 4), round(training_start_time - training_end_time, 0))
                        logging.info(
                            'Initially trained model: Testing set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',
                            round(eval[1][0], 2), round(eval[1][1], 4), round(eval[1][2], 4), round(eval[1][3], 4), f1_test, round(eval[1][4], 4))

                    else:
                        logging.info('Batch number %s model fetched from the server', self.iteration_number)
                        logging.info('################ Batch %s: final global model evalution after %s rounds ################', self.iteration_number, self.rounds)

                        eval = self.MODEL.evaluate()

                        try:
                            f1_train = round((2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4]), 4)
                            f1_test = round((2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4]), 4)
                        except ZeroDivisionError as e:
                            f1_train = "undefined"
                            f1_test = "undefined"

                        logging.info(
                            'Batch %s: Training set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s, training time - %s seconds',
                            self.iteration_number, round(eval[0][0], 4), round(eval[0][1], 4), round(eval[0][2], 4), round(eval[0][3], 4), f1_train, round(eval[0][4], 4), round(testing_start_time - testing_end_time, 0))
                        logging.info(
                            'Batch %s: Testing set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',
                            self.iteration_number, round(eval[1][0], 4), round(eval[1][1], 4), round(eval[1][2], 4), round(eval[1][3], 4), f1_test, round(eval[1][4], 4))

                        self.all_test_metric_values[0].append(round(eval[1][1], 4)) # accuracy
                        self.all_test_metric_values[1].append(round(eval[1][2], 4)) # recall
                        self.all_test_metric_values[2].append(round(eval[1][3], 4)) # auc
                        self.all_test_metric_values[3].append(f1_test) # f1
                        self.all_test_metric_values[4].append(round(eval[1][4], 4)) # precision
                        self.all_test_metric_values[5].append(round(testing_end_time - testing_start_time, 1))  # time

                else:
                    self.rounds += 1

                    # eval = self.MODEL.evaluate()
                    # try:
                    #     f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
                    #     f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])
                    # except ZeroDivisionError as e:
                    #     f1_train = "undefined"
                    #     f1_test = "undefined"
                    #
                    # logging.info('(Iteration id %s) Global model v%s - Training set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_number, self.rounds - 1, eval[0][0], eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                    # logging.info('(Iteration id %s) Global model v%s - Testing set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_number, self.rounds - 1,  eval[1][0], eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])

                    if self.iteration_number == 0:
                        logging.info('------------------------- Initial model training: round %s -------------------------', self.rounds)
                    else:
                        logging.info('------------------------- Batch %s training: round %s -------------------------', self.iteration_number, self.rounds)

                    hist = self.train()

                    if self.iteration_number == 0:
                        logging.info('------------------------- Training round %s, loss: %s -------------------------', self.rounds, str(round(np.mean(hist[1].history['loss']), 4)))
                        logging.info('------------------------- Training, round %s: Sent local model to the server -------------------------', self.rounds)
                    else:
                        logging.info('------------------------- Batch round %s, loss: %s -------------------------', self.rounds, str(round(np.mean(hist[1].history['loss']), 4)))
                        logging.info('------------------------- Batch %s, round %s: Sent local model to the server -------------------------', self.iteration_number, self.rounds)

                    self.send_model()

            self.STOP_FLAG = False
            self.rounds = 0
            self.iteration_number += 1
            self.epochs = self.testing_epochs
            # self.client_socket.close()
            if self.iteration_number <= self.NUM_TIMESTAMPS:
                edges = pd.read_csv('data/' + self.dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(self.iteration_number) + '_test_batch_edges.csv')
                nodes = pd.read_csv('data/' + self.dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(self.iteration_number) + '_test_batch_nodes.csv', index_col=0)

                logging.info('Batch %s initialized ', str(self.iteration_number))
                self.MODEL = Model(nodes, edges)
                num_train_ex, num_test_ex = self.MODEL.initialize()

                self.graph_params = (num_train_ex, num_test_ex)

                del nodes
                del edges
                gc.collect()

        logging.info(
            "______________________________________________________________________________________________________ Final Values ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")

        logging.info('Result report : Accuracy - %s (%s), Recall - %s (%s), AUC - %s (%s), F1 - %s (%s), Precision - %s (%s)', str(round(np.mean(self.all_test_metric_values[0]), 4)), str(round(np.std(self.all_test_metric_values[0]), 4)), str(round(np.mean(self.all_test_metric_values[1]), 4)), str(round(np.std(self.all_test_metric_values[1]), 4)), str(round(np.mean(self.all_test_metric_values[2]), 4)), str(round(np.std(self.all_test_metric_values[2]), 4)), str(round(np.mean(self.all_test_metric_values[3]), 4)), str(round(np.std(self.all_test_metric_values[3]), 4)), str(round(np.mean(self.all_test_metric_values[4]), 4)), str(round(np.std(self.all_test_metric_values[4]), 4)))
        logging.info('Result report : Accuracy 99th - 90th (%s, %s), Recall 99th - 90th (%s, %s), AUC 99th - 90th (%s, %s), F1 99th - 90th (%s, %s), Precision 99th - 90th (%s, %s), Mean time for a batch - %s (%s) seconds - 99th - 90th (%s, %s)', str(round(np.percentile(self.all_test_metric_values[0], 99), 4)), str(round(np.percentile(self.all_test_metric_values[0], 90), 4)), str(round(np.percentile(self.all_test_metric_values[1], 99), 4)), str(round(np.percentile(self.all_test_metric_values[1], 90), 4)), str(round(np.percentile(self.all_test_metric_values[2], 99), 4)), str(round(np.percentile(self.all_test_metric_values[2], 90), 4)), str(round(np.percentile(self.all_test_metric_values[3], 99), 4)), str(round(np.percentile(self.all_test_metric_values[3], 90), 4)), str(round(np.percentile(self.all_test_metric_values[4], 99), 4)), str(round(np.percentile(self.all_test_metric_values[4], 90), 4)), str(round(np.mean(self.all_test_metric_values[5]), 4)), str(round(np.std(self.all_test_metric_values[5]), 4)), str(round(np.percentile(self.all_test_metric_values[5], 99), 4)), str(round(np.percentile(self.all_test_metric_values[5], 90), 4)))
        logging.info(
            "______________________________________________________________________________________________________ Final Values ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")


        logging.info(str(self.all_test_metric_values))

        data = {"ACCURACY": str(round(np.mean(self.all_test_metric_values[0]), 4)), "ACCURACY_STD": str(round(np.std(self.all_test_metric_values[0]), 4)), "ACCURACY_99TH": str(round(np.percentile(self.all_test_metric_values[0], 99), 4)), "ACCURACY_90TH": str(round(np.percentile(self.all_test_metric_values[0], 90), 4)),
        "RECALL": str(round(np.mean(self.all_test_metric_values[1]), 4)), "RECALL_STD": str(round(np.std(self.all_test_metric_values[1]), 4)), "RECALL_99TH": str(round(np.percentile(self.all_test_metric_values[1], 99), 4)), "RECALL_90TH": str(round(np.percentile(self.all_test_metric_values[1], 90), 4)),
        "AUC": str(round(np.mean(self.all_test_metric_values[2]), 4)), "AUC_STD": str(round(np.std(self.all_test_metric_values[2]), 4)), "AUC_99TH": str(round(np.percentile(self.all_test_metric_values[2], 99), 4)), "AUC_90TH": str(round(np.percentile(self.all_test_metric_values[2], 90), 4)),
        "F1": str(round(np.mean(self.all_test_metric_values[3]), 4)), "F1_STD": str(round(np.std(self.all_test_metric_values[3]), 4)), "F1_99TH": str(round(np.percentile(self.all_test_metric_values[3], 99), 4)), "F1_90TH": str(round(np.percentile(self.all_test_metric_values[3], 90), 4)),
        "PRECISION": str(round(np.mean(self.all_test_metric_values[4]), 4)), "PRECISION_STD": str(round(np.std(self.all_test_metric_values[4]), 4)), "PRECISION_99TH": str(round(np.percentile(self.all_test_metric_values[4], 99), 4)), "PRECISION_90TH": str(round(np.percentile(self.all_test_metric_values[4], 90), 4)),
        "MEAN_TIME": str(round(np.mean(self.all_test_metric_values[5]), 4)), "MEAN_TIME_STD": str(round(np.std(self.all_test_metric_values[5]), 4)), "MEAN_TIME_99TH": str(round(np.percentile(self.all_test_metric_values[5], 99), 4)), "MEAN_TIME_90TH": str(round(np.percentile(self.all_test_metric_values[5], 90), 4)),
        "TOTAL_TIME": str(round(timer() - start_time, 2))}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data
        self.client_socket.sendall(data)

if __name__ == "__main__":

    if IP == 'localhost':
        IP = socket.gethostname()


    logging.warning('####################################### New Training Session: Client %s #######################################', PARTITION_ID)
    logging.info('Client started, graph name %s, graph ID %s, partition ID %s, training epochs %s, epochs %s', DATASET_NAME, GRAPH_ID, PARTITION_ID, TRAINING_EPOCHS, EPOCHS)


    edges = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(0) + '_training_batch_edges.csv')
    nodes = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(0) + '_training_batch_nodes.csv', index_col=0)

    from models.supervised import Model

    logging.info('Model initialized for training')
    model = Model(nodes, edges)
    num_train_ex, num_test_ex = model.initialize()

    graph_params = (num_train_ex, num_test_ex)

    logging.info('Number of training examples - %s, Number of testing examples - %s', num_train_ex,num_test_ex)

    client = Client(model, graph_params, weights_path=WEIGHTS_PATH, dataset_name=DATASET_NAME, graph_id=GRAPH_ID, partition_id=PARTITION_ID, training_epochs=TRAINING_EPOCHS, epochs=EPOCHS , IP=IP, PORT=PORT, iteration_number=0)

    del nodes
    del edges
    gc.collect()

    logging.info('Distributed training for streaming graphs started!')

    start = timer()
    client.run()
    end = timer()

    elapsed_time = end - start

    logging.info('Distributed training done!')
    logging.info('Training report : Total elapsed time %s seconds, graph name %s, graph ID %s, partition ID %s, training epochs %s, epochs %s', elapsed_time, DATASET_NAME, GRAPH_ID, PARTITION_ID, TRAINING_EPOCHS, EPOCHS)

    
