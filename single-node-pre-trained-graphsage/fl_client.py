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
parser.add_argument('--partition_id', type=int, default=0, help='Partition ID')
parser.add_argument('--partition_size', type=int, default=1, help='Partition size')
parser.add_argument('--graph_id', type=int, default=1, help='Graph ID')

######## Frequently configured #######
parser.add_argument('--dataset_name', type=str, default='wikipedia', help='Dataset name')
parser.add_argument('--partition_algorithm', type=str, default='hash', help='Partition algorithm')
parser.add_argument('--training_epochs', type=int, default=40, help='Initial Training: number of epochs')
parser.add_argument('--epochs', type=int, default=10, help='Streaming data training for batches: number of epochs')

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
        logging.FileHandler('logs/client/{}_{}_{}_partition_{}_client_{}.log'.format(
            str(time.strftime('%m %d %H:%M:%S # %l:%M%p on %b %d, %Y')), DATASET_NAME, PARTITION_ALGORITHM,
            PARTITION_SIZE,
            PARTITION_ID)),

        logging.StreamHandler(sys.stdout)
    ]
)
############################################
path = 'data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(0)
files = os.listdir(path)
paths = [os.path.join(path, basename) for basename in files]
NUM_TIMESTAMPS = (int(max(paths, key=os.path.getctime).split('/')[-1].split('_')[0]))


class Client:

    def __init__(self, MODEL, graph_params, weights_path, dataset_name, graph_id, partition_id, training_epochs=30,
                 epochs=2, IP=socket.gethostname(), PORT=5000, HEADER_LENGTH=10, iteration_number=1):

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

    def train(self):
        return self.MODEL.fit(epochs=self.epochs)

    def run(self):
        while True:
            if self.iteration_number > 0:
                self.MODEL.set_weights(self.GLOBAL_WEIGHTS)
                if self.iteration_number == 1:
                    logging.info(
                        '################################## Next batch processing started: pre-trained model will be used ##################################')

            if self.iteration_number == 0:
                training_start_time = timer()
                logging.info(
                    '################################## Initial model training started ##################################')
            else:
                testing_start_time = timer()

            if self.iteration_number == 0:

                self.train()
                training_end_time = timer()

                logging.info(
                    '################################## Initial model evaluation ##################################')

                eval = self.MODEL.evaluate()

                try:
                    f1_train = round((2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4]), 2)
                    f1_test = round((2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4]), 2)
                except ZeroDivisionError as e:
                    f1_train = "undefined"
                    f1_test = "undefined"

                logging.info(
                    'Initially trained model: Training set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s, training time - %s seconds',
                    round(eval[0][0], 4), round(eval[0][1], 4), round(eval[0][2], 4), round(eval[0][3], 4), f1_train,
                    round(eval[0][4], 4), round(training_start_time - training_end_time, 0))
                logging.info(
                    'Initially trained model: Testing set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',
                    round(eval[1][0], 4), round(eval[1][1], 4), round(eval[1][2], 4), round(eval[1][3], 4), f1_test,
                    round(eval[1][4], 4))


            else:
                logging.info('################ Batch %s: model evalution using pre-trained model ################',
                             self.iteration_number)

                eval = self.MODEL.evaluate()

                try:
                    f1_train = round((2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4]), 4)
                    f1_test = round((2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4]), 4)
                except ZeroDivisionError as e:
                    f1_train = "undefined"
                    f1_test = "undefined"

                logging.info(
                    'Batch %s: Training set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',
                    self.iteration_number, round(eval[0][0], 4), round(eval[0][1], 4), round(eval[0][2], 4),
                    round(eval[0][3], 4), f1_train, round(eval[0][4], 4))
                logging.info(
                    'Batch %s: Testing set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',
                    self.iteration_number, round(eval[1][0], 4), round(eval[1][1], 4), round(eval[1][2], 4),
                    round(eval[1][3], 4), f1_test, round(eval[1][4], 4))

                self.all_test_metric_values[0].append(round(eval[1][1], 4))  # accuracy
                self.all_test_metric_values[1].append(round(eval[1][2], 4))  # recall
                self.all_test_metric_values[2].append(round(eval[1][3], 4))  # auc
                self.all_test_metric_values[3].append(f1_test)  # f1
                self.all_test_metric_values[4].append(round(eval[1][4], 4))  # precision
                self.all_test_metric_values[5].append(0)  # time

            self.iteration_number += 1
            self.epochs = self.testing_epochs

            # self.client_socket.close()
            if self.iteration_number <= NUM_TIMESTAMPS:
                self.GLOBAL_WEIGHTS = self.MODEL.get_weights()
                edges = pd.read_csv(
                    'data/' + self.dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(
                        self.iteration_number) + '_test_batch_edges.csv')
                nodes = pd.read_csv(
                    'data/' + self.dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(
                        self.iteration_number) + '_test_batch_nodes.csv', index_col=0)
                logging.info('Batch %s initialized ', str(self.iteration_number))
                self.MODEL = Model(nodes, edges)
                num_train_ex, num_test_ex = self.MODEL.initialize()

                self.graph_params = (num_train_ex, num_test_ex)

                del nodes
                del edges
                gc.collect()

            else:
                break
        logging.info(
            "______________________________________________________________________________________________________ Final Values ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")

        logging.info(
            'Result report : Accuracy - %s (%s), Recall - %s (%s), AUC - %s (%s), F1 - %s (%s), Precision - %s (%s), Mean time for a batch - %s (%s) seconds',
            str(round(np.mean(self.all_test_metric_values[0]), 4)),
            str(round(np.std(self.all_test_metric_values[0]), 4)),
            str(round(np.mean(self.all_test_metric_values[1]), 4)),
            str(round(np.std(self.all_test_metric_values[1]), 4)),
            str(round(np.mean(self.all_test_metric_values[2]), 4)),
            str(round(np.std(self.all_test_metric_values[2]), 4)),
            str(round(np.mean(self.all_test_metric_values[3]), 4)),
            str(round(np.std(self.all_test_metric_values[3]), 4)),
            str(round(np.mean(self.all_test_metric_values[4]), 4)),
            str(round(np.std(self.all_test_metric_values[4]), 4)),
            str(round(np.mean(self.all_test_metric_values[5]), 2)),
            str(round(np.std(self.all_test_metric_values[5]), 2)))
        logging.info('Result report : Accuracy 99th - 90th (%s, %s), Recall 99th - 90th (%s, %s), AUC 99th - 90th (%s, %s), F1 99th - 90th (%s, %s), Precision 99th - 90th (%s, %s), Mean time for a batch - %s (%s) seconds - 99th - 90th (%s, %s)', str(round(np.percentile(self.all_test_metric_values[0], 99), 4)), str(round(np.percentile(self.all_test_metric_values[0], 90), 4)), str(round(np.percentile(self.all_test_metric_values[1], 99), 4)), str(round(np.percentile(self.all_test_metric_values[1], 90), 4)), str(round(np.percentile(self.all_test_metric_values[2], 99), 4)), str(round(np.percentile(self.all_test_metric_values[2], 90), 4)), str(round(np.percentile(self.all_test_metric_values[3], 99), 4)), str(round(np.percentile(self.all_test_metric_values[3], 90), 4)), str(round(np.percentile(self.all_test_metric_values[4], 99), 4)), str(round(np.percentile(self.all_test_metric_values[4], 90), 4)), str(0), str(0), str(0), str(0))
        logging.info(
            "______________________________________________________________________________________________________ Final Values ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")

        logging.info(str(self.all_test_metric_values))


if __name__ == "__main__":
    logging.warning(
        '####################################### New Training Session: Client %s #######################################',
        PARTITION_ID)
    logging.info('Client started, graph name %s, graph ID %s, partition ID %s, training epochs %s, epochs %s',
                 DATASET_NAME, GRAPH_ID, PARTITION_ID, TRAINING_EPOCHS, EPOCHS)

    edges = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(
        0) + '_training_batch_edges.csv')
    nodes = pd.read_csv('data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '/' + str(
        0) + '_training_batch_nodes.csv', index_col=0)

    from models.supervised import Model

    logging.info('Model initialized for training')
    model = Model(nodes, edges)
    num_train_ex, num_test_ex = model.initialize()

    graph_params = (num_train_ex, num_test_ex)

    logging.info('Number of training examples - %s, Number of testing examples - %s', num_train_ex, num_test_ex)

    client = Client(model, graph_params, weights_path=WEIGHTS_PATH, dataset_name=DATASET_NAME, graph_id=GRAPH_ID,
                    partition_id=PARTITION_ID, training_epochs=TRAINING_EPOCHS, epochs=EPOCHS, IP=IP, PORT=PORT,
                    iteration_number=0)

    del nodes
    del edges
    gc.collect()

    logging.info('Distributed training for streaming graphs started!')

    start = timer()
    client.run()
    end = timer()

    elapsed_time = end - start

    logging.info('Distributed training done!')
    logging.info(
        'Training report : Total elapsed time %s seconds, graph name %s, graph ID %s, partition ID %s, training epochs %s, epochs %s',
        elapsed_time, DATASET_NAME, GRAPH_ID, PARTITION_ID, TRAINING_EPOCHS, EPOCHS)


