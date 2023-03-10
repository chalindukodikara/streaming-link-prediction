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

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

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
        self.GLOBAL_WEIGHTS = MODEL.get_weights()

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

    from models.supervised import Model

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
    args['initial_num_rounds'] = '3'
    args['normal_num_rounds'] = '2'
    args['IP'] = 'localhost'
    args['PORT'] = '5000'
    args['name'] = 'tg'
    args['transfer_learning'] = True
    args['num_timestamps'] = '6'

    logging.warning('####################################### New Training Session #######################################')
    logging.info('Server started , graph ID %s, number of clients %s, number of rounds %s, transfer learning %s, number of timestamps %s',args['graph_id'],args['num_clients'], args['initial_num_rounds'], args['transfer_learning'], args['num_timestamps'])

    if 'IP' not in args.keys() or args['IP'] == 'localhost':
        args['IP'] = socket.gethostname()

    if 'PORT' not in args.keys():
        args['PORT'] = 5000
    # for timestamp_id in range(1, 25):
    #     logging.warning('###################### New Interation (%s) started #############################', str(timestamp_id))

    edges = pd.read_csv('data/' + args['name'] + '/' + str(1) + '_edges.csv')
    nodes = pd.read_csv('data/' + args['name'] + '/' + str(1) + '_nodes.csv',index_col=0)

    model = Model(nodes,edges)
    model.initialize()

    logging.info('Model initialized')

    server = Server(model, INITIAL_ROUNDS=int(args['initial_num_rounds']), NORMAL_ROUNDS=int(args['normal_num_rounds']), weights_path=args['path_weights'],graph_id=args['graph_id'],MAX_CONN=int(args['num_clients']),IP=args['IP'],PORT=int(args['PORT']),iteration_id=1,transfer_learning=args['transfer_learning'], NUM_TIMESTAMPS=int(args['num_timestamps']))

    del nodes
    del edges
    del model
    gc.collect()

    logging.info('Federated training started!')

    start = timer()
    server.run()
    end = timer()

    elapsed_time = end - start
    logging.info('Federated training done!')
    logging.info('Training report : Elapsed time %s seconds, graph ID %s, number of clients %s, number of rounds %s',elapsed_time,args['graph_id'],args['num_clients'],args['normal_num_rounds'])
