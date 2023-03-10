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

arg_names = [
    'path_weights',
    'path_nodes',
    'path_edges',
    'graph_id',
    'partition_id',
    'epochs',
    'IP',
    'PORT',
    'name'
]

# args = dict(zip(arg_names, sys.argv[1:]))
args = dict()
args['path_weights'] = './weights/'
args['path_nodes'] = './data/'
args['path_edges'] = './data/'
args['graph_id'] = '4'
args['partition_id'] = '0'
args['initial_epochs'] = '15'
args['normal_epochs'] = '5'
args['IP'] = 'localhost'
args['PORT'] = '5000'
args['name'] = 'elliptic'

partition_id = args['partition_id']

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler(f'client_{partition_id}.log'),
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

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'
############

######################
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
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
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
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
tgn = tgn.to(device)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logging.info('num of training instances: {}'.format(num_instance))
logging.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)


#######################

class Client:

    def __init__(self, MODEL, graph_params, weights_path, graph_id, partition_id, initial_epochs = 10, normal_epochs = 2, IP = socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10, iteration_id=1, dataset_name=""):

        self.HEADER_LENGTH =  HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT

        self.weights_path = weights_path
        self.graph_id = graph_id
        self.partition_id = partition_id

        # Initial training is bit large
        self.initial_epochs = initial_epochs
        self.normal_epochs = normal_epochs
        self.epochs = initial_epochs

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
        self.MODEL.load_state_dict(data)
        self.GLOBAL_WEIGHTS = data

    def train(self):
        new_nodes_val_aps = []
        new_nodes_val_aucs = []
        val_aps = []
        val_aucs = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []
        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(NUM_EPOCH):
            start_epoch = time.time()
            ### Training

            # Reinitialize memory of the model at the start of each epoch
            if USE_MEMORY:
                self.MODEL.memory.__init_memory__()

            # Train using only training graph
            self.MODEL.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            logging.info('start {} epoch'.format(epoch))
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                optimizer.zero_grad()

                # Custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(num_instance, start_idx + BATCH_SIZE)
                    sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                        train_data.destinations[start_idx:end_idx]
                    edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                    timestamps_batch = train_data.timestamps[start_idx:end_idx]

                    size = len(sources_batch)
                    _, negatives_batch = train_rand_sampler.sample(size)

                    with torch.no_grad():
                        pos_label = torch.ones(size, dtype=torch.float, device=device)
                        neg_label = torch.zeros(size, dtype=torch.float, device=device)

                    self.MODEL = self.MODEL.train()
                    pos_prob, neg_prob = self.MODEL.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                        negatives_batch,
                                                                        timestamps_batch, edge_idxs_batch,
                                                                        NUM_NEIGHBORS)

                    loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                loss /= args.backprop_every

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if USE_MEMORY:
                    self.MODEL.memory.detach_memory()

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            # x = self.MODEL.state_dict()
            # y = self.MODEL.state_dict()
            # for key in x:
            #     y[key] = x[key]+y[key]

            ### Validation
            # Validation uses the full graph
            self.MODEL.set_neighbor_finder(full_ngh_finder)

            if USE_MEMORY:
                # Backup memory at the end of training, so later we can restore it and use it for the
                # validation on unseen nodes
                train_memory_backup = self.MODEL.memory.backup_memory()

            val_ap, val_auc = eval_edge_prediction(model=self.MODEL,
                                                   negative_edge_sampler=val_rand_sampler,
                                                   data=val_data,
                                                   n_neighbors=NUM_NEIGHBORS)
            if USE_MEMORY:
                val_memory_backup = self.MODEL.memory.backup_memory()
                # Restore memory we had at the end of training to be used when validating on new nodes.
                # Also backup memory after validation so it can be used for testing (since test edges are
                # strictly later in time than validation edges)
                self.MODEL.memory.restore_memory(train_memory_backup)

            # Validate on unseen nodes
            nn_val_ap, nn_val_auc = eval_edge_prediction(model=self.MODEL,
                                                         negative_edge_sampler=val_rand_sampler,
                                                         data=new_node_val_data,
                                                         n_neighbors=NUM_NEIGHBORS)

            if USE_MEMORY:
                # Restore memory we had at the end of validation
                self.MODEL.memory.restore_memory(val_memory_backup)

            new_nodes_val_aps.append(nn_val_ap)
            new_nodes_val_aucs.append(nn_val_auc)
            val_aps.append(val_ap)
            val_aucs.append(val_auc)
            train_losses.append(np.mean(m_loss))

            # Save temporary results to disk
            # pickle.dump({
            #     "val_aps": val_aps,
            #     "new_nodes_val_aps": new_nodes_val_aps,
            #     "train_losses": train_losses,
            #     "epoch_times": epoch_times,
            #     "total_epoch_times": total_epoch_times
            # }, open(results_path, "wb"))

            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            logging.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logging.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
            logging.info(
                'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
            logging.info(
                'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

            # Early stopping
            if early_stopper.early_stop_check(val_ap):
                logging.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logging.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                self.MODEL.load_state_dict(torch.load(best_model_path))
                logging.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                self.MODEL.eval()
                break
            else:
                torch.save(self.MODEL.state_dict(), get_checkpoint_path(epoch))
        logging.info('Total Epoch mean loss: {}'.format(np.mean(train_losses)))
        logging.info(
            'Total val auc: {}, new node val auc: {}'.format(np.mean(val_aucs), np.mean(new_nodes_val_aucs)))
        logging.info(
            'Total val ap: {}, new node val ap: {}'.format(np.mean(val_aps), np.mean(new_nodes_val_aps)))

    def evaluate(self):
        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = self.MODEL.memory.backup_memory()
        val_ap, val_auc = eval_edge_prediction(model=self.MODEL,
                                               negative_edge_sampler=val_rand_sampler,
                                               data=val_data,
                                               n_neighbors=NUM_NEIGHBORS)
        if USE_MEMORY:
            val_memory_backup = self.MODEL.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            self.MODEL.memory.restore_memory(train_memory_backup)

        # Validate on unseen nodes
        nn_val_ap, nn_val_auc = eval_edge_prediction(model=self.MODEL,
                                                     negative_edge_sampler=val_rand_sampler,
                                                     data=new_node_val_data,
                                                     n_neighbors=NUM_NEIGHBORS)
        logging.info(
            'Evaluation--> val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logging.info(
            'Evaluation--> val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
        return [[val_auc, nn_val_auc], [val_ap, nn_val_ap]]

    def run(self):
        while not self.STOP_FLAG:
            if self.iteration_id > 1 and self.rounds == 0:
                self.MODEL.set_weights(self.GLOBAL_WEIGHTS)
            read_sockets, _, exception_sockets = select.select([self.client_socket], [], [self.client_socket])

            for soc in read_sockets:
                self.fetch_model()

            if self.STOP_FLAG:

                logging.info('_____________________________________________________ (Iteration id %s) Final model evalution ____________________________________________________________', self.iteration_id)
                logging.info('(Iteration id %s) Final model (v%s) fetched', self.iteration_id, self.rounds)
                eval = self.evaluate()

                # logging.info('(Iteration id %s) Training set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_id, [0][0], eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                # logging.info('(Iteration id %s) Testing set : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',  self.iteration_id, [1][0], eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])
            else:

                self.rounds += 1
                logging.info('_____________________________________________________ (Iteration id %s) Training Round ____________________________________________________________',self.rounds, self.iteration_id)
                logging.info('(Iteration id %s) Global model v%s fetched', self.iteration_id, self.rounds - 1)

                eval = self.evaluate()

                # logging.info('(Iteration id %s) Global model v%s - Training set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_id, self.rounds - 1, eval[0][0], eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                # logging.info('(Iteration id %s) Global model v%s - Testing set evaluation : loss - %s, accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s', self.iteration_id, self.rounds - 1,  eval[1][0], eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])


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

            while not self.ITERATION_FLAG:





if __name__ == "__main__":

    from models.supervised import Model

    if 'IP' not in args.keys() or args['IP'] == 'localhost':
        args['IP'] = socket.gethostname()

    if 'PORT' not in args.keys():
        args['PORT'] = 5000

    if 'epochs' not in args.keys():
        args['epoch'] = 10

    logging.warning('####################################### New Training Session #######################################')
    logging.info('Client started, graph ID %s, partition ID %s, epochs %s',args['graph_id'],args['partition_id'],args['initial_epochs'])


    graph_params = (num_train_ex, num_test_ex)

    logging.info('(Iteration id %s) Number of training examples - %s, Number of testing examples %s', str(1), num_train_ex,num_test_ex)
    client = Client(tgn, graph_params, weights_path=args['path_weights'], graph_id=args['graph_id'], partition_id=args['partition_id'], initial_epochs = int(args['initial_epochs']), normal_epochs = int(args['normal_epochs']) ,IP=args['IP'],PORT=int(args['PORT']), iteration_id=1, dataset_name=args['name'])

    del nodes
    del edges
    gc.collect()

    logging.info('(Iteration id %s) Distributed training started!', str(1))

    start = timer()
    client.run()
    end = timer()

    elapsed_time = end -start
    logging.info('Distributed training done!')
    logging.info('Training report : Elapsed time %s seconds, graph ID %s, partition ID %s, epochs %s', elapsed_time,args['graph_id'],args['partition_id'],args['normal_epochs'])

    
