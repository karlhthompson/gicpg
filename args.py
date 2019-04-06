
### program configuration
class Args():
    def __init__(self):
        ### if clean tensorboard
        self.clean_tensorboard = False
        ### Which CUDA GPU device is used for training
        self.cuda = -1

        ### Which GraphRNN model variant is used.
        # The simple version of Graph RNN
        self.note = 'GraphRNN_MLP'
        # The dependent Bernoulli sequence version of GraphRNN
        # self.note = 'GraphRNN_RNN'
        # Graph RNN VAE
        # self.note = 'GraphRNN_VAE'

        ### Which dataset is used to train the model
        self.graph_type = 'arch'

        # if none, then auto calculate
        self.max_num_node = None # max number of nodes in a graph
        self.max_prev_node = None # max previous node that looks back

        ### network config
        self.parameter_shrink = 1
        self.hidden_size_rnn = int(128/self.parameter_shrink) # hidden size for main RNN
        self.hidden_size_rnn_output = 16 # hidden size for output RNN
        self.embedding_size_rnn = int(64/self.parameter_shrink) # the size for LSTM input
        self.embedding_size_rnn_output = 8 # the embedding size for output rnn
        self.embedding_size_output = int(64/self.parameter_shrink) # the embedding size for output (VAE/MLP)

        self.batch_size = 2 # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 2 # normal: 32
        self.test_total_size = 10 # normal: 1000
        self.num_layers = 4

        ### training config
        self.num_workers = 4 # num workers to load data, default 4
        self.batch_ratio = 1 # normal: 32, how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = 10000 # normal: 3000, now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 250 # normal: 100
        self.epochs_test = 250 # normal: 100
        self.epochs_log = 250 # normal: 100
        self.epochs_save = 250 # normal: 100

        self.lr = 0.003
        self.milestones = [10000, 20000] # normal: [400, 1000]
        self.lr_rate = 0.3

        self.sample_time = 2 # sample time in each time step, when validating

        ### output config
        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = "./"
        self.model_save_path = self.dir_input+'model_save/' # only for nll evaluation
        self.graph_save_path = self.dir_input+'graphs/'
        self.figure_save_path = self.dir_input+'figures/'
        self.timing_save_path = self.dir_input+'timing/'
        self.figure_prediction_save_path = self.dir_input+'figures_prediction/'
        self.nll_save_path = self.dir_input+'nll/'

        self.load = False # if load model, default lr is very low
        self.load_epoch = 3000 # normal: 3000
        self.save = True

        ### filenames to save intemediate and final outputs
        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'