# Imports
import torch
import time as tm
import numpy as np
from torch import optim
from torch.autograd import Variable
from tensorboard_logger import Logger
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from gicpg.generate_model import sample_sigmoid, binary_cross_entropy_weight
from gicpg.generate_data import decode_adj, get_graph, save_graph_list

# Training function
def train(args, dataset_train, rnn, output):
    # Check whether or not to load an existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'gru_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # Initialize the optimizer
    optimizer_mlp = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_mlp = MultiStepLR(optimizer_mlp, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # Start the main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()

        # Run training
        train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                        optimizer_mlp, optimizer_output,
                        scheduler_mlp, scheduler_output)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start

        # Run testing
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < args.test_total_size:
                    G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size = args.test_batch_size, 
                        sample_time = sample_time)
                    G_pred.extend(G_pred_step)

                # Save the generated graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)

            print('test done, graphs saved')


        # Save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'gru_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
        
    np.save(args.timing_save_path + args.fname, time_all)

# Define train function
def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_mlp, optimizer_output,
                    scheduler_mlp, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0

    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]

        # Initialize gru hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # Sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x)
        y = Variable(y)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]

        # Use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()

        # Update deterministic and gru
        optimizer_output.step()
        optimizer_mlp.step()
        scheduler_output.step()
        scheduler_mlp.step()

        # Output only the first batch's statistics for each epoch
        if batch_idx==0: 
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # Logging
        Logger('loss_'+args.fname, loss.item(), epoch*args.batch_ratio+batch_idx)

        # Update the loss sum
        loss_sum += loss.item()

    return loss_sum/(batch_idx+1)


# Define test function
def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node))
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data)
    y_pred_long_data = y_pred_long.data.long()

    # Save the generated graphs as a pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list