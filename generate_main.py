# Imports
import os
from generate_args import Args
from generate_model import *
from generate_data import *
from generate_train import *

if __name__ == '__main__':
    # Load the training arguments
    args = Args()

    # Check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)

    # Load the training graph(s)
    graphs = create_graphs(args)

    # Define test, train and validate data
    graphs_test = graphs
    graphs_train = graphs
    graphs_validate = graphs

    # Show graphs statistics
    args.max_num_node = max([len(graphs[i].nodes._nodes._atlas) for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])
    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # Save ground truth graphs
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    # Dataset initialization
    dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                        num_samples=args.batch_size*args.batch_ratio, replacement=True)

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        sampler=sample_strategy)

    # Model initialization
    rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn, hidden_size=args.hidden_size_rnn, 
                    num_layers=args.num_layers, has_input=True, has_output=False)

    output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node)

    # Start training
    train(args, dataset_loader, rnn, output)