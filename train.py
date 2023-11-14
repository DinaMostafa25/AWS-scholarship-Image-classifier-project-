import argparse
from data_utility import load_data
import  model_functions


parser = argparse.ArgumentParser(description='Traning............')
parser.add_argument('data_directory')
parser.add_argument('--save_dir')
parser.add_argument('--arch')
parser.add_argument('--learning_rate')
parser.add_argument('--hidden_units')
parser.add_argument('--epoches')
parser.add_argument('--gpu', action = 'store_true')


args = parser.parse_args()

save_dir = ''if args.save_dir is None else args.save_dir 
network_archticture = 'densenet121' if args.arch is None else args.arch
lr = .003 if args.learning_rate is None else float(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epoches = 8 if args.epoches is None else args.epoches
gpu = False if args.gpu is None else args.gpu

train_data, trainloader, validloader, testloader = load_data(args.data_directory)
model = model_functions.network(network_archticture,hidden_units)
model.class_to_idx = train_data.class_to_idx
model, criterion = model_functions.train_network(model, epoches, lr, trainloader, validloader, gpu)
model_functions.test_network(model, testloader, criterion, gpu)
model_functions.save_check_point(model, network_archticture, hidden_units, epoches, lr, save_dir )