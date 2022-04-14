import torch
import torch.nn.functional as F
import time
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, DataListLoader
from tqdm import tqdm
from torch.utils.data import random_split
from preprocessing import * 
import sys

'''
GNN Model
'''


class GNNModel(torch.nn.Module):
    def __init__(self, in_features, hidden_features, output_features):
        super(GNNModel, self). __init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.droputout_value = 0.3
        self.conv1 = GCNConv(self.in_features, self.hidden_features)
        self.linear2 = torch.nn.Linear(self.hidden_features, self.output_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #print(data.x.shape)
        #print(data.edge_index.shape)
        edge_attr = None

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = gmp(x, batch)

        x = F.log_softmax(self.linear2(x), dim=-1)

        return x


@torch.no_grad()
def compute_test(loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if not multi_gpu:
			data = data.to(args.device)
		out = model(data)
		if multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test

test_class = DataFakeNews(root="data/")
test_class = test_class.shuffle()


num_features = test_class.num_features

num_training = int(len(test_class) * 0.2)
num_val = int(len(test_class) * 0.1)
num_test = len(test_class) - (num_training + num_val)
training_set, validation_set, test_set = random_split(test_class, [num_training, num_val, num_test])


#print(type(training_set[0].edge_index))
#sys.exit("BREKAING")
#Loading training set in data loader
loader = DataLoader
multi_gpu = False
hidden_features = 128
num_classification = 2
batch_size = 128
train_loader = loader(training_set, batch_size=batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=batch_size, shuffle=False)

# Num features is the x input which currently is [poltifactID, ]

device = torch.device('cuda:0')
model = GNNModel(num_features, hidden_features, num_classification).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__== '__main__':
    #Model Training
    min_loss = 1e10
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in tqdm(range(30)):
        loss_train = 0.0
        out_log = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            #print("DATA:", data)
            #print("Is Tensor: ", torch.is_tensor(data))
            if not multi_gpu:
                data = data.to(device)
            
            out = model(data)
            sys.exit("BREAKING")
            print("NOT HERE")
            if multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
            else:
                y = data.y
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
                f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
                f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
                f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

    [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
            f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
