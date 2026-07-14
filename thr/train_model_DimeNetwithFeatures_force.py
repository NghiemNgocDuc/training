import sys
import os
sys.path.append('../')
sys.path.append('../multistage')
import pandas as pd

from util import *
import torch
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.model_selection import KFold


import argparse
from torch_geometric.nn import DimeNet
from torch_geometric.data import Data
# from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from DimeModels import DimeNetPlus
import time

seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Run Model Training')
parser.add_argument('-b','--batchsize',default=5,type=int,help='Batchsize')
parser.add_argument('-p','--per',default=0.8,type=float,help='fraction of training')
parser.add_argument('-f','--fra',default=0.1,type=float,help='scaling parameter')
parser.add_argument('-r','--random',default=161311,type=int,help='random seed')
parser.add_argument('-ra','--radius',default=2.0,type=float,help='radius')
parser.add_argument('-l','--lr',default=0.001,type=float,help='learning rate')
parser.add_argument('-e','--epochs',default=2,type=int,help='epochs to train for')

parser.add_argument('-n','--name',default='',type=str,help='name of model')
parser.add_argument('-c','--clip',default=0,type=float,help='norm clipping')

parser.add_argument('--use_energy', action='store_true')
args = parser.parse_args()



#### torch-cluster/torch-sparse not supporting mps yet
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using gpu")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    device = torch.device('cpu')
    print("Using cpu")

# device = torch.device('cpu')

# Customize our own dataset with atomic positions, atomic numbers, and forces
class CustomDataset:
    def __init__(self, root):
        self.dataset = self.get_data(root)
        pass

    def get_data(self, folder_path):
        Dataset = []
        feats = []
        feat_names = []
        sample_dirs = os.listdir(os.path.join(folder_path, "WINDOWS"))
        filecount = 0
        for line in open(os.path.join(folder_path, "embeddings.out")):
            toks = list(filter(None, line.strip().split(" ")))
            if len(toks) < 10:
                continue
            if toks[0] == "ATOM":
                feat_names = toks
            else:
                feats.append(toks)
        attr = pd.DataFrame(feats, columns=feat_names)
        # print(attr)
        # atom_features = attr.loc[:, ['At.#', 'Charge']]
        atom_features = attr.loc[:, ['At.#', 'LJ_Radius', 'LJ_Depth', 'Charge', 'GB_Radius', 'GB_Screen']]

        for sample_dir in sample_dirs:
            coordinates = readXYZ(os.path.join(folder_path, "WINDOWS", sample_dir, "coordinates.xyz"))
            filecount += 1
            GBn2_forces = readXYZ(os.path.join(folder_path, "WINDOWS", sample_dir, "GBn2.frc.xyz")) ## we should add this in the input, and also the partial charges
            forces = readXYZ(os.path.join(folder_path, "WINDOWS", sample_dir, "OPC.frc.xyz"))
            Y = forces - GBn2_forces
            X = atom_features.to_numpy(dtype=float)
            X = np.concatenate((X,GBn2_forces), axis=1)
            data_sample = Data(x=torch.tensor(X, dtype=torch.float), y=torch.tensor(Y, dtype=torch.float),
                               pos=torch.tensor(coordinates, dtype=torch.float, requires_grad=True),
                               forces=torch.tensor(forces, dtype=torch.float),
                               GBn2_forces=torch.tensor(GBn2_forces, dtype=torch.float))
            Dataset.append(data_sample)

        print('Constructed Dataset with %i positions ' % (len(Dataset)))

        return Dataset

    def __getitem__(self, idx):
        # Return atomic numbers (z), positions (pos), and forces (forces) for each molecule
        # return self.dataset[idx].x, self.dataset[idx].pos, self.dataset[idx].y
        return self.dataset[idx]
        pass

    def __len__(self):
        # Return the total number of data points
        return len(self.dataset)
        pass


Model = DimeNetPlus
# model = model_class(radius=args.radius,max_num_neighbors=10000,parameters=gbneck_parameters,device=device,fraction=args.fra,unique_radii=unique_radii)





criterion = torch.nn.MSELoss()


# Training function
def train(loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        if isinstance(data, list):
            data = [d.to(device) for d in data]
        else:
            data = data.to(device)
        data.pos.requires_grad_()
        optimizer.zero_grad()
        # Forces predicted from energy gradients
        # energies = model(data.x, data.pos, data.batch)
        # forces_pred = -torch.autograd.grad(
        #     outputs=energies,
        #     inputs=data.pos,
        #     grad_outputs=torch.ones_like(energies),
        #     create_graph=True
        # )[0]


        #Forces predicted directly from GNN
        forces_pred = model(data.x, data.pos, data.batch)

        loss = criterion(forces_pred, data.y)  # Compare predicted vs. ground-truth forces
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation function
def validate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            if isinstance(data, list):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            data.pos.requires_grad_()
            forces_pred = model(data.x, data.pos, data.batch)
            loss = criterion(forces_pred, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)


def test_model(test_loader, model):
    # Test the model
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, list):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            data.pos.requires_grad_()
            forces_pred = model(data.x, data.pos, data.batch)
            loss = criterion(forces_pred, data.y)
            total_loss += loss.item()
    test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")


dataset = CustomDataset(root="../Data/THR_data_19SB_OPC_GBn2")
k_folds = 5
batch_size = args.batchsize
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Create data loaders for train and test splits
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    input_channels = train_subset[0].x.shape[1]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    model = Model(
        in_channels= input_channels,
        hidden_channels=128,
        out_channels=3,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=args.radius,
        max_num_neighbors=32,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=2
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # Train the model
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(train_loader, optimizer)
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Time: {epoch_duration:.2f} seconds")

        # val_loss = validate(val_loader)

        if (epoch+1)%10==0:
            test_model(test_loader, model)
    test_model(test_loader, model)
    break



