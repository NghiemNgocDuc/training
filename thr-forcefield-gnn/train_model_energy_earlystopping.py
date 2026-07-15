### This is for early stopping.



import sys
import os
sys.path.append('../')
import pandas as pd

from util import *
import torch
import torch.optim as optim
from torch.utils.data import Subset, random_split
from sklearn.model_selection import KFold, train_test_split


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
parser.add_argument('-e','--epochs',default=100,type=int,help='epochs to train for')

parser.add_argument('-n','--name',default='',type=str,help='name of model')
parser.add_argument('-c','--clip',default=0,type=float,help='norm clipping')

parser.add_argument('--use_energy', action='store_true')
parser.add_argument('--noise_std', default=0.0, type=float, help='Std of Gaussian noise added to positions during training (PES smoothing)')
parser.add_argument('--weight_decay', default=0.0, type=float, help='L2 weight decay for optimizer')
parser.add_argument('--lambda_lap', default=0.0, type=float, help='Laplacian smoothness penalty weight ||div F||^2')
args = parser.parse_args()

output_folder = "results_nogbforce_earlystopping"

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
        # atom_features = attr.loc[:, ['At.#']]
        atom_features = attr.loc[:, ['At.#', 'Charge']]
        # atom_features = attr.loc[:, ['At.#', 'LJ_Radius', 'LJ_Depth', 'Charge', 'GB_Radius', 'GB_Screen']]

        for sample_dir in sample_dirs:
            coordinates = readXYZ(os.path.join(folder_path, "WINDOWS", sample_dir, "coordinates.xyz"))
            filecount += 1
            GBn2_forces = readXYZ(os.path.join(folder_path, "WINDOWS", sample_dir, "GBn2.frc.xyz")) ## we should add this in the input, and also the partial charges
            forces = readXYZ(os.path.join(folder_path, "WINDOWS", sample_dir, "OPC.frc.xyz"))
            Y = forces - GBn2_forces
            X = atom_features.to_numpy(dtype=float)
            X = np.concatenate((X, GBn2_forces), axis=1)
            data_sample = Data(x=torch.tensor(X, dtype=torch.float), y=torch.tensor(Y, dtype=torch.float),
                               pos=torch.tensor(coordinates, dtype=torch.float, requires_grad=True),
                               forces=torch.tensor(forces, dtype=torch.float),
                               GBn2_forces=torch.tensor(GBn2_forces, dtype=torch.float))
            # Save the sample directory name as an attribute so it can be used later.
            data_sample.sample_dir = sample_dir
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
        optimizer.zero_grad()

        pos = data.pos.clone()
        if args.noise_std > 0:
            pos = pos + torch.randn_like(pos) * args.noise_std
        pos.requires_grad_()

        energies = model(data.x, pos, data.batch)
        forces_pred = -torch.autograd.grad(
            outputs=energies,
            inputs=pos,
            grad_outputs=torch.ones_like(energies),
            create_graph=True
        )[0]

        loss = criterion(forces_pred, data.y)

        if args.lambda_lap > 0:
            lap = torch.zeros(pos.shape[0], device=pos.device)
            for d in range(3):
                g = torch.autograd.grad(
                    forces_pred[:, d].sum(), pos,
                    create_graph=True, retain_graph=(d < 2)
                )[0]
                lap = lap + g[:, d]
            loss = loss + args.lambda_lap * lap.pow(2).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation function
def validate(loader):
    model.eval()
    total_loss = 0
    with torch.enable_grad():
        for data in loader:
            if isinstance(data, list):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            data.pos.requires_grad_()
            energies = model(data.x, data.pos, data.batch)
            forces_pred = -torch.autograd.grad(
                outputs=energies,
                inputs=data.pos,
                grad_outputs=torch.ones_like(energies),
                create_graph=True
            )[0]
            loss = criterion(forces_pred, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)


def test_model(test_loader, model):
    model.eval()
    predictions = []
    ground_truth = []
    total_loss = 0
    with torch.enable_grad():
        for data in test_loader:
            data = data.to(device)
            data.pos.requires_grad_()
            energies = model(data.x, data.pos, data.batch)
            forces_pred = -torch.autograd.grad(
                outputs=energies,
                inputs=data.pos,
                grad_outputs=torch.ones_like(energies),
                create_graph=True
            )[0]
            predictions.append(forces_pred.cpu().detach().numpy())
            ground_truth.append(data.y.cpu().detach().numpy())
            loss = criterion(forces_pred, data.y)
            total_loss += loss.item()
    test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    # Concatenate the results_nogbforce_earlystopping and return
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    return predictions, ground_truth


def test_model_individual(test_loader, model):
    """
    Process test samples one at a time (batch_size=1) so that we can save each predicted label
    with its sample_dir as the file name.
    """
    model.eval()
    total_loss = 0
    with torch.enable_grad():
        for data in test_loader:
            # data is a batch of one sample
            data = data.to(device)
            data.pos.requires_grad_()
            energies = model(data.x, data.pos, data.batch)
            forces_pred = -torch.autograd.grad(
                outputs=energies,
                inputs=data.pos,
                grad_outputs=torch.ones_like(energies),
                create_graph=True
            )[0]
            loss = criterion(forces_pred, data.y)
            total_loss += loss.item()
            # Convert prediction and ground truth to numpy arrays
            pred_np = forces_pred.cpu().detach().numpy()
            gt_np = data.y.cpu().detach().numpy()
            row_losses = np.mean((pred_np - gt_np) ** 2, axis=1)
            # Retrieve the sample name. The DataLoader collates non-tensor attributes as a list.
            sample_name = data.sample_dir[0] if isinstance(data.sample_dir, list) else data.sample_dir
            # Save the predicted label (and ground truth if desired) to a CSV file.
            df = pd.DataFrame(pred_np)
            # Optionally, you can add ground truth as well:
            df_gt = pd.DataFrame(gt_np)
            df.columns = [f"Pred_{i}" for i in range(df.shape[1])]
            df_gt.columns = [f"GT_{i}" for i in range(df_gt.shape[1])]
            df_loss = pd.DataFrame(row_losses, columns=['Loss'])
            df_all = pd.concat([df, df_gt, df_loss], axis=1)
            save_path = output_folder+f"/{sample_name}_prediction.csv"
            df_all.to_csv(save_path, index=False)
            print(f"Saved prediction for sample '{sample_name}' to {save_path}")
    test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")



dataset = CustomDataset(root="../Data/THR_data_19SB_OPC_GBn2")
k_folds = 5
batch_size = args.batchsize
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
patience = 15
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Create data loaders for train and test splits
    # Split the training data into training and validation sets
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    input_channels = train_subset[0].x.shape[1]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    model = Model(
        in_channels= input_channels,
        hidden_channels=128,
        out_channels=1,
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
        num_output_layers=2,
        is_energy=True
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # Train the model
    # Early stopping parameters

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(train_loader, optimizer)
        val_loss = validate(val_loader)
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} sec")

        if epoch < 10: ##skip the burning phase in case of unstable training
            continue

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    # Load the best model state before testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Save the model for this fold
    model_save_path = output_folder + f"/fold_{fold + 1}_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model for fold {fold + 1} to {model_save_path}")

    # # Get test predictions and save them
    # preds, gt = test_model(test_loader, model)
    # # Save predictions and ground truth to a CSV file (adjust columns as necessary)
    # df_results = pd.DataFrame({
    #     "Ground_Truth": list(gt.reshape(-1)),
    #     "Prediction": list(preds.reshape(-1))
    # })
    # results_save_path = f"fold_{fold + 1}_predictions.csv"
    # df_results.to_csv(results_save_path, index=False)
    # print(f"Saved test predictions for fold {fold + 1} to {results_save_path}")

    # test_model(test_loader, model)

    test_model_individual(test_loader, model)



