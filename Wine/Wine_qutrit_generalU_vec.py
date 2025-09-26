#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 10:33:39 2025

@author: ccristiano
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef
import pandas as pd
import time
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.utils import shuffle
from copy import deepcopy

#%%

'''
     ==================
       some functions
     ==================
'''

def evaluate_accuracy(dataloader, model):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            preds = torch.argmax(output, dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return (
        accuracy_score(y_true, y_pred, ) * 100,
        f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        cohen_kappa_score(y_true, y_pred),
        matthews_corrcoef(y_true, y_pred),
        y_true, 
        y_pred
        )
        
def cost_fidelity(output, target):
    return torch.mean((1 - output[range(len(target)), target]) ** 2)


#%%

'''
      ==================
         1 qutrit QNN
      ==================
'''

class QNN_0_Sum(nn.Module):

    def __init__(self, num_layers, num_features, num_params, init_weights=None):
        super(QNN_0_Sum, self).__init__()

        self.num_layers = num_layers
        self.num_features = num_features
        self.num_params = num_params

        # Qutrit basis
        q0 = torch.tensor([[1], [0], [0]], dtype=torch.complex64)
        q1 = torch.tensor([[0], [1], [0]], dtype=torch.complex64)
        q2 = torch.tensor([[0], [0], [1]], dtype=torch.complex64)
        self.register_buffer("q0", q0)
        self.register_buffer("q1", q1)
        self.register_buffer("q2", q2)

        # Outer product helper
        gm = lambda A, B: torch.kron(A, B.T)

        # Gell-Mann generators
        # Gell-Mann generators
        gm1 = (gm(q0, q1) + gm(q1, q0)).to(torch.complex64)
        gm2 = (-1j * (gm(q0, q1) - gm(q1, q0))).to(torch.complex64)
        gm3 = (gm(q0, q0) - gm(q1, q1)).to(torch.complex64)
        gm4 = (gm(q0, q2) + gm(q2, q0)).to(torch.complex64)
        gm5 = (-1j * (gm(q0, q2) - gm(q2, q0))).to(torch.complex64)
        gm6 = (gm(q1, q2) + gm(q2, q1)).to(torch.complex64)
        gm7 = (-1j * (gm(q1, q2) - gm(q2, q1))).to(torch.complex64)
        gm8 = (1 / torch.sqrt(torch.tensor(3., dtype=torch.float32)) * (gm(q0, q0) + gm(q1, q1) - 2 * gm(q2, q2))).to(torch.complex64)
            
        Sz12 = (gm(q1, q1) - gm(q2, q2)).to(torch.complex64)
        Sz02 = (gm(q0, q0) - gm(q2, q2)).to(torch.complex64)
        
        
        # Configurations
        
        #C30
        generators12345678 = [gm1,
                              gm2,
                              gm3, 
                              gm4, 
                              gm5,
                              gm6,
                              gm7,
                              gm8]
        
        #C31
        generators123 = [gm1,
                          gm2, 
                          gm3]
        
        generators456 = [gm4,
                          gm5, 
                          gm6]
        
        #C32     
        generatorsYYY = [gm2,  # Sy_01
                          gm5,  # Sy_02
                          gm7]  # Sy_12
                     


        #C32           
        generatorsYYYZ = [gm2,  # Sy_01
                          gm5,  # Sy_02
                          gm7,  # Sy_12
                          gm3]  # Sz_01
        
    
        
        generators_enc = generators12345678
        
        generators_var = generators12345678
            
        
        self.register_buffer("gens_enc", torch.stack(generators_enc[:self.num_features]))  # [F, 3, 3]
        self.register_buffer("gens_var", torch.stack(generators_var[:self.num_params]))  # [F, 3, 3]
        
        # Label projectors
        self.register_buffer("label_ops", torch.stack([gm(q, q) for q in [q0, q1, q2]]))


        # Per-layer weights with num_features components
        self.weights = nn.ParameterList([
            nn.Parameter(init_weights[i] if init_weights else torch.rand(self.num_params) * 2 - 1)
            for i in range(num_layers)
        ])

    def forward(self, batch):
        """
        batch: Tensor of shape [B, num_features]
        """
        B = batch.shape[0]

        # Initial qutrit state expanded for batch: [B, 3, 1]
        state = self.q0.expand(B, -1, -1).clone()  # [B, 3, 1]
        
        one_j = torch.tensor(1j, dtype=torch.complex64)

        for i in range(self.num_layers):
            # Encoding
            gen = self.gens_enc.unsqueeze(0)  # [1, F, 3, 3]
            x = batch.unsqueeze(-1).unsqueeze(-1)  # [B, F, 1, 1]
            encoded = (x * gen).sum(dim=1).to(torch.complex64)  # [B, 3, 3]
            U_enc = torch.matrix_exp(one_j * encoded)
            state = torch.matmul(U_enc, state)
            
            # Variational
            gen = self.gens_var.unsqueeze(0) 
            w = self.weights[i].unsqueeze(0).expand(B, -1).unsqueeze(-1).unsqueeze(-1)
            encoded = (w * gen).sum(dim=1).to(torch.complex64)
            U_var = torch.matrix_exp(one_j * encoded)
            state = torch.matmul(U_var, state)


        # ----- Density matrix -----
        rho = torch.matmul(state, state.conj().transpose(-2, -1))  # [B, 3, 3]
        rho = (rho + rho.conj().transpose(-2, -1)) / 2  # Hermitian

        # ----- Fidelities -----
        fidelities = []
        for op in self.label_ops:
            op_batch = op.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
            product = torch.matmul(rho, op_batch)  # [B, 3, 3]
            eigvals, _ = torch.linalg.eig(product)
            eigvals_real = torch.clamp(eigvals.real, min=1e-10)
            fidelities.append(torch.sum(torch.sqrt(eigvals_real), dim=1) ** 2)

        fstack = torch.stack(fidelities, dim=1)  # [B, 3]
        return fstack / fstack.sum(dim=1, keepdim=True)


# General SU(N)-----------------
class QNN_0_Prod(nn.Module):
    
    def __init__(self, num_layers, num_features, num_params, init_weights=None):
        super(QNN_0_Prod, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_params = num_params

        # Qutrit basis
        q0 = torch.tensor([[1], [0], [0]], dtype=torch.complex64)
        q1 = torch.tensor([[0], [1], [0]], dtype=torch.complex64)
        q2 = torch.tensor([[0], [0], [1]], dtype=torch.complex64)
        self.register_buffer("q0", q0)
        self.register_buffer("q1", q1)
        self.register_buffer("q2", q2)

        # Outer product helper
        gm = lambda A, B: torch.kron(A, B.T)
        
        # Gell-Mann generators
        gm1 = (gm(q0, q1) + gm(q1, q0)).to(torch.complex64)
        gm2 = (-1j * (gm(q0, q1) - gm(q1, q0))).to(torch.complex64)
        gm3 = (gm(q0, q0) - gm(q1, q1)).to(torch.complex64)
        gm4 = (gm(q0, q2) + gm(q2, q0)).to(torch.complex64)
        gm5 = (-1j * (gm(q0, q2) - gm(q2, q0))).to(torch.complex64)
        gm6 = (gm(q1, q2) + gm(q2, q1)).to(torch.complex64)
        gm7 = (-1j * (gm(q1, q2) - gm(q2, q1))).to(torch.complex64)
        gm8 = (1 / torch.sqrt(torch.tensor(3., dtype=torch.float32)) * (gm(q0, q0) + gm(q1, q1) - 2 * gm(q2, q2))).to(torch.complex64)
            
        Sz12 = (gm(q1, q1) - gm(q2, q2)).to(torch.complex64)
        Sz02 = (gm(q0, q0) - gm(q2, q2)).to(torch.complex64)
        
        # Configurations
        
        # C32
        generatorsYYY = [gm2,    # RY_01
                         gm5,    # RY_02
                         gm7]    # RY_12
        
        generatorsZZZ = [gm3,    # RZ_01
                         Sz02,   # RZ_02
                         Sz12]   # RZ_12
        
        generatorsYYYZ = [gm2,   # RY_01
                          gm5,   # RY_02
                          gm7,   # RY_12
                          gm3]   # RZ_01
        
        generatorYZYZ = [gm2,    # RY_01
                         gm3,    # RZ_01
                         gm7,    # RY_12
                         Sz12]   # RZ_12

        generators_generalU = [gm8,
                               gm3,
                               gm2,
                               gm3,
                               gm5,
                               gm3,
                               gm2,
                               gm3]

        generators_enc = generators_generalU
        generators_var = generators_generalU
     
        # Subset para codificación, todos para la parte variacional
        self.register_buffer("gens_enc", torch.stack(generators_enc[:self.num_features]))  # [F, 3, 3]
        self.register_buffer("gens_var", torch.stack(generators_var[:self.num_params]))    # [P, 3, 3]
        
        # Label projectors: |0><0|, |1><1|, |2><2|
        self.register_buffer("label_ops", torch.stack([gm(q, q) for q in [q0, q1, q2]]))

        # initial state
        self.q0 = q0

        # parameters per layer
        self.weights = nn.ParameterList([
            nn.Parameter(init_weights[i] if init_weights else torch.rand(6)*2 - 1)
            for i in range(num_layers)
        ])

    def forward(self, batch):
        batch_size = batch.shape[0]
        batch_c = batch.to(torch.cfloat)

        state = self.q0.expand(batch_size, -1, -1).clone()

        for i in range(self.num_layers):
            # Encoding
            for j in range(self.num_features):
                G = self.gens_enc[j]
                x = batch_c[:, j].view(-1, 1, 1)
                U = torch.matrix_exp(1j * x * G)
                state = torch.bmm(U, state)

            # Variational
            for j, G in enumerate(self.gens_var):
                theta = self.weights[i][j]
                U = torch.matrix_exp(1j * theta * G)
                state = torch.matmul(U, state)

        # density matrix
        rho = state @ state.conj().transpose(-2, -1)
        rho = (rho + rho.conj().transpose(-2, -1)) / 2  # ensure Hermitian

        # fidelity
        fidelities = []
        for op in self.label_ops:
            op_batch = op.unsqueeze(0).expand(batch_size, -1, -1)
            product = rho @ op_batch
            eigvals, _ = torch.linalg.eig(product)
            eigvals_real = torch.clamp(eigvals.real, min=1e-10)
            fidelities.append(torch.sum(torch.sqrt(eigvals_real), dim=1) ** 2)

        fstack = torch.stack(fidelities, dim=1)
        return fstack / fstack.sum(dim=1, keepdim=True)


#%%


'''
      ==================
            data
      ==================
'''

# create MinMaxScaler object
scaler = MinMaxScaler()


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def load_dataset(n_components):

    # dataset
    wine = load_wine()

    # features
    X_data = wine.data
    X_data = scaler.fit_transform(X_data)

    # labels
    y_data = wine.target

    pca = PCA(n_components=n_components)

    x_data = pca.fit_transform(X_data)

    x_data, y_data = shuffle(x_data, y_data, random_state=42)

    return x_data, y_data, pca.explained_variance_ratio_.sum()

#%%

'''
     =====================================
                training (prod)
     =====================================
'''

def eval_loss_full(dataloader, model):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for data, target in dataloader:
            out = model(data)
            l = cost_fidelity(out, target)
            bs = data.size(0)
            tot += float(l) * bs
            n += bs
    return tot / max(n, 1)

# Configs
N_TRIALS     = 4000
EPOCHS       = 150
BATCH_SIZE   = 32
LAYERS       = 6
LR           = 0.006
SAVE_EVERY   = 100
NUM_FEATURES = 8
NUM_PARAMS   = 5
TRIAL_OFFSET = 0

BASE_SAVE_DIR = "/Users/ccristiano/Documents/Codigos/QML/Wine/qutrit/data/SequentialTraining"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)


# Tracking (Option 1)
results_all = []
best_params_by_param = {}
best_loss = float('inf')
best_params = None
best_train_acc = 0.0
best_init_weights = None
best_training_time = None
best_so_far_rows = [] 
log_progress = []
chunk = 1


# Dataset
X, y, exp_var = load_dataset(NUM_FEATURES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)
print(f"Explained variance is {exp_var}")

for trial in range(1, N_TRIALS + 1):
    print(f"\n🎯 Trial {trial}/{N_TRIALS}...")

    time_start_trial = time.time()

    # Inicialization
    init_weights = [torch.rand(NUM_PARAMS) * 2 - 1 for _ in range(LAYERS)]
    model = QNN_0_Prod(LAYERS, NUM_FEATURES, NUM_PARAMS, init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # (Opcional)
    for p in model.parameters():
        p.data.clamp_(-2 * np.pi, 2 * np.pi)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_dl:
            optimizer.zero_grad()
            out = model(data)
            loss = cost_fidelity(out, target)   
            loss.backward()
            optimizer.step()

    # Loss over the whole dataset
    final_loss = eval_loss_full(train_dl, model)

    # Train accuracy
    train_acc, *_ = evaluate_accuracy(train_dl, model)

    # Trial training time
    time_end_trial = time.time() - time_start_trial

    # Save best trial (by loss)
    if final_loss < best_loss:
        best_loss = final_loss
        best_params = deepcopy(model.state_dict())
        best_train_acc = train_acc
        best_init_weights = deepcopy(init_weights)
        best_training_time = time_end_trial
        
    # Register
    best_so_far_rows.append({
        "Trial": trial + TRIAL_OFFSET,
        "BestLossSoFar": best_loss,
        "BestTrainAccSoFar": best_train_acc
    })

    print(f"   🔍 Final Train Loss (avg): {final_loss:.6f} | "
          f"Best so far: {best_loss:.6f} | Train Acc: {train_acc:.2f}% | "
          f"Time: {time_end_trial:.2f}s")


    # Progress log
    log_progress.append({
        "NUM_FEATURES": NUM_FEATURES,
        "Trial": trial + TRIAL_OFFSET,
        "BestTrainLoss": best_loss,
        "BestTrainAcc": best_train_acc,
        "Time": time_end_trial
    })

    # Save data 
    if trial % SAVE_EVERY == 0:
        df_log = pd.DataFrame(log_progress)
        file_name = f"{BASE_SAVE_DIR}/train_progress_NUMFEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}_{chunk}.csv"
        df_log.to_csv(file_name, index=False)

        # Reiniciar logs del bloque
        log_progress = []
        chunk += 1

# Evaluate in test dataset
model = QNN_0_Prod(LAYERS, NUM_FEATURES, NUM_PARAMS, [w.detach().clone() for w in best_init_weights])
model.load_state_dict(best_params)
test_acc, test_f1, test_prec, test_rec, test_kappa, test_matthews, _, _ = evaluate_accuracy(test_dl, model)

# Save model and metrics
torch.save(best_params, f"{BASE_SAVE_DIR}/params_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.pt")

df_test_metrics = pd.DataFrame([{
    "NUM_FEATURES": NUM_FEATURES,
    "NUM_PARAMS": NUM_PARAMS,
    "Loss": best_loss,  
    "TrainAcc": best_train_acc,
    "TestAcc": test_acc,
    "TestF1": test_f1,
    "TestPrecision": test_prec,
    "TestRecall": test_rec,
    "CohenKappa": test_kappa,
    "MatthewsCorr": test_matthews,
    "InitWeights": [w.detach().cpu().numpy().tolist() for w in best_init_weights],
    "TrainingTime": best_training_time
}])

df_test_metrics.to_csv(
    f"{BASE_SAVE_DIR}/metrics_NUMFEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.csv",
    index=False
)

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
pd.DataFrame(best_so_far_rows).to_csv(
    f"{BASE_SAVE_DIR}/best_so_far_NUMFEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.csv",
    index=False
)


#%%

import os, time
import pandas as pd
from copy import deepcopy


def eval_loss_full(dataloader, model):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for data, target in dataloader:
            out = model(data)
            l = cost_fidelity(out, target)
            bs = data.size(0)
            tot += float(l) * bs
            n += bs
    return tot / max(n, 1)


# ====== config ======
EPOCHS       = 300
BATCH_SIZE   = 32
LAYERS       = 6
LR           = 0.01
NUM_FEATURES = 5
BASE_SAVE_DIR = f"/Users/ccristiano/Documents/Codigos/QML/Wine/qutrit/data/5params"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# dataset (lo mismo que antes)
X, y, _ = load_dataset(NUM_FEATURES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

# ====== función genérica: entrena de N → N+1 ======
def train_incremental(prev_num_params, next_num_params):
    print(f"\n🚀 Entrenando de {prev_num_params} → {next_num_params} parámetros...")

    # 1) cargar óptimo de prev_num_params
    prev_pt_path = f"{BASE_SAVE_DIR}/params_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{prev_num_params}_LAYERS_{LAYERS}.pt"
    prev_state = torch.load(prev_pt_path, map_location="cpu")

    # 2) inicializar next con prev + 0
    prev_weights = [prev_state[f"weights.{i}"] for i in range(LAYERS)]
    init_weights_next = [torch.cat([w, torch.tensor([0.0])]) for w in prev_weights]

    model = QNN_0_Prod(LAYERS, NUM_FEATURES, next_num_params, init_weights_next)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) check equivalencia inicial
    loss_prev_train = eval_loss_full(train_dl,
        QNN_0_Prod(LAYERS, NUM_FEATURES, prev_num_params,
              [prev_state[f"weights.{i}"].detach().clone() for i in range(LAYERS)])
    )
    loss_init_train = eval_loss_full(train_dl, model)
    print(f"🔎 Check init: loss({prev_num_params})={loss_prev_train:.6f} vs loss({next_num_params} init)={loss_init_train:.6f}")

    # 4) entrenamiento con logging
    epoch_logs = []
    best_loss_so_far = float("inf")
    best_state = None

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_dl:
            optimizer.zero_grad()
            out = model(data)
            loss = cost_fidelity(out, target)
            loss.backward()
            optimizer.step()

        # evaluar promedio
        train_loss_epoch = eval_loss_full(train_dl, model)
        test_loss_epoch  = eval_loss_full(test_dl, model)

        epoch_logs.append({
            "Epoch": epoch,
            "TrainLossAvg": train_loss_epoch,
            "TestLossAvg": test_loss_epoch
        })

        print(f"[Epoch {epoch:03d}] train_loss={train_loss_epoch:.6f}  test_loss={test_loss_epoch:.6f}  best_train={best_loss_so_far:.6f}")

    train_time = time.time() - t0

    # 5) guardar curva y checkpoint
    curve_csv = f"{BASE_SAVE_DIR}/loss_curve_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{next_num_params}_LAYERS_{LAYERS}.csv"
    pd.DataFrame(epoch_logs).to_csv(curve_csv, index=False)
    print(f"📈 Guardé curva de loss en: {curve_csv}")

    if best_state is None:
        best_state = deepcopy(model.state_dict())
    torch.save(best_state, f"{BASE_SAVE_DIR}/params_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{next_num_params}_LAYERS_{LAYERS}.pt")

    # métricas finales
    model.load_state_dict(best_state)
    final_train_loss = eval_loss_full(train_dl, model)
    final_test_loss  = eval_loss_full(test_dl,  model)
    train_acc, *_ = evaluate_accuracy(train_dl, model)
    test_acc,  test_f1, test_prec, test_rec, test_kappa, test_matthews, *_ = evaluate_accuracy(test_dl, model)

    pd.DataFrame([{
        "NUM_FEATURES": NUM_FEATURES,
        "NUM_PARAMS": next_num_params,
        "Loss": final_train_loss,
        "FinalTestLossAvg": final_test_loss,
        "TrainAcc": train_acc,
        "TestAcc": test_acc,
        "TestF1": test_f1,
        "TestPrecision": test_prec,
        "TestRecall": test_rec,
        "CohenKappa": test_kappa,
        "MatthewsCorr": test_matthews,
        "TrainTimeSec": train_time
    }]).to_csv(
        f"{BASE_SAVE_DIR}/metrics_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{next_num_params}_LAYERS_{LAYERS}.csv",
        index=False
    )
    print(f"✅ Guardé checkpoint + métricas de {next_num_params} parámetros.")


for j in range(5, 8):
    train_incremental(j, j+1)



#%%



'''
     =====================================
                training (prod)
     =====================================
'''

def eval_loss_full(dataloader, model):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for data, target in dataloader:
            out = model(data)
            l = cost_fidelity(out, target)
            bs = data.size(0)
            tot += float(l) * bs
            n += bs
    return tot / max(n, 1)

# Configs
N_TRIALS     = 4000
EPOCHS       = 150
BATCH_SIZE   = 32
LAYERS       = 6
LR           = 0.006
SAVE_EVERY   = 100
NUM_FEATURES = 4
NUM_PARAMS   = 4
TRIAL_OFFSET = 0

BASE_SAVE_DIR = "/Users/ccristiano/Documents/Codigos/QML/Wine/qutrit/data/SequentialTraining"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)


# Tracking (Option 1)
results_all = []
best_params_by_param = {}
best_loss = float('inf')
best_params = None
best_train_acc = 0.0
best_init_weights = None
best_training_time = None
best_so_far_rows = [] 
log_progress = []
chunk = 1


# Dataset
X, y, exp_var = load_dataset(NUM_FEATURES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)
print(f"Explained variance is {exp_var}")

for trial in range(1, N_TRIALS + 1):
    print(f"\n🎯 Trial {trial}/{N_TRIALS}...")

    time_start_trial = time.time()

    # Inicialization
    init_weights = [torch.rand(NUM_PARAMS) * 2 - 1 for _ in range(LAYERS)]
    model = QNN_0_Prod(LAYERS, NUM_FEATURES, NUM_PARAMS, init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # (Opcional)
    for p in model.parameters():
        p.data.clamp_(-2 * np.pi, 2 * np.pi)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_dl:
            optimizer.zero_grad()
            out = model(data)
            loss = cost_fidelity(out, target)   
            loss.backward()
            optimizer.step()

    # Loss over the whole dataset
    final_loss = eval_loss_full(train_dl, model)

    # Train accuracy
    train_acc, *_ = evaluate_accuracy(train_dl, model)

    # Trial training time
    time_end_trial = time.time() - time_start_trial

    # Save best trial (by loss)
    if final_loss < best_loss:
        best_loss = final_loss
        best_params = deepcopy(model.state_dict())
        best_train_acc = train_acc
        best_init_weights = deepcopy(init_weights)
        best_training_time = time_end_trial
        
    # Register
    best_so_far_rows.append({
        "Trial": trial + TRIAL_OFFSET,
        "BestLossSoFar": best_loss,
        "BestTrainAccSoFar": best_train_acc
    })

    print(f"   🔍 Final Train Loss (avg): {final_loss:.6f} | "
          f"Best so far: {best_loss:.6f} | Train Acc: {train_acc:.2f}% | "
          f"Time: {time_end_trial:.2f}s")


    # Progress log
    log_progress.append({
        "NUM_FEATURES": NUM_FEATURES,
        "Trial": trial + TRIAL_OFFSET,
        "BestTrainLoss": best_loss,
        "BestTrainAcc": best_train_acc,
        "Time": time_end_trial
    })

    # Save data 
    if trial % SAVE_EVERY == 0:
        df_log = pd.DataFrame(log_progress)
        file_name = f"{BASE_SAVE_DIR}/train_progress_NUMFEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}_{chunk}.csv"
        df_log.to_csv(file_name, index=False)

        # Reiniciar logs del bloque
        log_progress = []
        chunk += 1

# Evaluate in test dataset
model = QNN_0_Prod(LAYERS, NUM_FEATURES, NUM_PARAMS, [w.detach().clone() for w in best_init_weights])
model.load_state_dict(best_params)
test_acc, test_f1, test_prec, test_rec, test_kappa, test_matthews, _, _ = evaluate_accuracy(test_dl, model)

# Save model and metrics
torch.save(best_params, f"{BASE_SAVE_DIR}/params_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.pt")

df_test_metrics = pd.DataFrame([{
    "NUM_FEATURES": NUM_FEATURES,
    "NUM_PARAMS": NUM_PARAMS,
    "Loss": best_loss,  
    "TrainAcc": best_train_acc,
    "TestAcc": test_acc,
    "TestF1": test_f1,
    "TestPrecision": test_prec,
    "TestRecall": test_rec,
    "CohenKappa": test_kappa,
    "MatthewsCorr": test_matthews,
    "InitWeights": [w.detach().cpu().numpy().tolist() for w in best_init_weights],
    "TrainingTime": best_training_time
}])

df_test_metrics.to_csv(
    f"{BASE_SAVE_DIR}/metrics_NUMFEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.csv",
    index=False
)

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
pd.DataFrame(best_so_far_rows).to_csv(
    f"{BASE_SAVE_DIR}/best_so_far_NUMFEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.csv",
    index=False
)

#%%

import torch, time, pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# --- Config rápido ---
EPOCHS       = 500
BATCH_SIZE   = 32
LAYERS       = 6
LR           = 0.005
NUM_FEATURES = 4
NUM_PARAMS   = 4

def eval_loss_full(dataloader, model):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for data, target in dataloader:
            out = model(data)
            l = cost_fidelity(out, target)
            bs = data.size(0)
            tot += float(l) * bs
            n += bs
    return tot / max(n, 1)


# Dataset
X, y, exp_var = load_dataset(NUM_FEATURES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

print(f"Explained variance is {exp_var}")

# --- Inicialización única ---
init_weights = [torch.rand(NUM_PARAMS) * 2 - 1 for _ in range(LAYERS)]
model = QNN_0_Sum(LAYERS, NUM_FEATURES, NUM_PARAMS, init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Loop de entrenamiento con log de pérdida global por epoch ---
epoch_losses = []
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    for data, target in train_dl:
        optimizer.zero_grad()
        out = model(data)
        loss = cost_fidelity(out, target)
        loss.backward()
        optimizer.step()
    # evalúa pérdida en TODO el train set
    avg_loss = eval_loss_full(train_dl, model)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1:03d}/{EPOCHS} - Train Loss: {avg_loss:.6f}")

t1 = time.time()
print(f"Entrenamiento terminado en {t1 - t0:.2f}s")

# --- Métricas finales ---
train_acc, *_ = evaluate_accuracy(train_dl, model)
test_acc, *_  = evaluate_accuracy(test_dl,  model)
print(f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# --- Guardar y graficar ---
pd.DataFrame({"epoch": range(1, EPOCHS + 1), "train_loss": epoch_losses}) \
  .to_csv("train_loss_curve.csv", index=False)

plt.figure(figsize=(7,4))
plt.plot(range(1, EPOCHS + 1), epoch_losses, marker="o", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Train Loss (full set)")
plt.title("Curva de pérdida durante el entrenamiento")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



#%%

import pickle

def train_incremental(prev_num_params, next_num_params):
    print(f"\n🚀 Entrenando de {prev_num_params} → {next_num_params} parámetros...")

    # === 1) Cargar el mejor estado desde el archivo .pkl de prev_num_params ===
    pkl_path = (f"{BASE_SAVE_DIR}/results_NUM_FEATURES_{NUM_FEATURES}_"
                f"NUM_PARAMS_{prev_num_params}_LAYERS_{LAYERS}.pkl")
    with open(pkl_path, "rb") as f:
        trials = pickle.load(f)

    # trial con menor loss
    best_trial = min(trials, key=lambda r: r["loss"])
    prev_state_dict = best_trial["state_dict"]

    # Si querés las init_weights originales
    prev_init_weights = best_trial["init_weights"]

    # === 2) Preparar pesos iniciales para next_num_params ===
    # Cada capa tiene que crecer en 1 parámetro: concatenamos un 0
    prev_weights = [prev_state_dict[f"weights.{i}"] for i in range(LAYERS)]
    init_weights_next = [torch.cat([w, torch.tensor([0.0])]) for w in prev_weights]

    # Crear el modelo “next”
    model = QNN_0_Prod(LAYERS, NUM_FEATURES, next_num_params, init_weights_next)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # (El resto del entrenamiento queda igual…)


#%%

import os, time
import pandas as pd
from copy import deepcopy


def eval_loss_full(dataloader, model):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for data, target in dataloader:
            out = model(data)
            l = cost_fidelity(out, target)
            bs = data.size(0)
            tot += float(l) * bs
            n += bs
    return tot / max(n, 1)


# ====== config ======
EPOCHS       = 1000
BATCH_SIZE   = 32
LAYERS       = 6
LR           = 0.0005
NUM_FEATURES = 8
BASE_SAVE_DIR = f"/Users/ccristiano/Documents/Codigos/QML/Wine/qutrit/data/4params_sum_2"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# dataset (lo mismo que antes)
X, y, _ = load_dataset(NUM_FEATURES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)


import pickle

def train_incremental(prev_num_params, next_num_params):
    print(f"\n🚀 Entrenando de {prev_num_params} → {next_num_params} parámetros...")

    # === 1) Cargar el mejor estado desde el archivo .pkl de prev_num_params ===
    pkl_path = (f"{BASE_SAVE_DIR}/results_NUM_FEATURES_{NUM_FEATURES}_"
                f"NUM_PARAMS_{prev_num_params}_LAYERS_{LAYERS}.pkl")
    with open(pkl_path, "rb") as f:
        trials = pickle.load(f)

    # trial con menor loss
    best_trial = min(trials, key=lambda r: r["loss"])
    prev_state_dict = best_trial["params_opt"]


    # === 2) Preparar pesos iniciales para next_num_params ===
    # Cada capa tiene que crecer en 1 parámetro: concatenamos un 0
    prev_weights = [prev_state_dict[f"weights.{i}"] for i in range(LAYERS)]
    init_weights_next = [torch.cat([w, torch.tensor([0.0])]) for w in prev_weights]

    # Crear el modelo “next”
    model = QNN_0_Sum(LAYERS, NUM_FEATURES, next_num_params, init_weights_next)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    
    # 3) check equivalencia inicial
    loss_prev_train = eval_loss_full(train_dl,
        QNN_0_Prod(LAYERS, NUM_FEATURES, prev_num_params,
              [prev_state_dict[f"weights.{i}"].detach().clone() for i in range(LAYERS)])
    )
    loss_init_train = eval_loss_full(train_dl, model)
    print(f"🔎 Check init: loss({prev_num_params})={loss_prev_train:.6f} vs loss({next_num_params} init)={loss_init_train:.6f}")

    # 4) entrenamiento con logging
    epoch_logs = []


    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_dl:
            optimizer.zero_grad()
            out = model(data)
            loss = cost_fidelity(out, target)
            loss.backward()
            optimizer.step()

        # evaluar promedio
        train_loss_epoch = eval_loss_full(train_dl, model)
        train_acc, *_ = evaluate_accuracy(train_dl, model)

        epoch_logs.append({
            "Epoch": epoch,
            "train_loss": train_loss_epoch,
            "train_acc": train_acc
        })

        print(f"[Epoch {epoch:03d}] train_loss={train_loss_epoch:.6f}  train_acc={train_acc:.6f}")

    train_time = time.time() - t0

    # 5) guardar curva y checkpoint
    curve_csv = f"{BASE_SAVE_DIR}/loss_curve_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{next_num_params}_LAYERS_{LAYERS}.csv"
    pd.DataFrame(epoch_logs).to_csv(curve_csv, index=False)
    print(f"📈 Guardé curva de loss en: {curve_csv}")

    best_state = deepcopy(model.state_dict())
    torch.save(best_state, f"{BASE_SAVE_DIR}/params_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{next_num_params}_LAYERS_{LAYERS}.pt")

    # métricas finales
    model.load_state_dict(best_state)
    final_train_loss = eval_loss_full(train_dl, model)
    final_test_loss  = eval_loss_full(test_dl,  model)
    train_acc, *_ = evaluate_accuracy(train_dl, model)
    test_acc,  test_f1, test_prec, test_rec, test_kappa, test_matthews, *_ = evaluate_accuracy(test_dl, model)

    pd.DataFrame([{
        "NUM_FEATURES": NUM_FEATURES,
        "NUM_PARAMS": next_num_params,
        "Loss": final_train_loss,
        "FinalTestLossAvg": final_test_loss,
        "TrainAcc": train_acc,
        "TestAcc": test_acc,
        "TestF1": test_f1,
        "TestPrecision": test_prec,
        "TestRecall": test_rec,
        "CohenKappa": test_kappa,
        "MatthewsCorr": test_matthews,
        "TrainTimeSec": train_time
    }]).to_csv(
        f"{BASE_SAVE_DIR}/metrics_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{next_num_params}_LAYERS_{LAYERS}.csv",
        index=False
    )
    print(f"✅ Guardé checkpoint + métricas de {next_num_params} parámetros.")


    pkl_metrics = {
    "trial": 1,  # o el número que quieras (aquí solo 1 trial)
    "loss": final_train_loss,
    "train_acc": train_acc,
    "params_opt": best_state,
    "init_weights": [w.detach().clone() for w in init_weights_next],
    "train_time": train_time,
    }
    pkl_path = (
        f"{BASE_SAVE_DIR}/results_NUM_FEATURES_{NUM_FEATURES}_"
        f"NUM_PARAMS_{next_num_params}_LAYERS_{LAYERS}.pkl"
        )
    with open(pkl_path, "wb") as f:
        pickle.dump([pkl_metrics], f)   # lista de un solo dict, como en los barridos
        print(f"💾 Guardé resultados .pkl en: {pkl_path}")


for j in range(4, 8):
    train_incremental(j, j+1)
    
    
#%%

