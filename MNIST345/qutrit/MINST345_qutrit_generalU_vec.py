#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:43:56 2025

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
from sklearn.utils import shuffle
from copy import deepcopy
from IPython.display import clear_output


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
        
def cost(output, target):
    return torch.mean((1 - output[range(len(target)), target]) ** 2)




#%%
'''
     ==================
        1 qutrit QNN
     ==================
'''

class QNN_0(nn.Module):

    def __init__(self, num_layers, num_features, num_params, init_weights=None):
        super(QNN_0, self).__init__()

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
        generators = [
            (gm(q0, q1) + gm(q1, q0)).to(torch.complex64),
            (-1j * (gm(q0, q1) - gm(q1, q0))).to(torch.complex64),
            (gm(q0, q0) - gm(q1, q1)).to(torch.complex64),
            (gm(q0, q2) + gm(q2, q0)).to(torch.complex64),
            (-1j * (gm(q0, q2) - gm(q2, q0))).to(torch.complex64),
            (gm(q1, q2) + gm(q2, q1)).to(torch.complex64),
            (-1j * (gm(q1, q2) - gm(q2, q1))).to(torch.complex64),
            (1 / torch.sqrt(torch.tensor(3., dtype=torch.float32)) * (gm(q0, q0) + gm(q1, q1) - 2 * gm(q2, q2))).to(torch.complex64)
            ]

        self.register_buffer("gens_enc", torch.stack(generators[:self.num_features]))  # [F, 3, 3]
        self.register_buffer("gens_var", torch.stack(generators[:self.num_params]))  # [F, 3, 3]
        
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



# ## General SU(N)-----------------
# class QNN_0(nn.Module):
#     def __init__(self, num_layers, num_features, init_weights=None):
#         super().__init__()
#         self.num_layers = num_layers
#         self.num_features = num_features

#         # Qutrit basis
#         q0 = torch.tensor([[1], [0], [0]], dtype=torch.complex64)
#         q1 = torch.tensor([[0], [1], [0]], dtype=torch.complex64)
#         q2 = torch.tensor([[0], [0], [1]], dtype=torch.complex64)
#         self.register_buffer("q0", q0)
#         self.register_buffer("q1", q1)
#         self.register_buffer("q2", q2)

#         # Outer product helper
#         gm = lambda A, B: torch.kron(A, B.T)
        
#         # Gell-Mann generators
#         gm1 = (gm(q0, q1) + gm(q1, q0)).to(torch.complex64)
#         gm2 = (-1j * (gm(q0, q1) - gm(q1, q0))).to(torch.complex64)
#         gm3 = (gm(q0, q0) - gm(q1, q1)).to(torch.complex64)
#         gm4 = (gm(q0, q2) + gm(q2, q0)).to(torch.complex64)
#         gm5 = (-1j * (gm(q0, q2) - gm(q2, q0))).to(torch.complex64)
#         gm6 = (gm(q1, q2) + gm(q2, q1)).to(torch.complex64)
#         gm7 = (-1j * (gm(q1, q2) - gm(q2, q1))).to(torch.complex64)
#         gm8 = (1 / torch.sqrt(torch.tensor(3., dtype=torch.float32)) * (gm(q0, q0) + gm(q1, q1) - 2 * gm(q2, q2))).to(torch.complex64)
                
#         # Gell-Mann generators
#         generators = [
#                     gm3,
#                     gm2,
#                     gm3,
#                     gm5,
#                     gm3,
#                     gm2,
#                     gm3,
#                     gm8]
     
#         # Subset para codificación, todos para la parte variacional
#         self.register_buffer("gens_enc", torch.stack(generators[:self.num_features]))  # [F, 3, 3]
#         self.register_buffer("gens_var", torch.stack(generators))  # [8, 3, 3]
        
#         # Label projectors: |0><0|, |1><1|, |2><2|
#         self.register_buffer("label_ops", torch.stack([gm(q, q) for q in [q0, q1, q2]]))

#         # initial state
#         self.q0 = q0

#         # parameters per layer
#         self.weights = nn.ParameterList([
#             nn.Parameter(init_weights[i] if init_weights else torch.rand(6)*2 - 1)
#             for i in range(num_layers)
#         ])

#     def forward(self, batch):
#         batch_size = batch.shape[0]
#         batch_c = batch.to(torch.cfloat)

#         state = self.q0.expand(batch_size, -1, -1).clone()

#         for i in range(self.num_layers):
#             # Encoding: apply each encoding generator separately
#             for j in range(self.num_features):
#                 G = self.generators_var[j]
#                 x = batch_c[:, j].view(-1, 1, 1)  # shape: (batch, 1, 1)
#                 U = torch.matrix_exp(1j * x * G)  # broadcasting over batch
#                 state = torch.bmm(U, state)

#             # Variational: apply each generator separately with its own param
#             for j, G in enumerate(self.generators_var):
#                 theta = self.weights[i][j]
#                 U = torch.matrix_exp(1j * theta * G)
#                 state = torch.matmul(U, state)

#         # density matrix
#         rho = state @ state.conj().transpose(-2, -1)
#         rho = (rho + rho.conj().transpose(-2, -1)) / 2  # ensure Hermitian

#         # fidelity
#         fidelities = []
#         for op in self.label_ops:
#             op_batch = op.unsqueeze(0).expand(batch_size, -1, -1)
#             product = rho @ op_batch
#             eigvals, _ = torch.linalg.eig(product)
#             eigvals_real = torch.clamp(eigvals.real, min=1e-10)
#             fidelities.append(torch.sum(torch.sqrt(eigvals_real), dim=1) ** 2)

#         fstack = torch.stack(fidelities, dim=1)
#         return fstack / fstack.sum(dim=1, keepdim=True)

#%%
'''
     ==================
        load dataset
     ==================
'''

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
    
    digits = datasets.load_digits()

    # 345 ---------------------------------------------
    threes_x = digits["images"][digits["target"]==3]
    fours_x  = digits["images"][digits["target"]==4]
    fives_x  = digits["images"][digits["target"]==5]

    threes_y  = digits["target"][digits["target"]==3]
    fours_y   = digits["target"][digits["target"]==4]
    fives_y = digits["target"][digits["target"]==5]
    
    threes_y[threes_y == 3] = 0
    fours_y[fours_y == 4] = 1
    fives_y[fives_y == 5] = 2

    x_data = np.vstack((threes_x, fours_x))
    x_data = np.vstack((x_data, fives_x))

    y_data = np.hstack((threes_y, fours_y))
    y_data = np.hstack((y_data, fives_y))

    pca = PCA(n_components=n_components)

    x_data = pca.fit_transform(x_data.reshape(x_data.shape[0], 64))
    x_data = x_data/np.amax(x_data)

    x_data, y_data = shuffle(x_data, y_data, random_state=42)

    return x_data, y_data, pca.explained_variance_ratio_.sum()


#%%
# '''
#      ============================
#           training (1 copie)
#      ============================
# '''

# # Configuraciones fijas
# N_TRIALS         = 400
# EPOCHS           = 100
# BATCH_SIZE       = 32
# LAYERS           = 5
# LR               = 0.006
# number_of_copies = 1

# # Loop sobre distintos valores de NUM_FEATURES
# for NUM_FEATURES in range(4, 9):  # 4 a 8 inclusive
#     print(f"\n🧪 Iniciando entrenamiento con NUM_FEATURES = {NUM_FEATURES}...\n")
    
#     plt.ion()  # Modo interactivo
#     fig, ax = plt.subplots(figsize=(8, 4))  # Una sola figura persistente
#     line, = ax.plot([], [], marker='o')
#     ax.set_xlabel("Trial")
#     ax.set_ylabel("Best Train Loss so far")
#     ax.set_title(f"Live Loss | Features: {NUM_FEATURES}")
#     ax.grid(True)

#     # Cargar y dividir datos
#     X, y, _ = load_dataset(NUM_FEATURES)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#         )

#     train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
#     test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

#     best_loss = float("inf")
#     best_params = None
#     best_init = None
#     best_train_acc = 0.0
#     loss_progress = []
#     log_progress = []

#     start_total = time.time()

#     for trial in range(1, N_TRIALS + 1):
#         init_weights = [torch.rand(NUM_FEATURES)*2*torch.pi - 1 for _ in range(LAYERS)]
#         model = QNN_0(LAYERS, NUM_FEATURES, init_weights)
#         optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        

#         for p in model.parameters():
#             p.data.clamp_(-4*np.pi, 4*np.pi)

#         for epoch in range(EPOCHS):
#             model.train()
#             for data, target in train_dl:
#                 optimizer.zero_grad()
#                 out = model(data)
#                 loss = cost(out, target)
#                 loss.backward()
#                 optimizer.step()

#         final_loss = float(loss.detach())
#         train_acc, *_ = evaluate_accuracy(train_dl, model)

#         if final_loss < best_loss:
#             best_loss = final_loss
#             best_params = deepcopy(model.state_dict())
#             best_init = init_weights
#             best_train_acc = train_acc

#         # Guardar progreso acumulado
#         loss_progress.append(best_loss)
#         log_progress.append({
#             "NUM_FEATURES": NUM_FEATURES,
#             "Trial": trial,
#             "BestTrainLoss": best_loss,
#             "BestTrainAcc": best_train_acc
#         })
        
#         # Mostrar progreso cada 100 trials
#         if trial % 100 == 0:
#             print(f"📌 Trial {trial}/{N_TRIALS} | Best Loss: {best_loss:.6f} | Train Acc: {best_train_acc:.2f}%")


#         # Gráfico en vivo
#         # clear_output(wait=True)
#         # plt.figure(figsize=(8, 4))
#         # plt.plot(range(1, len(loss_progress)+1), loss_progress, marker='o')
#         # plt.xlabel("Trial")
#         # plt.ylabel("Best Train Loss so far")
#         # plt.title(f"Live Loss | Features: {NUM_FEATURES}")
#         # plt.grid(True)
#         # plt.tight_layout()
#         # plt.show()
#         line.set_data(range(1, len(loss_progress) + 1), loss_progress)
#         ax.relim()
#         ax.autoscale_view()
#         fig.canvas.draw()
#         fig.canvas.flush_events()

#     # Evaluación final en test
#     print(f"\n🏁 Finalizando modelo para NUM_FEATURES = {NUM_FEATURES}")
#     print(f"🏆 Mejor Train Loss: {best_loss:.6f} | Train Acc: {best_train_acc:.4f}")

#     final_model = QNN_0(LAYERS, NUM_FEATURES, None)
#     final_model.load_state_dict(best_params)
#     final_model.eval()
#     test_acc, f1, prec, rec, kappa, mcc, y_true, y_pred = evaluate_accuracy(test_dl, final_model)

#     print(f"\n🧪 Test Accuracy: {test_acc:.4f}")
#     print(f"F1: {f1:.4f}, MCC: {mcc:.4f}")
#     print(f"⏱️ Tiempo total: {time.time() - start_total:.2f} s")

#     # Guardar CSV con el log de progreso
#     df_log = pd.DataFrame(log_progress)
#     file_name = f"/Users/ccristiano/Documents/Codigos/QuantumMachineLearning3/MINST/Data/1qutrit/MultipleStarts/loss_trainacc _generalU_dynamicalparams_vec/train_progress_NUMFEATURES_{NUM_FEATURES}_100epochs_2.csv"
#     df_log.to_csv(file_name, index=False)


#%%
'''
     ====================================
          training (save every n trials)
     ====================================
'''

# Configuraciones fijas
N_TRIALS         = 4000
EPOCHS           = 100
BATCH_SIZE       = 32
LAYERS           = 5
LR               = 0.006
# NUM_PARAMS       = 8
NUM_FEATURES     = 5
SAVE_EVERY       = 200
    

# Loop sobre distintos valores de NUM_FEATURES
for NUM_PARAMS in range(4, 8):
    print(f"\n🧪 Iniciando entrenamiento con NUM_PARAMS = {NUM_PARAMS}...\n")
    
    # Cargar y dividir datos
    X, y, _ = load_dataset(NUM_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

    # Inicializar variables de tracking del chunk actual
    best_loss = float("inf")
    best_params = None
    best_init = None
    best_train_acc = 0.0
    loss_progress = []
    log_progress = []

    start_total = time.time()

    # Configurar gráfico interactivo persistente
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [], marker='o')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Train Loss so far")
    ax.set_title(f"Live Loss | Features: {NUM_FEATURES} | Params: {NUM_PARAMS}")
    ax.grid(True)
    
    chunk = 1 

    for trial in range(1, N_TRIALS + 1):
        print(f"\n🚀 Trial {trial}/{N_TRIALS} (Features: {NUM_FEATURES})")

        # Inicialización aleatoria de pesos
        init_weights = [torch.rand(NUM_PARAMS) * 2 - 1 for _ in range(LAYERS)]
        model = QNN_0(LAYERS, NUM_FEATURES, NUM_PARAMS, init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for p in model.parameters():
            p.data.clamp_(-2 * np.pi, 2 * np.pi)

        # Entrenamiento por EPOCHS
        for epoch in range(EPOCHS):
            model.train()
            for data, target in train_dl:
                optimizer.zero_grad()
                out = model(data)
                loss = cost(out, target)
                loss.backward()
                optimizer.step()

        final_loss = float(loss.detach())
        train_acc, *_ = evaluate_accuracy(train_dl, model)
        print(f"  ✅ Train Acc = {train_acc:.4f}, Loss = {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = deepcopy(model.state_dict())
            best_init = init_weights
            best_train_acc = train_acc
            print("  🔝 Nuevo mejor modelo!")
        else:
            print("  ➖ No mejoró.")

        # Guardar progreso del bloque actual
        loss_progress.append(best_loss)
        log_progress.append({
            "NUM_FEATURES": NUM_FEATURES,
            "Trial": trial,
            "BestTrainLoss": best_loss,
            "BestTrainAcc": best_train_acc
        })

        # Actualizar gráfico en vivo
        line.set_data(range(1, len(loss_progress) + 1), loss_progress)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # === Guardado cada SAVE_EVERY trials ===
        if trial % SAVE_EVERY == 0:
            df_log = pd.DataFrame(log_progress)
            start_t = (chunk - 1) * SAVE_EVERY + 1
            end_t = chunk * SAVE_EVERY
            file_name = (
                f"/Users/ccristiano/Documents/Codigos/QuantumMachineLearning3/MINST/Data/1qutrit/TestConvergence/"
                f"train_progress_NUMFEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_{chunk}.csv"
            )
            df_log.to_csv(file_name, index=False)

            # Reiniciar logs del bloque
            log_progress = []
            loss_progress = []

            chunk += 1

    

#%%
# Configuraciones fijas
N_TRIALS         = 200
EPOCHS           = 100
BATCH_SIZE       = 32
LAYERS           = 5
LR               = 0.006
number_of_copies = 15

NUM_PARAMS       = 8

# Loop sobre distintos valores de NUM_FEATURES
for NUM_FEATURES in range(8, 9): 
    print(f"\n🧪 Iniciando entrenamiento con NUM_FEATURES = {NUM_FEATURES}...\n")
    
    # Cargar y dividir datos
    X, y, _ = load_dataset(NUM_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(CustomDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

    for copie in range(number_of_copies+1):

        print(f"Number of copie: {copie}")

        best_loss = float("inf")
        best_params = None
        best_init = None
        best_train_acc = 0.0
        loss_progress = []
        log_progress = []

        start_total = time.time()
        
        plt.ion()  # Modo interactivo
        fig, ax = plt.subplots(figsize=(8, 4))  # Una sola figura persistente
        line, = ax.plot([], [], marker='o')
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best Train Loss so far")
        ax.set_title(f"Live Loss | Features: {NUM_FEATURES}")
        ax.grid(True)

        for trial in range(1, N_TRIALS + 1):
            print(f"\n🚀 Trial {trial}/{N_TRIALS} (Features: {NUM_FEATURES})")
            init_weights = [torch.rand(NUM_PARAMS)*2 - 1 for _ in range(LAYERS)]
            model = QNN_0(LAYERS, NUM_FEATURES, init_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            for p in model.parameters():
                p.data.clamp_(-4*np.pi, 4*np.pi)

            for epoch in range(EPOCHS):
                model.train()
                for data, target in train_dl:
                    optimizer.zero_grad()
                    out = model(data)
                    loss = cost(out, target)
                    loss.backward()
                    optimizer.step()

            final_loss = float(loss.detach())
            train_acc, *_ = evaluate_accuracy(train_dl, model)
            print(f"  ✅ Train Acc = {train_acc:.4f}, Loss = {final_loss:.6f}")

            if final_loss < best_loss:
                best_loss = final_loss
                best_params = deepcopy(model.state_dict())
                best_init = init_weights
                best_train_acc = train_acc
                print("  🔝 Nuevo mejor modelo!")
            else:
                print("  ➖ No mejoró.")

            # Guardar progreso acumulado
            loss_progress.append(best_loss)
            log_progress.append({
                "NUM_FEATURES": NUM_FEATURES,
                "Trial": trial,
                "BestTrainLoss": best_loss,
                "BestTrainAcc": best_train_acc
            })

            # Gráfico en vivo
            line.set_data(range(1, len(loss_progress) + 1), loss_progress)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Evaluación final en test
        print(f"\n🏁 Finalizando modelo para NUM_FEATURES = {NUM_FEATURES}")
        print(f"🏆 Mejor Train Loss: {best_loss:.6f} | Train Acc: {best_train_acc:.4f}")

        final_model = QNN_0(LAYERS, NUM_FEATURES, None)
        final_model.load_state_dict(best_params)
        final_model.eval()
        test_acc, f1, prec, rec, kappa, mcc, y_true, y_pred = evaluate_accuracy(test_dl, final_model)

        print(f"\n🧪 Test Accuracy: {test_acc:.4f}")
        print(f"F1: {f1:.4f}, MCC: {mcc:.4f}")
        print(f"⏱️ Tiempo total: {time.time() - start_total:.2f} s")

        # Guardar CSV con el log de progreso
        df_log = pd.DataFrame(log_progress)
        file_name = f"/Users/ccristiano/Documents/Codigos/QuantumMachineLearning3/MINST/Data/1qutrit/TestConvergence/train_progress_NUMFEATURES_{NUM_FEATURES}_100epochs_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_{copie}.csv"
        df_log.to_csv(file_name, index=False)


        
        
#%%


'''
    Train with progressive initialization (weights extended from previous training)
'''

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from IPython.display import clear_output
import time
import os

# Configuraciones fijas
N_TRIALS = 3000 
EPOCHS = 100
BATCH_SIZE = 32
LAYERS = 5
LR = 0.006

# Ruta base para guardar logs
base_path = "/Users/ccristiano/Documents/Codigos/QuantumMachineLearning3/MINST/Data/1qutrit/MultipleStarts/loss_trainacc _generalU_dynamicalparams_vec_progressive/"

init_weights_best = None  # Para entrenamiento progresivo

# Loop sobre distintos valores de NUM_FEATURES
for NUM_FEATURES in range(4, 9):  # de 4 a 8 inclusive
    print(f"\n🧪 Iniciando entrenamiento con NUM_FEATURES = {NUM_FEATURES}...\n")
    
    plt.ion()  # Modo interactivo
    fig, ax = plt.subplots(figsize=(8, 4))  # Una sola figura persistente
    line, = ax.plot([], [], marker='o')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Train Loss so far")
    ax.set_title(f"Live Loss | Features: {NUM_FEATURES}")
    ax.grid(True)

    # Cargar y dividir datos
    X, y, _ = load_dataset(NUM_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_dl = DataLoader(CustomDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(CustomDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    best_loss = float("inf")
    best_params = None
    best_train_acc = 0.0
    loss_progress = []
    log_progress = []
    init_weights_log = []

    start_total = time.time()

    for trial in range(1, N_TRIALS + 1):
        print(f"\n🚀 Trial {trial}/{N_TRIALS} (Features: {NUM_FEATURES})")

        if init_weights_best is not None:
            init_weights = [
                torch.cat([w, torch.rand(1) * 2 * torch.pi - 1])
                for w in init_weights_best
            ]
            init_type = "from_trained"
            print("↪️ Inicializado con parámetros entrenados extendidos.")
        else:
            init_weights = [torch.rand(NUM_FEATURES) * 2 * torch.pi - 1 for _ in range(LAYERS)]
            init_type = "random"

        model = QNN_0(LAYERS, NUM_FEATURES, init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        for p in model.parameters():
            p.data.clamp_(-4 * np.pi, 4 * np.pi)

        for epoch in range(EPOCHS):
            model.train()
            for data, target in train_dl:
                optimizer.zero_grad()
                out = model(data)
                loss = cost(out, target)
                loss.backward()
                optimizer.step()

        final_loss = float(loss.detach())
        train_acc, *_ = evaluate_accuracy(train_dl, model)
        print(f"  ✅ Train Acc = {train_acc:.4f}, Loss = {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = deepcopy(model.state_dict())
            best_train_acc = train_acc
            print("  🔝 Nuevo mejor modelo!")

        loss_progress.append(best_loss)
        log_progress.append({
            "NUM_FEATURES": NUM_FEATURES,
            "Trial": trial,
            "BestTrainLoss": best_loss,
            "BestTrainAcc": best_train_acc,
        })

        # clear_output(wait=True)
        # plt.figure(figsize=(8, 4))
        # plt.plot(range(1, len(loss_progress) + 1), loss_progress, marker='o')
        # plt.xlabel("Trial")
        # plt.ylabel("Best Train Loss so far")
        # plt.title(f"Live Loss | Features: {NUM_FEATURES}")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        line.set_data(range(1, len(loss_progress) + 1), loss_progress)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Guardar pesos iniciales reales del modelo
        for name, param in model.named_parameters():
            init_weights_log.append({
                "Trial": trial,
                "Layer": name,
                "Weights": param.data.detach().cpu().numpy().tolist(),
                "NUM_FEATURES": NUM_FEATURES,
                "InitType": init_type
            })

    final_model = QNN_0(LAYERS, NUM_FEATURES, None)
    final_model.load_state_dict(best_params)
    final_model.eval()
    test_acc, f1, prec, rec, kappa, mcc, y_true, y_pred = evaluate_accuracy(test_dl, final_model)

    print(f"\n🏁 Finalizado NUM_FEATURES = {NUM_FEATURES} | Train Acc: {best_train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"F1: {f1:.4f}, MCC: {mcc:.4f}")
    print(f"⏱️ Tiempo total: {time.time() - start_total:.2f} s")

    df_log = pd.DataFrame(log_progress)
    df_log.to_csv(os.path.join(base_path, f"train_progress_NUMFEATURES_{NUM_FEATURES}.csv"), index=False)

    df_init = pd.DataFrame(init_weights_log)
    df_init.to_csv(os.path.join(base_path, f"init_weights_NUMFEATURES_{NUM_FEATURES}.csv"), index=False)

    print(f"📁 Guardado log y pesos de inicialización para NUM_FEATURES = {NUM_FEATURES}")

    # Extraer pesos ENTRENADOS del mejor modelo para extender
    trained_model = QNN_0(LAYERS, NUM_FEATURES, None)
    trained_model.load_state_dict(best_params)
    trained_model.eval()

    init_weights_best = [
        trained_model.weights[i].data.detach().clone()
        for i in range(LAYERS)
    ]



#%%
