#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:17:30 2025

@author: ccristiano
"""
#%%
# --- imports ---
import time
import torch
import numpy as np
from sklearn import datasets
from copy import deepcopy
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.utils import shuffle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
from copy import deepcopy
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# Limitar hilos internos por proceso
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def run_single_trial(trial_id, train_dl, NUM_FEATURES, NUM_PARAMS, LAYERS, LR, EPOCHS):       
    t0 = time.perf_counter()
    init_weights = [torch.rand(NUM_PARAMS) * 2 - 1 for _ in range(LAYERS)]
    model = QNN_0_Sum(LAYERS, NUM_FEATURES, NUM_PARAMS, init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for p in model.parameters():
        p.data.clamp_(-2 * np.pi, 2 * np.pi)
    for _ in range(EPOCHS):
        model.train()
        for data, target in train_dl:
            opt.zero_grad()
            out = model(data)
            loss = cost_fidelity(out, target)
            loss.backward()
            opt.step()
    final_loss = eval_loss_full(train_dl, model)
    train_acc, *_ = evaluate_accuracy(train_dl, model)
    return {
        "trial": trial_id,
        "loss": final_loss,
        "train_acc": train_acc,
        "params_opt": deepcopy(model.state_dict()),
        "init_weights": [w.detach().clone() for w in init_weights],
        "train_time": time.perf_counter() - t0,
    }

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
# class QNN_0_Prod(nn.Module):
    
#     def __init__(self, num_layers, num_features, num_params, init_weights=None):
#         super(QNN_0_Prod, self).__init__()
#         self.num_layers = num_layers
#         self.num_features = num_features
#         self.num_params = num_params

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
            
#         Sz12 = (gm(q1, q1) - gm(q2, q2)).to(torch.complex64)
#         Sz02 = (gm(q0, q0) - gm(q2, q2)).to(torch.complex64)
        
#         # Configurations
        
#         # C32
#         generatorsYYY = [gm2,    # RY_01
#                           gm5,    # RY_02
#                           gm7]    # RY_12
        
#         generatorsZZZ = [gm3,    # RZ_01
#                           Sz02,   # RZ_02
#                           Sz12]   # RZ_12
        
#         generatorsYYYZ = [gm2,   # RY_01
#                           gm5,   # RY_02
#                           gm7,   # RY_12
#                           gm3]   # RZ_01
        
#         generatorYZYZ = [gm2,    # RY_01
#                           gm3,    # RZ_01
#                           gm7,    # RY_12
#                           Sz12]   # RZ_12

#         generators_generalU = [gm8,
#                                 gm3,
#                                 gm2,
#                                 gm3,
#                                 gm5,
#                                 gm3,
#                                 gm2,
#                                 gm3]

#         generators_enc = generators_generalU
#         generators_var = generators_generalU
     
#         # Subset para codificación, todos para la parte variacional
#         self.register_buffer("gens_enc", torch.stack(generators_enc[:self.num_features]))  # [F, 3, 3]
#         self.register_buffer("gens_var", torch.stack(generators_var[:self.num_params]))    # [P, 3, 3]
        
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
#             # Encoding
#             for j in range(self.num_features):
#                 G = self.gens_enc[j]
#                 x = batch_c[:, j].view(-1, 1, 1)
#                 U = torch.matrix_exp(1j * x * G)
#                 state = torch.bmm(U, state)

#             # Variational
#             for j, G in enumerate(self.gens_var):
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



def load_dataset(data_path, n_features, seed):
    """
    HAR local -> filtra 3 clases (1=WALKING, 4=SITTING, 6=LAYING),
    StandardScaler -> PCA (fit en train) -> MinMaxScaler a [-1,1] sobre los scores de PCA.
    Devuelve: X_train_final, y_train, X_test_final, y_test, explained_variance
    """
    data_path = Path(data_path)

    # --- nombres únicos de features ---
    raw_feats = pd.read_csv(
        data_path / "features.txt", sep=r"\s+", header=None, names=["idx", "feature"]
    )["feature"].tolist()

    def make_unique(names):
        seen, out = {}, []
        for n in names:
            k = seen.get(n, 0)
            out.append(n if k == 0 else f"{n}_{k}")
            seen[n] = k + 1
        return out

    features = make_unique(raw_feats)

    # --- cargar train/test ---
    X_train = pd.read_csv(data_path / "train" / "X_train.txt",
                          sep=r"\s+", header=None, names=features)
    y_train = pd.read_csv(data_path / "train" / "y_train.txt",
                          sep=r"\s+", header=None, names=["activity"])

    X_test  = pd.read_csv(data_path / "test" / "X_test.txt",
                          sep=r"\s+", header=None, names=features)
    y_test  = pd.read_csv(data_path / "test" / "y_test.txt",
                          sep=r"\s+", header=None, names=["activity"])

    # --- filtrar 3 clases ---
    keep = [1, 4, 6]  # 1=WALKING, 4=SITTING, 6=LAYING
    mtr = y_train["activity"].isin(keep)
    mte = y_test["activity"].isin(keep)
    X_train, y_train = X_train[mtr], y_train[mtr].reset_index(drop=True)
    X_test,  y_test  = X_test[mte],  y_test[mte].reset_index(drop=True)

    # re-etiquetar a 0,1,2
    mapping = {cls: i for i, cls in enumerate(keep)}
    y_train = y_train["activity"].map(mapping).values
    y_test  = y_test["activity"].map(mapping).values

    # --- 1) StandardScaler antes del PCA ---
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std  = std_scaler.transform(X_test)

    # --- 2) PCA (fit en train) ---
    pca = PCA(n_components=n_features)
    Z_train = pca.fit_transform(X_train_std)
    Z_test  = pca.transform(X_test_std)
    explained_var = pca.explained_variance_ratio_.sum()

    # --- 3) MinMaxScaler a [-1,1] sobre los scores de PCA ---
    mm_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_final = mm_scaler.fit_transform(Z_train)
    X_test_final  = mm_scaler.transform(Z_test)

    
    # --- 4) Shuffle con semilla reproducible ---
    X_train_final, y_train = shuffle(X_train_final, y_train, random_state=seed)
    X_test_final,  y_test  = shuffle(X_test_final,  y_test,  random_state=seed)


    return X_train_final, y_train, X_test_final, y_test, explained_var


#%%

'''
     =====================================
          training NUM_FEATURES fixed
     =====================================
'''

if __name__ == "__main__":
    
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    # --- Hiperparámetros ---
    N_TRIALS   = 10
    N_WORKERS  = 5
    EPOCHS     = 250
    BATCH_SIZE = 32
    NUM_FEATURES = 7
    NUM_PARAMS   = 4
    LAYERS       = 6
    LR           = 0.006
    SAVE_DIR = "/Users/ccristiano/Documents/Codigos/QML/Wine/qutrit/data/4params_sum_2"

    # --- DataLoaders
    data_path = "/Users/ccristiano/Documents/Codigos/QML/HAR/UCI HAR Dataset"

    X_train, y_train, X_test, y_test, exp_var = load_dataset(data_path, NUM_FEATURES, 42)
    
    train_dl = DataLoader(CustomDataset(X_train[:560], y_train[:560]),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(CustomDataset(X_test[:140],  y_test[:140]),
                          batch_size=BATCH_SIZE, shuffle=False)
 
    # --- Cronómetro global ---
    global_start = time.perf_counter()

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:

        futures = [
            ex.submit(run_single_trial,
                      i, train_dl,
                      NUM_FEATURES, NUM_PARAMS, LAYERS, LR, EPOCHS)
            for i in range(1, N_TRIALS + 1)
        ]
               

        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            print(f"Trial {res['trial']} listo "
                  f"(loss={res['loss']:.4f}, "
                  f"tiempo={res['train_time']:.1f}s)")

    total_time = time.perf_counter() - global_start
    print(f"\n⏱ Tiempo total (todos los {N_TRIALS} trials en paralelo): "
          f"{total_time:.2f}s")

    # # --- Save results ---
    # with open(f"{SAVE_DIR}/results_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.pkl", "wb") as f:
    #     pickle.dump(results, f)

#%%

if __name__ == "__main__":
    import multiprocessing as mp
    import signal, sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # ---- usar spawn para evitar herencia de estado ----
    mp.set_start_method("spawn", force=True)

    # --- Hiperparámetros ---
    N_TRIALS   = 10
    N_WORKERS  = 5
    EPOCHS     = 250
    BATCH_SIZE = 32
    NUM_FEATURES = 7
    NUM_PARAMS   = 4
    LAYERS       = 6
    LR           = 0.006
    SAVE_DIR = "/Users/ccristiano/Documents/Codigos/QML/Wine/qutrit/data/4params_sum_2"

    # --- Datos (solo tensores, no DataLoader) ---
    data_path = "/Users/ccristiano/Documents/Codigos/QML/HAR/UCI HAR Dataset"
    X_train, y_train, X_test, y_test, exp_var = load_dataset(data_path, NUM_FEATURES, 42)

    train_dl = DataLoader(CustomDataset(X_train[:560], y_train[:560]),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(CustomDataset(X_test[:140],  y_test[:140]),
                          batch_size=BATCH_SIZE, shuffle=False)

    global_start = time.perf_counter()
    results = []

    # --- creamos el pool afuera para poder referenciarlo en el handler ---
    executor = ProcessPoolExecutor(max_workers=N_WORKERS)

    # handler para Ctrl-C / botón Stop
    def shutdown_handler(signum, frame):
        print("\n🛑 Interrupción detectada: cancelando workers…")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        futures = [
            executor.submit(run_single_trial,
                      i, train_dl,
                      NUM_FEATURES, NUM_PARAMS, LAYERS, LR, EPOCHS)
            for i in range(1, N_TRIALS + 1)
        ]

        for f in as_completed(futures):
            res = f.result()  # si hay excepción en un worker se propaga aquí
            results.append(res)
            print(
                f"Trial {res['trial']} listo "
                f"(loss={res['loss']:.4f}, "
                f"tiempo={res['train_time']:.1f}s)"
            )

    except KeyboardInterrupt:
        # por si el signal handler no atrapó a tiempo
        print("\n🛑 KeyboardInterrupt: cerrando workers…")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    finally:
        # cierre ordenado si todo terminó bien
        executor.shutdown(wait=True)

    total_time = time.perf_counter() - global_start
    print(f"\n⏱ Tiempo total (todos los {N_TRIALS} trials en paralelo): {total_time:.2f}s")

    # Guardar resultados si querés
    # with open(f"{SAVE_DIR}/results_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.pkl", "wb") as f:
    #     pickle.dump(results, f)


#%%


'''
     =======================================
       training NUM_FEATURES = [4,5,6,7,8]
     =======================================
'''


if __name__ == "__main__":
    # --- Hiperparámetros globales ---
    N_TRIALS   = 4000
    N_WORKERS  = 7
    EPOCHS     = 200
    BATCH_SIZE = 32
    NUM_PARAMS = 4
    LAYERS     = 6
    LR         = 0.006
    SAVE_DIR = "/Users/ccristiano/Documents/Codigos/QML/HAR/qutrit/data/4params_sum"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Probar diferentes números de features
    for NUM_FEATURES in range(4, 9):    # 4,5,6,7,8
        print("\n" + "="*40)
        print(f"🔎 Training con NUM_FEATURES = {NUM_FEATURES}")
        print("="*40)

        # --- DataLoaders para este NUM_FEATURES ---
        data_path = "/Users/ccristiano/Documents/Codigos/QML/HAR/UCI HAR Dataset"

        X_train, y_train, X_test, y_test, exp_var = load_dataset(data_path, NUM_FEATURES, 42)
        
        train_dl = DataLoader(CustomDataset(X_train[:560], y_train[:560]),
                              batch_size=BATCH_SIZE, shuffle=True)
        test_dl  = DataLoader(CustomDataset(X_test[:140],  y_test[:140]),
                              batch_size=BATCH_SIZE, shuffle=False)

        # --- Cronómetro por configuración ---
        global_start = time.perf_counter()

        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [
                ex.submit(
                    run_single_trial,
                    i, train_dl,
                    NUM_FEATURES, NUM_PARAMS, LAYERS, LR, EPOCHS
                )
                for i in range(1, N_TRIALS + 1)
            ]
            for f in as_completed(futures):
                res = f.result()
                results.append(res)
                print(f"Trial {res['trial']} listo "
                      f"(loss={res['loss']:.4f}, "
                      f"tiempo={res['train_time']:.1f}s)")

        total_time = time.perf_counter() - global_start
        print(f"⏱ Tiempo total para NUM_FEATURES={NUM_FEATURES}: "
              f"{total_time/60:.2f} min")

        #--- Guardar resultados de esta configuración ---
        out_file = (
            f"{SAVE_DIR}/results_NUM_FEATURES_{NUM_FEATURES}_"
            f"NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}.pkl"
        )
        with open(out_file, "wb") as f:
            pickle.dump(results, f)


#%%

# import torch
# import matplotlib.pyplot as plt
# import time

# # Hiperparámetros para un solo trial
# EPOCHS       = 200
# BATCH_SIZE   = 32
# LAYERS       = 6
# LR           = 0.006
# NUM_FEATURES = 4
# NUM_PARAMS   = 4

# data_path = "/Users/ccristiano/Documents/Codigos/QML/HAR/UCI HAR Dataset"

# X_train, y_train, X_test, y_test, exp_var = load_dataset(data_path, NUM_FEATURES, 42)


# train_dl = DataLoader(CustomDataset(X_train[:560], y_train[:560]), batch_size=BATCH_SIZE, shuffle=True)

# # Inicialización de pesos y modelo
# init_weights = [torch.rand(NUM_PARAMS) * 2 - 1 for _ in range(LAYERS)]
# model = QNN_0_Sum(LAYERS, NUM_FEATURES, NUM_PARAMS, init_weights)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # Lista para guardar la pérdida por epoch
# epoch_losses = []

# t0 = time.time()
# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0
#     total_samples = 0
    
#     for data, target in train_dl:
#         optimizer.zero_grad()
#         out = model(data)
#         loss = cost_fidelity(out, target)
#         loss.backward()
#         optimizer.step()

#         # Acumulamos la pérdida para promediar por batch
#         batch_size = data.size(0)
#         running_loss += float(loss) * batch_size
#         total_samples += batch_size

#     # Promedio de la pérdida en todo el epoch
#     train_acc, *_ = evaluate_accuracy(train_dl, model)
#     avg_loss = running_loss / total_samples
#     epoch_losses.append(avg_loss)
#     print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f} - train acc:{train_acc:.1f}")

# print(f"Entrenamiento terminado en {time.time() - t0:.2f} s")

# # === Graficar ===
# plt.figure(figsize=(7,4))
# plt.plot(range(1, EPOCHS+1), epoch_losses, marker='o', linewidth=1)
# plt.title("Training Loss por Epoch (1 trial)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss (fidelity cost)")
# plt.grid(True)
# plt.show()


#%%

'''
     =======================================
           Incremental training
     =======================================
'''
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
LR           = 0.0001
NUM_FEATURES = 8
BASE_SAVE_DIR = f"/Users/ccristiano/Documents/Codigos/QML/HAR/qutrit/data/4params_sum"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
 
# dataset (lo mismo que antes)
data_path = "/Users/ccristiano/Documents/Codigos/QML/HAR/UCI HAR Dataset"

X_train, y_train, X_test, y_test, exp_var = load_dataset(data_path, NUM_FEATURES, 42)

train_dl = DataLoader(CustomDataset(X_train[:560], y_train[:560]),
                      batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(CustomDataset(X_test[:140],  y_test[:140]),
                      batch_size=BATCH_SIZE, shuffle=False)
 
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)

    warmup_epochs = int(0.1 * EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                          total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=EPOCHS - warmup_epochs,
                                                   eta_min=5e-6)  # ~ lr/100
    ],
    milestones=[warmup_epochs]
    )

    # 3) check equivalencia inicial
    loss_prev_train = eval_loss_full(train_dl,
        QNN_0_Sum(LAYERS, NUM_FEATURES, prev_num_params,
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
            
        scheduler.step()
 
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