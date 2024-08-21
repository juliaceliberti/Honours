import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

### DATA PREPARATIONc
gene_matrix_array1 = np.load("../../modelling/init_model/gene_matrix_list.npy")[:10]
gene_matrix_array = np.load("../../../modelling/init_model/gene_matrix_list.npy")[:10]
rna_expression_df = pd.read_csv(
    "../../../modelling/init_model/rna_expression_list.csv"
).iloc[:100]
