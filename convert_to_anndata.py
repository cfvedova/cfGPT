import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np

### Convert spliced .txt files into one anndata with X being count matrix [n_obs x n_genes], variables with gene names [n_genes], obs with cell proportions [n_obs x n_cells]

counts = pd.read_csv("data_sim_bulk__counts.txt", sep='\t', index_col=0)
counts = counts.T
adata = ad.AnnData(counts, dtype=np.float32)
adata.var_names = counts.columns

cell_proportions = pd.read_csv("data_sim_bulk__percent_real.txt", sep='\t', index_col=0)

adata.obsm = cell_proportions