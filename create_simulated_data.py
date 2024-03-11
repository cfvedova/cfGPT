import anndata as ad
import numpy as np
import os
import pandas as pd
import random
import scipy
from scipy.sparse import csr_matrix
import sys
import tqdm

# LOAD DATA
num_samples = 50000
print("\nNumber of bulk RNAseq samples to simulate =", num_samples)

num_cells_to_extract = 300
print("Number of cells to extract for each simulated bulk RNAseq sample =", num_cells_to_extract)
   
h5adfile='./Dataset/TabulaSapiens.h5ad'
print('processing:'+ h5adfile)

bname=os.path.splitext(os.path.basename(h5adfile))[0]

adata=ad.read_h5ad(h5adfile)
print(adata)

celltype_labels = pd.DataFrame(adata.obs['cell_ontology_class'])
print(celltype_labels)
celltype_labels.columns = ['Celltype']
all_cell_types = pd.unique(celltype_labels['Celltype'])

gene_symbols = adata.var['gene_symbol']
bulk_rnaseq_mean_expression_df = pd.DataFrame(columns=gene_symbols)
cell_type_proportions_list = []
cell_type_proportions_df = pd.DataFrame(columns=all_cell_types)

# Precompute cells of each type
cells_by_type = {cell_type: np.where(celltype_labels == cell_type)[0] for cell_type in all_cell_types}

# Loop to create the simulated samples
for sample_num in tqdm.tqdm(range(num_samples)):
    selected_cells_list = []
    random_proportions = np.random.dirichlet(np.ones(len(all_cell_types)))
    cell_type_proportions = pd.Series(index=all_cell_types, dtype=float).fillna(0)
    total_cells = 0

    for cell_type, proportion in zip(all_cell_types, random_proportions):
        num_cells_this_type = int(proportion * num_cells_to_extract)
        cells_of_this_type = cells_by_type[cell_type]
        num_cells_this_type = min(num_cells_this_type, len(cells_of_this_type))
        sampled_cells_indexes = np.random.choice(cells_of_this_type, num_cells_this_type)
        samples = adata.layers['raw_counts'][sampled_cells_indexes].todense().astype(int)
        selected_cells_list.append(samples)
        cell_type_proportions[cell_type] = num_cells_this_type
        total_cells += num_cells_this_type

    # Concatenate all the sampled cells for this iteration into a DataFrame
    selected_cells = np.concatenate(selected_cells_list)
    selected_cells = selected_cells.sum(axis=0)

    # Calculate sum of expression levels for each gene
    selected_cells = pd.DataFrame(selected_cells.tolist(), columns=gene_symbols, index=[f"Sample_{sample_num}"])
    bulk_rnaseq_mean_expression_df = pd.concat([bulk_rnaseq_mean_expression_df, selected_cells])

    # Normalize the cell counts to get proportions
    cell_type_proportions /= cell_type_proportions.sum()
    cell_type_proportions = pd.DataFrame(cell_type_proportions).T
    cell_type_proportions.index = [f"Sample_{sample_num}"]
    cell_type_proportions_df = pd.concat([cell_type_proportions_df, cell_type_proportions])

print("\nSimulated bulk RNAseq dataframe with variable cell proportions.\n", bulk_rnaseq_mean_expression_df.head())
print("Cell type proportions of simulated RNAseq dataframe", cell_type_proportions_df)

bulk_rnaseq_mean_expression_df.to_csv("./Dataset/simulated_big.csv")
cell_type_proportions_df.to_csv("./Dataset/cell_type_big.csv")
#Convert to sparse matrix
"""print(bulk_rnaseq_mean_expression_df.values)
print(bulk_rnaseq_mean_expression_df.astype(pd.SparseDtype("int32",0)).sparse)
print(bulk_rnaseq_mean_expression_df.astype(pd.SparseDtype("int32",0)).values)
bulk_sparse = scipy.sparse.csr_matrix(bulk_rnaseq_mean_expression_df.values)
print(bulk_sparse)
#bulk_temp = bulk_rnaseq_mean_expression_df.astype(pd.SparseDtype("int32",0)).sparse
#print(bulk_temp)
#bulk_rnaseq_counts_sparse = csr_matrix(bulk_temp.to_coo())
adata_out = ad.AnnData(bulk_sparse)
adata_out.var["gene_symbol"] = gene_symbols
adata_out.obsm["cell_proportions"] = cell_type_proportions_df
print(adata_out)
print(adata_out.X)
print(len(adata_out.X))
#print("\nIndividual element")
#print(adata_out[["Sample_1"]][["OR4G4P"]])
print("\nGene symbols info")
print(gene_symbols)
print(type(gene_symbols))
print(gene_symbols[0])
print(type(gene_symbols[0]))
#print("\nobsm")
#print(adata_out.obsm)
print("\nobsm actual proportions")
print(adata_out.obsm["cell_proportions"])
print("\nTotal thing")
print(adata_out)
#print("\nVariable")
#print(adata_out.var)
#print("\nObs should be empty")
#print(adata_out.obs)

print("\nadata X")
print(adata.X)
print(type(adata.X))
#print(adata.X.shape())
print(adata.obs['cell_ontology_class'])

print("\nspecific")
#print(adata_out[["Sample_0"], ["OR4G4P"]])
adata_out.write_h5ad("./Dataset/TabulaSapiensSimulatedBulk.h5ad")"""
