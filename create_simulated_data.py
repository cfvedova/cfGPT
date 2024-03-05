import anndata as ad
import numpy as np
import os
import pandas as pd
import random
import sys
import tqdm

# LOAD DATA
num_samples = 1000
print("\nNumber of bulk RNAseq samples to simulate =", num_samples)

num_cells_to_extract = 5000
print("Number of cells to extract for each simulated bulk RNAseq sample =", num_cells_to_extract)
   
h5adfile='./Dataset/TabulaSapiens.h5ad'
print('processing:'+ h5adfile)

bname=os.path.splitext(os.path.basename(h5adfile))[0]

adata=anndata.read_h5ad(h5adfile)
print(adata)

celltype_labels = pd.DataFrame(adata.obs['cell_ontology_class'])
all_cell_types = pd.unique(celltype_labels['cell_type'])

bulk_rnaseq_mean_expression_df = pd.DataFrame()
cell_type_proportions_list = []

# Precompute cells of each type
cells_by_type = {cell_type: np.where(celltype_labels == cell_type)[0] for cell_type in all_cell_types}

# Loop to create the simulated samples
for sample_num in tqdm(range(1, num_samples + 1)):
    selected_cells_list = []
    random_proportions = np.random.dirichlet(np.ones(len(all_cell_types)))
    cell_type_proportions = pd.Series(index=all_cell_types, dtype=float).fillna(0)
    total_cells = 0

    for cell_type, proportion in zip(all_cell_types, random_proportions):
        num_cells_this_type = int(proportion * num_cells_to_extract)
        cells_of_this_type = cells_by_type[cell_type]
        num_cells_this_type = min(num_cells_this_type, len(cells_of_this_type))
        sampled_cells_indexes = random.sample(cells_of_this_type, num_cells_this_type)
        selected_cells_list.append(adata.X[sampled_cells_indexes].todense())
        cell_type_proportions[cell_type] = num_cells_this_type
        total_cells += num_cells_this_type

    # Concatenate all the sampled cells for this iteration into a DataFrame
    selected_cells = pd.concat(selected_cells_list)

    # Print actual number of cells in this sample
    print(f"Actual number of cells in sample {sample_num}: {total_cells}")

    # Calculate sum of expression levels for each gene
    bulk_rnaseq_mean_expression_df.append(selected_cells.drop(columns=['cell']).sum())

    # Normalize the cell counts to get proportions
    cell_type_proportions /= cell_type_proportions.sum()

    # Print the sum of each row to ensure it is equal to 1
    print(f"Sum of proportions in sample {sample_num}: {cell_type_proportions.sum()}")

    cell_type_proportions_list.append(cell_type_proportions)

print("\nSimulated bulk RNAseq dataframe with variable cell proportions.\n", bulk_rnaseq_mean_expression_df.head())
print("Cell type proportions of simulated RNAseq dataframe", cell_type_proportions_df.head())

#Write h5ad
adata = ad.AnnData(bulk_rnaseq_mean_expression_df, dtype=np.float32)
adata.var_names = bulk_rnaseq_mean_expression_df.columns
adata.obsm = cell_type_proportions_list

adata.write_h5ad("./Dataset/TabulaSapiensSimulatedBulk.h5ad")