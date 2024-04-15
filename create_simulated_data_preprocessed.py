import anndata as ad
import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
from scipy import sparse

# Hyperparameters
NUM_SAMPLES = 50000
NUM_CELLS_TO_EXTRACT = 1000
HVG_SELECTION = True
print("\nNumber of bulk RNAseq samples to simulate =", NUM_SAMPLES)
print("Number of cells to extract for each simulated bulk RNAseq sample =", NUM_CELLS_TO_EXTRACT)

print("Loading scRNA-seq data...")
adata=ad.read_h5ad("./Dataset/TabulaSapiens.h5ad")

print("Loaded scRNAseq data")
print(f"Original adata celltype labels: {adata.obs['cell_ontology_class'].unique()}")
print(f"Length of adata celltype labels: {len(adata.obs['cell_ontology_class'].unique())}")

print(adata)
if HVG_SELECTION:
    sc.pp.highly_variable_genes(
                adata,
                layer="raw_counts",
                n_top_genes=3000,
                batch_key=None,
                flavor="seurat_v3",
                subset=True,
            )
    print(adata)
    scrnaseq_data = adata.copy()
    scrnaseq_data.X = adata.layers['raw_counts']
else:
    # Load your actual bulk RNAseq data
    # Make sure to adjust the path to the file containing your data
    cfrna_data = pd.read_csv('./Dataset/arp3_protein_coding_feature_counts.txt',
                            sep='\t', header=None, names=['gene_names', 'counts'])
    cfrna_data['counts'] = cfrna_data['counts'].astype(float)
    cpm = cfrna_data['counts'] / cfrna_data['counts'].sum() * 1e6
    log_cpm = np.log1p(cpm)
    cfrna_df = pd.DataFrame({'GeneName': cfrna_data['gene_names'], 'Expression': log_cpm})
    cfrna_df = cfrna_df.set_index('GeneName')
    print("Contents of cfrna_data:\n", cfrna_df)
    cfrna_df = cfrna_df.transpose()
    cfrna_df = cfrna_df.sort_index(axis=1)
    print("Contents of cfrna_df:\n", cfrna_df)
    common_genes = set(adata.var_names) & set(cfrna_df.columns)
    common_gene_indices = [idx for idx, gene in enumerate(adata.var_names) if gene in common_genes]
    scrnaseq_data = adata[:, common_gene_indices].copy()
    scrnaseq_data.X = adata[:, common_gene_indices].layers['raw_counts']

# FILTER SC_RNASEQ DATA
print("\nFiltering scRNAseq data")

scrnaseq_data.raw = scrnaseq_data
min_genes = 500
max_mito = 0.05
mito_genes = scrnaseq_data.var_names.str.startswith('mt-')
scrnaseq_data.obs['percent_mito'] = np.sum(scrnaseq_data[:, mito_genes].X, axis=1) / np.sum(scrnaseq_data.X, axis=1)
scrnaseq_data.obs['n_genes'] = np.sum(scrnaseq_data.X > 0, axis=1)
scrnaseq_data = scrnaseq_data[scrnaseq_data.obs['n_genes'] > min_genes, :]
scrnaseq_data = scrnaseq_data[scrnaseq_data.obs['percent_mito'] < max_mito, :]

print("Filtered scRNAseq data")
print("After filtering:", scrnaseq_data.shape)

# PREPARE INTERSECTED PREPROCESSED SCRNASEQ DATA FOR SIMULATION --> USES RAW COUNTS, NOT PROCESSED EXPRESSION DATA
print("Preparing input data...")

# Extract cluster labels
cell_type_labels = scrnaseq_data.obs['cell_ontology_class']
print("Cell type labels extracted.")
print("\nUnique cell_type_labels labels:\n", cell_type_labels.unique(), sep='\n')
print("Number of them:", len(cell_type_labels.unique()))

# Convert the sparse matrix to a dense array
dense_X = scrnaseq_data.X.toarray() if sparse.issparse(scrnaseq_data.X) else scrnaseq_data.X
# Convert adata.X to DataFrame
expression_data = pd.DataFrame(dense_X, columns=scrnaseq_data.var_names)
print("Converted scrnaseq_data.X to DataFrame.")
print("\nContents of expression_data before adding cluster labels:\n", expression_data.head())

# Convert cell type labels to a regular series with the same index as expression_data
cell_type_labels = pd.Series(cell_type_labels.values, index=expression_data.index, name='cell_type')
print("--> Converted cluster labels to a regular series.")
print("\n--> Added cell labels to scRNAseq dataframe")

with pd.option_context('display.max_rows', None):
    print("\nAll unique cell ontology class counts in the scRNAseq data:\n\n", pd.Series(cell_type_labels.value_counts()))

# Insert cell_type_labels as the first column in expression_data
expression_data.insert(0, 'cell', cell_type_labels)

print("\nscRNAseq dataframe with cell type labels:\n\n", expression_data.head())

# List of all cell types (assuming 'cell' column contains the cell types)
all_cell_types = expression_data['cell'].unique()

# Dataframes to store the results
bulk_rnaseq_mean_expression_df = pd.DataFrame()
cell_type_proportions_df = pd.DataFrame()
# SIMULATE RANDOMLY SAMPLED BULK RNASEQ SAMPLES WITH NON-FIXED CELL PROPORTIONS AND SUM OF RAW COUNTS --> correct one - FASTER

bulk_rnaseq_mean_expression_df = pd.DataFrame()
cell_type_proportions_df = pd.DataFrame()

bulk_rnaseq_mean_expression_list = []
cell_type_proportions_list = []

# Precompute cells of each type
cells_by_type = {cell_type: expression_data[expression_data['cell'] == cell_type] for cell_type in all_cell_types}
all_cell_types_set = set(all_cell_types)
print(all_cell_types)
print(f"Num celltypes {len(all_cell_types)}")

# Loop to create the simulated samples
for sample_num in tqdm.tqdm(range(1, NUM_SAMPLES + 1)):
    selected_cells_list = []
    random_proportions = np.random.dirichlet(np.ones(len(all_cell_types)), size=1)[0]
    cell_type_proportions = pd.Series(index=all_cell_types, dtype=float).fillna(0)
    total_cells = 0

    for cell_type, proportion in zip(all_cell_types, random_proportions):
        num_cells_this_type = int(proportion * NUM_CELLS_TO_EXTRACT)
        cells_of_this_type = cells_by_type[cell_type]
        num_cells_this_type = min(num_cells_this_type, cells_of_this_type.shape[0])
        sampled_cells = cells_of_this_type.sample(num_cells_this_type)
        selected_cells_list.append(sampled_cells)
        cell_type_proportions[cell_type] = num_cells_this_type
        total_cells += num_cells_this_type

    # Concatenate all the sampled cells for this iteration into a DataFrame
    selected_cells = pd.concat(selected_cells_list)

    # Calculate sum of expression levels for each gene
    bulk_rnaseq_mean_expression_list.append(selected_cells.drop(columns=['cell']).sum())

    # Normalize the cell counts to get proportions
    cell_type_proportions /= cell_type_proportions.sum()

    cell_type_proportions_list.append(cell_type_proportions)

# Create DataFrame outside the loop
bulk_rnaseq_mean_expression_df = pd.DataFrame(bulk_rnaseq_mean_expression_list)
cell_type_proportions_df = pd.DataFrame(cell_type_proportions_list)

# Setting the index names to Sample1, Sample2, ...
bulk_rnaseq_mean_expression_df.index = [f'Sample{i}' for i in range(1, len(bulk_rnaseq_mean_expression_df) + 1)]
cell_type_proportions_df.index = [f'Sample{i}' for i in range(1, len(cell_type_proportions_df) + 1)]

# Printing the bulk RNAseq sample df
print("\nSimulated bulk RNAseq dataframe with variable cell proportions.\n", bulk_rnaseq_mean_expression_df.head())

# Printing the cell type proportions
print("Cell type proportions of simulated RNAseq dataframe", cell_type_proportions_df.head())
print(f"Num celltypes result: {len(cell_type_proportions_df.columns)}")

bulk_rnaseq_mean_expression_df.to_csv("./Dataset/bulk_data.csv")
cell_type_proportions_df.to_csv("./Dataset/label_data.csv")
