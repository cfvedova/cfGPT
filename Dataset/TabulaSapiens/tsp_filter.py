import scanpy as sc
import scipy.cluster.hierarchy as shc
import pandas as pd

# Load data
adata = sc.read("/home/yxu/data/TabulaSapiens/TabulaSapiens.h5ad")  

print(adata.obs)

# Exclude cells with certain annotations
exclude_cells = ['epithelial cell', 'ocular surface cell', 'radial glial cell',
'lacrimal gland functional unit cell', 'connective tissue cell', 'corneal keratocyte', 
'ciliary body', 'bronchial smooth muscle cell', 'fast muscle cell', 'muscle cell', 
'myometrial cell', 'skeletal muscle satellite stem cell', 'slow muscle cell', 'tongue muscle cell', 
'vascular associated smooth muscle cell', 'alveolar fibroblast', 'fibroblast of breast', 
'fibroblast of cardiac tissue', 'myofibroblast cell']
adata = adata[~adata.obs['cell_ontology_class'].isin(exclude_cells),:]

#Exclude cells in Eye tissue
adata = adata[adata.obs['organ_tissue'] != 'Eye', :]

print(adata.obs)

#Import joinedLabels from Vorperian Github
cutTreeDF = "./joinedLabels_cutPerCompartment_cleanTSP_sigMat18_08012021.csv"
cellAnnotCol = "cell_ontology_class"
cutTreeDF = pd.read_csv(cutTreeDF, index_col = 0)
# convert to a dictionary, called lumpDict
lumpDict = pd.DataFrame.to_dict(cutTreeDF, orient = "dict")["joinedLabel"] # replace "joinedLabel" with the correct column name in cutTreeDF

# Rename the labels in adata
adata.obs[cellAnnotCol] = adata.obs[cellAnnotCol].map(lumpDict)

adata = adata[~adata.obs['cell_ontology_class'].isnull()]

adata.write('/home/yxu/data/TabulaSapiens/TabulaSapiensClean.h5ad')

