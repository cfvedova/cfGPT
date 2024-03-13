#Take in h5ad file and convert it into format suitable for scaden 

import anndata
import pandas as pd
import numpy as np

import sys
import os
   
h5adfile='./Dataset/TabulaSapiens.h5ad'
print('processing:'+ h5adfile)

bname=os.path.splitext(os.path.basename(h5adfile))[0]

adata=anndata.read_h5ad(h5adfile)
print(adata)

y = pd.DataFrame(adata.obs['cell_ontology_class'])[:100000]
print(y.shape)

#Prints out txt with all celltypes 
y.columns=['Celltype']
celltypes_file='_celltypes.txt'
print('out:'+celltypes_file)
y.to_csv(celltypes_file, sep='\t', index=False)

#Create x as integer counts
gene_names = adata.var_names
adata.layers['raw_counts'] = adata.layers['raw_counts'].astype(int)
x = pd.DataFrame(adata.layers['raw_counts'].A)[:100000]
print(x.shape)

x.columns=gene_names
counts_file='_counts.txt'
print('out:'+counts_file)
x.to_csv(counts_file, sep='\t', index=False)

#preprocessing

""" import csv

def delete_second_column_in_place(input_file):
    # Read the data
    with open(input_file, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    # Modify the data (delete second column)
    for row in data:
        if len(row) > 1:
            del row[1]  # Delete the second column (0-based index)

    # Write the modified data back to the original file
    with open(input_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Example usage
input_csv = './cfrna_exon_counts_protein_coding.csv'
delete_second_column_in_place(input_csv)

def convert_to_tab_separated(input_file, output_file):
    # Read data from the input file
    with open(input_file, 'r') as file:
        data = file.read()

    # Split lines and extract headers and rows
    lines = data.split('\n')
    headers = lines[0].split(',')
    rows = [line.split(',') for line in lines[1:] if line.strip()]  # Exclude empty lines

    # Format the data into tab-separated rows/columns
    formatted_data = '\t'.join(headers) + '\n'
    formatted_data += '\n'.join(['\t'.join(row) for row in rows])

    # Save the formatted data to the output file
    with open(output_file, 'w') as file:
        file.write(formatted_data)

# Example usage
input_file = './cfrna_exon_counts_protein_coding.csv'  # Replace with your input file name
output_file = './cfrna_exon_counts_protein_coding.txt'  # Replace with your desired output file name
convert_to_tab_separated(input_file, output_file)   

#Scaden will simulate bulk data based on given celltypes + counts txt, spits out data.h5ad 

!scaden simulate --data TabulaSapiens_3_2_2/ -n 50000 --pattern _counts.txt """
