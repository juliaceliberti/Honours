### Script to create data file for initial model

import numpy as np
import pandas as pd

# loading relevant data

# altering file paths slightly for Kaya
wgbs_data = pd.read_csv(
    "HepG2_data/HepG2_DNAm/annotated/wgbs_init_model_data.csv",
    header=0,
    names=["loc", "gene_start", "gene_end", "gene_id"],
)
k27 = pd.read_csv(
    "HepG2_data/HepG2_histone/annotated/H3K27me3_bp_genes.bed",
    sep="\t",
    header=None,
    usecols=[1, 7, 8, 10],
    names=["loc", "gene_start", "gene_end", "gene_id"],
)
k9 = pd.read_csv(
    "HepG2_data/HepG2_histone/annotated/H3K9me3_bp_genes.bed",
    sep="\t",
    header=None,
    usecols=[1, 7, 8, 10],
    names=["loc", "gene_start", "gene_end", "gene_id"],
)
rna = pd.read_csv("HepG2_data/HepG2_DNAm/ENCFF649XOG.tsv", header=0, sep="\t")

# slight alteration to rna file (select only rows with 'ENS' gene ids and get relevant columns)
rna = rna.loc[rna["gene_id"].str.contains("ENS")]
rna = rna[["gene_id", "expected_count"]]

# set the index to 'gene_id' for faster access
rna.set_index("gene_id", inplace=True)

# all genes from RNA data
all_genes = rna.index


# create an empty matrix for each gene
def create_gene_matrix():
    return np.zeros(
        (4000, 3)
    )  # 4000 rows for locations (+-2KB around TSS), 3 columns for DNAm, H3K9me3, H3K27me3 mods


gene_matrices = {gene_id: create_gene_matrix() for gene_id in all_genes}


# convert loc to index in the matrix
def loc_to_index(loc, gene_start):
    return loc - gene_start  # gets idx (i.e. 1591 - 1590 = idx 1)


# update gene matrices
def process_data(
    data, modification_index
):  # modification_index represents the column / modification type for the matrix
    for _, row in data.iterrows():
        gene_id = row["gene_id"]
        if (
            gene_id in gene_matrices
        ):  # prevent error if gene in input file not present in output file
            index = loc_to_index(row["loc"], row["gene_start"])
            if 0 <= index < 4000:  # check index is within bounds
                gene_matrices[gene_id][
                    index, modification_index
                ] = 1  # presence of the modification (binary)


# Process each dataset
print("Processing wgbs")
process_data(wgbs_data, 0)  # process WGBS data and update the first column
print("Finished wgbs")
print("\n Processing k9")
process_data(k9, 1)  #  H3K9me3 data, second column
print("Finished k9")
print("\n Processing k27")
process_data(k27, 2)  # H3K27me3 data, third column
print("Finished k27")


# list of matrices to feed model
gene_matrix_list = [gene_matrices[gene_id] for gene_id in all_genes]
gene_matrix_array = np.array(gene_matrix_list)

# organise rna data in accordance with input data
rna_expression_list = [rna.loc[gene_id, "expected_count"] for gene_id in all_genes]
rna_expression_df = pd.DataFrame(
    {"gene_id": all_genes, "expression": rna_expression_list}
)


# saving data
np.save("gene_matrix_list.npy", gene_matrix_list)
rna_expression_df.to_csv("rna_expression_list.csv", index=False)
