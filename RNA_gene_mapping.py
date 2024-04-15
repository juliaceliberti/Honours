import pandas as pd

### FILE PATH VARIABLES

annotation_path = 'HepG2_data/RefGenomes/gencode.v29.annotation.gtf'
mapped_path = 'annotation_data/gene_coordinates.csv'
RNA_path = 'HepG2_data/HepG2_DNAm/ENCFF649XOG.tsv'

mapped_output_path = 'annotation_data/gene_coordinates.csv'
merged_output_path = 'HepG2_data/HepG2_DNAm/annotated/rna_gene_quant_with_annotations.csv'


### CREATING MAPPING FILE
def create_mapping(annotation_path, output_path):

    annotation_data = pd.read_csv(
        annotation_path,
        comment='#',
        sep='\t',
        header=None,
        usecols=[0, 2, 3, 4, 8],  # Chromosome, feature type, start, end, attributes
        names=['chromosome', 'type', 'start', 'end', 'attributes']
    )

    # get gene entries (ignoring others such as exons)
    genes = annotation_data[annotation_data['type'] == 'gene']

    # get gene id
    def parse_attributes(attributes):
        attr_dict = {}
        for attr in attributes.split(';'):
            if attr.strip():
                key, value = attr.strip().split(' ', 1)
                attr_dict[key] = value.strip('"')
        return attr_dict

    # get location attributes from gene rows
    genes['parsed'] = genes['attributes'].apply(parse_attributes)

    # get Ensembl gene IDs (as this is current state of the RNA gene quant file)
    genes['gene_id'] = genes['parsed'].apply(lambda x: x.get('gene_id'))
    genes['gene_name'] = genes['parsed'].apply(lambda x: x.get('gene_name'))  # Ensembl gene name is now 'gene_name'

    # save to a CSV file
    genes[['chromosome', 'start', 'end', 'gene_id', 'gene_name']].to_csv(output_path, index=False)

    return

### MERGING MAPPING FILE AND RNA QUANT FILE
def merge_files(mapped_path, RNA_path, merged_output_path):
    # load mapping file and gene quant file
    gene_coordinates = pd.read_csv(mapped_path)
    RNA_data = pd.read_csv(RNA_path, sep='\t')

    # merge on ensembl id
    merged_data = pd.merge(RNA_data, gene_coordinates, on='gene_id', how='left')

    # Now, merged_data contains both the expression data and the corresponding genomic coordinates
    print(merged_data.head())

    # You can save this merged data to a new CSV file
    merged_data.to_csv(merged_output_path, index=False)
    # may add additional code to return a bed file if this is a useful addition
    return


create_mapping(annotation_path, mapped_output_path)
merge_files(mapped_path, RNA_path, merged_output_path)
