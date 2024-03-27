# Data Breakdown

## Filtering Process

This data was downloaded from [ENCODE](https://www.encodeproject.org/search/?type=Experiment&control_type%21=*&status=released).

Due to the filtering options available, it was not possible to filter by both DNA mehtylation data and histone modification data. Selecting a target such as H3K9me3 for the assay resulted in the deselection of other data, like WGBS and RNA-Seq.

To work around this filtering limitation, we first filtered the data to cell lines that had WGBS, RNA-Seq and Histone ChIP-Seq data. This left the following cell lines: cell lines: A549, H1, K562, IMR-90, H9, GM12878, HepG2, GM23248, and OCI-LY7. See the filters selected for this [here](https://www.encodeproject.org/matrix/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&status=released&assay_title=WGBS&biosample_ontology.classification=cell+line&biosample_ontology.term_name=A549&biosample_ontology.term_name=K562&biosample_ontology.term_name=H1&biosample_ontology.term_name=H9&biosample_ontology.term_name=IMR-90&biosample_ontology.term_name=HepG2&biosample_ontology.term_name=GM12878&biosample_ontology.term_name=GM23248&biosample_ontology.term_name=OCI-LY7&assay_title=Histone+ChIP-seq&assay_title=total+RNA-seq). 
Then, these cell lines were utilised to filter a second batch of data. By selecting these cell lines and filtering to Histone ChIP-Seq and assay targets of H3K9me3 and H3K27me3, we can gather histone modification data for cell lines also represented in DNA methylation data. See this second filtering process [here](https://www.encodeproject.org/matrix/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=Histone+ChIP-seq&target.label=H3K9me3&target.label=H3K27me3&biosample_ontology.classification=cell+line&biosample_ontology.term_name=A549&biosample_ontology.term_name=K562&biosample_ontology.term_name=H1&biosample_ontology.term_name=H9&biosample_ontology.term_name=IMR-90&biosample_ontology.term_name=HepG2&biosample_ontology.term_name=GM12878&biosample_ontology.term_name=GM23248&biosample_ontology.term_name=OCI-LY7).


## Description of items downloaded

1. **.bed.gz**: This is a compressed BED file used to store genomic regions of interest, presented in a tab-delimited format with columns for the chromosome, start position, and end position, among possible others. After decompressing (using gunzip for example), the BED file can be read.
    - This will be utilised to examine ChIP-seq peaks and transcript start and end sites.

2. **.bigWig**: The bigWig format is used for displaying dense, continuous data that can be zoomed in and out (such as genome browser tracks). It is particularly useful for representing measurements like gene expression levels.
    - This will be utilised to examine signal intensity.

3. **.bigBed**: Similar to bigWig, the bigBed format is used for displaying genomic intervals (like BED files) but for large datasets.
    - Same utilisation as BED files but are also indexed (and so are more efficient for querying).


## Processing the data for visualisation

As the data downloaded consisted of files in the formats of .bed.gz, .bigbed, and .bigwig, visualising the data requires preprocessing. For the initial stages of this project, the HepG2 cell line data was isolated and utilised. The script data_processing.py enters the HepG2_data folder and its subsequent directories (HepG2_DNAm and HepG2_histone) and processes the files with suffixes of .bigbed, .bed.gz, and .bigwig. It converted the former two into .bed files and the latter into .bedgraph files. 

Now that the data is in a usable format, we can begin to visualise it. 

## Visualising the data