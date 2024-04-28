# Data Breakdown

## Filtering Process

This data was downloaded from [ENCODE](https://www.encodeproject.org/search/?type=Experiment&control_type%21=*&status=released).

Due to the filtering options available, it was not possible to filter by both DNA mehtylation data and histone modification data. Selecting a target such as H3K9me3 for the assay resulted in the deselection of other data, like WGBS and RNA-Seq.

To work around this filtering limitation, we first filtered the data to cell lines that had WGBS, RNA-Seq and Histone ChIP-Seq data. This left the following cell lines: cell lines: A549, H1, K562, IMR-90, H9, GM12878, HepG2, GM23248, and OCI-LY7. See the filters selected for this [here](https://www.encodeproject.org/matrix/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&status=released&assay_title=WGBS&biosample_ontology.classification=cell+line&biosample_ontology.term_name=A549&biosample_ontology.term_name=K562&biosample_ontology.term_name=H1&biosample_ontology.term_name=H9&biosample_ontology.term_name=IMR-90&biosample_ontology.term_name=HepG2&biosample_ontology.term_name=GM12878&biosample_ontology.term_name=GM23248&biosample_ontology.term_name=OCI-LY7&assay_title=Histone+ChIP-seq&assay_title=total+RNA-seq). 
Then, these cell lines were utilised to filter a second batch of data. By selecting these cell lines and filtering to Histone ChIP-Seq and assay targets of H3K9me3 and H3K27me3, we can gather histone modification data for cell lines also represented in DNA methylation data. See this second filtering process [here](https://www.encodeproject.org/matrix/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=Histone+ChIP-seq&target.label=H3K9me3&target.label=H3K27me3&biosample_ontology.classification=cell+line&biosample_ontology.term_name=A549&biosample_ontology.term_name=K562&biosample_ontology.term_name=H1&biosample_ontology.term_name=H9&biosample_ontology.term_name=IMR-90&biosample_ontology.term_name=HepG2&biosample_ontology.term_name=GM12878&biosample_ontology.term_name=GM23248&biosample_ontology.term_name=OCI-LY7).


## Description of items downloaded

1. **.bed.gz (.bigBed)**: This is a compressed BED file used to store genomic regions of interest, presented in a tab-delimited format with columns for the chromosome, start position, and end position, among possible others. After decompressing (using gunzip for example), the BED file can be read.
    - This will be utilised to examine ChIP-seq peaks and transcript start and end sites.

2. **.bigWig**: The bigWig format is used for displaying dense, continuous data that can be zoomed in and out (such as genome browser tracks). It is particularly useful for representing measurements like gene expression levels.
    - This will be utilised to examine signal intensity.

3. **.bigBed**: Similar to bigWig, the bigBed format is used for displaying genomic intervals (like BED files) but for large datasets.
    - Same utilisation as BED files but are also indexed (and so are more efficient for querying).

For DNAm and histone ChIP-Seq, we have data types: .bigWig, .bigBed, .bed. For RNA-Seq, we have bigwig and .tsv. For the RNA .tsv files, these contain normalised data, and as such XXX.

The .bed, .bigWig, and .bigBed files contain signal data, and can be binned into intervals. We can then quantify the methylation or enrichment of the histone modifications. We can align this with RNA-Seq (.bigwig) files, binning by the same intervals. We can then quantify the level of gene expression and assign this a binary value based on some threshold for expression.

Each row within the metadata file represents a file. For WGBS data, multiple rows may be associated with the same experiement and represent different forms of output (for example, analysing the forward strand, the backward strand, and both). The same applies for the RNA data. Histone data output are separated by "signal p-value" and "pseudoreplicated peaks". "Signal p-value" provides a statistical measure of the significance of the observed signal, while "pseudoreplicated peaks" represent regions of the genome identified as potential binding sites based on the ChIP-seq data analysis. 

Further descriptions of the data, including column names, can be accessed here:
- [WGBS](https://www.encodeproject.org/data-standards/wgbs-encode4/)
![alt text](image.png)

- [RNA]()
- [Histone ChIP-Seq](https://www.encodeproject.org/chip-seq/histone/)

All HepG2 data utilised in the data_viualisation file is assembled using GRCh38 verison of the human genome. 

### More information on WGBS data
### More information on RNA data

The RNA data available consists of three files: normalised expression data for the plus and minus strand and gene quantification data in a tsv file. As this file type is not compatible with IVG and contains the data to visualise the expression levels (but is missing gene annotations), additional work was done to this file. 

According to the metadata file downloaded from encode, file ENCFF649XOG is derived from V29 gene annotations. V29 gene annotation files can be downloaded from https://www.gencodegenes.org/human/release_29.html. 

Downloading the gencode.v29.annotation.gtf.gz and decompressing it with `gunzip -k gencode.v29.annotation.gtf.gz` allows us to create a new Bed file. 

^^ Didn't work. Also trying files found at:
- https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/
- https://github.com/ENCODE-DCC/rna-seq-pipeline/releases/tag/1.2.4 (docs > reference)
    - https://www.encodeproject.org/files/ENCFF598IDH/
    - https://www.encodeproject.org/files/ENCFF285DRD/
    - https://www.encodeproject.org/files/ENCFF471EAM/
    - https://www.encodeproject.org/files/GRCh38_EBV.chrom.sizes/
- https://www.encodeproject.org/references/ENCSR151GDH/

### More information on Histone ChIP-Seq data

Pseudoreplicates definition:
- derived from the same biological sample but processed separately
- a single sample might be split after the initial preparation steps, and then each split ("pseudoreplicate") is processed and sequenced separately
- creates two datasets that are not independent at the biological level but are independent at the technical level (library preparation, sequencing, and data processing independently).

Pseudoreplicates use case: 
- a strategy to assess technical variability or to boost confidence in the data when true biological replicates are not available
- look for consistent findings across these technical replicates to ensure that the results are not due to technical artifacts
- stable peaks would be those that are identified in both pseudoreplicates

Partition concordance definition:
- comparing the peaks found in each pseudoreplicate to identify "stable" peaks (found in both pseudoreplicates)
- likely to be real signals rather than noise or artifacts
- overlap of at least 50% with peaks from both pseudoreplicates = threshold to determine this stability and reliability


## Processing the data for visualisation

As the data downloaded consisted of files in the formats of .bed.gz, .bigbed, and .bigwig, visualising the data requires preprocessing. For the initial stages of this project, the HepG2 cell line data was isolated and utilised. The script data_processing.py enters the HepG2_data folder and its subsequent directories (HepG2_DNAm and HepG2_histone) and processes the files with suffixes of .bigbed, .bed.gz, and .bigwig. It converted the former two into .bed files and the latter into .bedgraph files. 

Now that the data is in a usable format, we can begin to visualise it. 

## Visualising the data

To visualise the .bigWig, .bed and .bigBed files, we can utilise a genome browser like IGV to view the signal. We can then utlise our Jupyter Notebook to break this data down into intervals and quantify signals to work with tabular data.