# General Notes

## Venv
- Utilising a venv to track package usage
- installing pybedtools, pyBigWig, matplotlib, and pandas to get started (couldn't instal - may need substitue packages)

## Bash

**Unzipping gzip files**
- `gunzip -k hg38.refGene.gtf.gz`

**Simplifying the WGBS data to keep only the first 9 columns (for IGV visualisation)**
 - `zcat ENCFF690FNR.bed.gz | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9}' | gzip > simplified_ENCFF690FNR.bed.gz`

**Filtering WGBS to keep standard chromosomes**


**Annotating WGBS dataset with genes (& install BedTools)**
- Example use [here](https://bedtools.readthedocs.io/en/latest/content/example-usage.html).
- `sudo apt install bedtools`
- `bedtools intersect -a ENCFF690FNR.bed.gz -b gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf.gz -wa -wb | gzip > wgbs_with_gene_annotations.bed.gz`
- `intersect` finds the instances where regions in Data A (WGBS) overlap with regions in Data B (Annotation file)
    - can be full or partial overlap
- output: new dataset combining info from both datasets. 
    - each line is a region from WGBS data with addional info on overlapping regions from gene anotation file


Updated method:
- filtering gtf file to only contain necessary information: `awk '$3 == "gene" {print $1, $4-1, $5, $10}' HepG2_data/RefGenomes/gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf | tr -d '";' > simplified_gencode.v29.genes.bed`
- seocnd version of the above to keep chr, start, end, strand, gene_id and feature type: 
- gets chromosome, start, end and geneid
- intersect files with new command: `bedtools intersect -a HepG2_data/HepG2_DNAm/ENCFF690FNR.bed.gz -b simplified_gencode.v29.genes.bed -wa -wb | awk '{print $0, $NF}' | gzip > HepG2_data/HepG2_DNAm/annotated/wgbs_with_gene_annotations_2.bed.gz`
`bedtools intersect -a HepG2_data/HepG2_DNAm/processed/ENCFF690FNR.bed -b HepG2_data/RefGenomes/simplified_gencode_genes.bed -wa -wb > HepG2_data/HepG2_DNAm/annotated/wgbs_with_annotations_simplified.bed`

**Annotating Histone ChIP-Seq data**
Pseudoreplicated Peaks
- `bedtools intersect -a ENCFF170GMN.bed -b gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf.gz -wa -wb | gzip > H3K9me3_pp_annotations.bed.gz`
- `bedtools intersect -a ENCFF505LOU.bed -b gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf.gz -wa -wb | gzip > H3K27me3_pp_annotations.bed.gz`
Signal p-value
- `bedtools intersect -a ENCFF125NHB.bedGraph -b gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf.gz -wa -wb > H3K9me3_pval_annotations.bedGraph`
- `bedtools intersect -a ENCFF896BFP.bedGraph -b gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf.gz -wa -wb > H3K27me3_pval_annotations.bedGraph`

**Annotating RNA-Seq data**
RNA gene quant files
- As there is no genomic location data (i.e. chromosome number, start, end), a mapping script is required
- First, utilising the GENCODE V29 annotation file, we extract all gene entries and their location
- This is then saved as a mapping.csv file which we can then use to merge location information with the tsv file
- This tsv file is then saved as a BED file
- We then use bedTools intersect to properly annotate with the V29 gene names
- See file RNA_gene_mapping.py and its output file gene_coordinates.csv
- From here, these two files (gene_corrdinates.csv and the RNA gene quant tsv file) are merged and the latter now contains the V29 gene name