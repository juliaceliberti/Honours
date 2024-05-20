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

**Creating a secondary annotation file with extended gene regions (+-2KB dilated genes)**
- This requires a FASTA file to update the annotation file (this is to better handle edge cases,particularly at the edges of chromosomes or scaffolds)
- FASTA file downloaded from [GENCODE](https://www.gencodegenes.org/human/release_29.html) (selecting the Genome sequence, primary assembly (GRCh38) file)
- This file was then decompressed (`gunzip GRCh38.primary_assembly.genome.fa.gz`)
- The file was then used to create an indexed file (`samtools faidx GRCh38.primary_assembly.genome.fa`)
- The indexed file was then used to alter the start and end positions of the gene annotation file (extend each gene region by 2000 bases on each side)
- `bedtools slop -i gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf.gz -g GRCh38.primary_assembly.genome.fa.fai -b 2000 > extended_genes.gtf`

**Creating a third annotation file with 2KB before and after TSS**
- We need to adjust these manually (doesn't appear to be a built in bedTools method to handle suh an operation)
- We can use bedTools to validate the cut-offs (ensure they don't exceed past chromosome boundaries)

1. Adjust start and end in command line (after gunzip gtf annotation file):

![gtf file composition](HepG2_data/HepG2_Data_img/gtf_file.png)

    ```sh
    awk 'BEGIN {OFS="\t"} \
        {
            if ($6 == "+") {
            $3 = $3 - 2000;
            $4 = $3 + 2000;
        } else if ($6 == "-") {
            $4 = $4 + 2000;
            $3 = $4 - 2000;
        }
        if ($3 < 0) $3 = 0;
        print $0;
        }' HepG2_data/RefGenomes/gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf > HepG2_data/RefGenomes/tss_2kb_genes.gtf
    ```  

2. validate stat and end positions using bedTools
`bedtools slop -i  HepG2_data/RefGenomes/tss_2kb_genes.gtf -g HepG2_data/GenomeFASTA/GRCh38.primary_assembly.genome.fa.fai -b 0 >  HepG2_data/RefGenomes/tss_2kb_genes_final.gtf`

**Annotating WGBS dataset with genes (& install BedTools)**
- Example use [here](https://bedtools.readthedocs.io/en/latest/content/example-usage.html).
- `sudo apt install bedtools`
- `bedtools intersect -a ENCFF690FNR.bed.gz -b gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf.gz -wa -wb | gzip > wgbs_with_gene_annotations.bed.gz`
- `intersect` finds the instances where regions in Data A (WGBS) overlap with regions in Data B (Annotation file)
    - can be full or partial overlap
- output: new dataset combining info from both datasets. 
    - each line is a region from WGBS data with addional info on overlapping regions from gene anotation file

- UPDATED METHOD: 28/04/24
- ISSUE: previous method omits lines that don't have overlap with genes
- `(bedtools intersect -a wgbs.bed -b genes.gtf -wa -wb; bedtools intersect -a wgbs.bed -b genes.gtf -v -wa) | gzip > combined_wgbs.bed.gz`
- `gunzip -k hg38.refGene.gtf.gz`
- This method first gets the overlaps, then wites out the ones without overlaps and combines in a gzip file


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

UPDATED METHOD:
- `(bedtools intersect -a HepG2_data/HepG2_histone/processed/ENCFF170GMN.bed -b  HepG2_data/RefGenomes/extended_genes.gtf -wa -wb; bedtools intersect -a HepG2_data/HepG2_histone/processed/ENCFF170GMN.bed -b  HepG2_data/RefGenomes/extended_genes.gtf -v -wa) | gzip > combined_H3K9me3_pp_annotations.bed.gz`
- `(bedtools intersect -a HepG2_data/HepG2_histone/processed/ENCFF505LOU.bed -b  HepG2_data/RefGenomes/extended_genes.gtf -wa -wb; bedtools intersect -a HepG2_data/HepG2_histone/processed/ENCFF505LOU.bed -b  HepG2_data/RefGenomes/extended_genes.gtf -v -wa) | gzip > combined_H3K27me3_pp_annotations.bed.gz`

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