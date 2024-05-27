#!/bin/bash

### UNSURE IF THIS IS REQUIRED?
# Define file paths
GTF_FILE="./HepG2_data/RefGenomes/gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf"
TSS_FILE="./HepG2_data/RefGenomes/tss_2kb.gtf"
CLIPPED_TSS_FILE="./HepG2_data/RefGenomes/clipped_tss_2kb.gtf"
CHROM_LENGTHS_FILE="./HepG2_data/RefGenomes/chrom_lengths.txt"
OVERLAPS_FILE="./HepG2_data/RefGenomes/overlaps.txt"

# Clip intervals to chromosome lengths
bedtools slop -i $TSS_FILE -g $CHROM_LENGTHS_FILE -b 0 > $CLIPPED_TSS_FILE

# Check for overlaps
bedtools intersect -a $CLIPPED_TSS_FILE -b $CLIPPED_TSS_FILE -wo > $OVERLAPS_FILE

echo "Validation complete. Check $OVERLAPS_FILE for overlaps."
