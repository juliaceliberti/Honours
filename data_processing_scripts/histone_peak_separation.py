import pandas as pd
import re

# load .bed psuedoreplicated peaks file (using ENCFF170GMN.bed (H3K9me3_pp_annot_2kb.bed) and ENCFF841QVP (H3K27me3_pp_annot_2kb.bed))
peak_file = "./HepG2_data/HepG2_histone/annotated/H3K9me3_pp_annot_2kb.bed"
output_file = "./HepG2_data/HepG2_histone/annotated/H3K9me3_bp.bed"


# batch size for processing
chunk_size = 1000

# Create a new DataFrame to store the expanded peaks
expanded_peaks = []


for chunk_index, chunk in enumerate(
    pd.read_csv(
        peak_file,
        sep="\t",
        header=None,
        names=[
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
            "peak",
            "chrom_g",
            "source_g",
            "feature_type_g",
            "start_g",
            "end_g",
            "score_g",
            "strand_g",
            "frame_g",
            "attributes_g",
        ],
        chunksize=chunk_size,
    )
):

    # loop through each row in file
    for index, row in chunk.iterrows():
        chrom = row["chrom"]
        start = row["chromStart"]
        end = row["chromEnd"]
        name = row["name"]
        score = row["score"]
        strand = row["strand"]
        feature = row["feature_type_g"]
        gene_strand = row["strand_g"]
        attributes = row["attributes_g"]

        # Extract gene ID from attributes
        match = re.search(r'gene_id\s*"([^"]+)"', attributes)
        gene_id = match.group(1) if match else "Unknown"

        # for each row, create new rows so that every bp between start and end has its own row
        for pos in range(start, end):
            peak_start = pos
            peak_end = pos + 1  # start and end are 1 bp apart
            expanded_peaks.append(
                [
                    chrom,
                    peak_start,
                    peak_end,
                    name,
                    score,
                    strand,
                    feature,
                    gene_strand,
                    gene_id,
                ]
            )

    # Track progress
    print(f"Processed chunk {chunk_index + 1}")

# Create a new DataFrame from the expanded peaks
expanded_df = pd.DataFrame(
    expanded_peaks,
    columns=[
        "chrom",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "feature_type",
        "gene_strand",
        "gene_id",
    ],
)

# Save the expanded DataFrame to a new .bed file
expanded_df.to_csv(output_file, sep="\t", header=False, index=False)
