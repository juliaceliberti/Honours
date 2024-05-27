import pandas as pd

# load .bed psuedoreplicated peaks file (using ENCFF170GMN.bed (K9) and ENCFF841QVP (K27))
peak_file = "./HepG2_data/HepG2_histone/processed/ENCFF170GMN.bed"
output_file = "./HepG2_data/HepG2_histone/processed/H3K9me3_bp.bed"


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

        # for each row, create new rows so that every bp between start and end has its own row
        for pos in range(start, end):
            peak_start = pos
            peak_end = pos + 1  # start and end are 1 bp apart
            expanded_peaks.append([chrom, peak_start, peak_end, name, score, strand])

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
    ],
)

# Save the expanded DataFrame to a new .bed file
expanded_df.to_csv(output_file, sep="\t", header=False, index=False)
