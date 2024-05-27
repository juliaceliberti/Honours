# importing required packages
import pandas as pd

# defining file paths
gtf_file_path = (
    "./HepG2_data/RefGenomes/gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf"
)
tss_file_path = "./HepG2_data/RefGenomes/tss_2kb.gtf"

# Define chunk size
chunk_size = 1000000


# function to alter boundaries
def tss_2kb(row):

    # if strand is positive, we take +-2KB around the TSS, which is at the 'start' site
    if row["strand"] == "+":
        new_start = row["start"] - 2000
        new_end = row["start"] + 2000

    # if strand is negative, we take +-2KB around the TSS, which is at the 'end' site
    elif row["strand"] == "-":
        new_start = row["end"] - 2000
        new_end = row["end"] + 2000

    # Ensure the new coordinates are within valid ranges
    new_start = max(new_start, 1)  # Ensure start is at least 1 (1-based)

    return pd.Series(
        [
            row["chrom"],
            row["source"],
            row["feature_type"],
            new_start,
            new_end,
            row["score"],
            row["strand"],
            row["frame"],
            row["attributes"],
        ]
    )


chunk_counter = 0

# Open the output file for writing
with open(tss_file_path, "w") as output_file:
    # Process the input file in chunks
    # loading in gtf file
    for chunk in pd.read_csv(
        gtf_file_path,
        header=None,
        delimiter="\t",
        skiprows=5,
        names=[
            "chrom",
            "source",
            "feature_type",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attributes",
        ],
        chunksize=chunk_size,
    ):

        # alter the gtf file by applying our tss function
        tss_chunk = chunk.apply(tss_2kb, axis=1)

        # save file
        tss_chunk.to_csv(tss_file_path, sep="\t", header=False, index=False, mode="a")

        # Increment the chunk counter and print progress
        chunk_counter += 1
        print(f"Processed chunk {chunk_counter}")

print(f"Processing complete. Output saved to {tss_file_path}.")
