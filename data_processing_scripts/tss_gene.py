# importing required packages
import pandas as pd

# defining file paths
gtf_file_path = (
    "./HepG2_data/RefGenomes/gencode.v29.chr_patch_hapl_scaff.basic.annotation.gtf"
)
tss_file_path = "./HepG2_data/RefGenomes/tss_2kb.gtf"

# loading in gtf file
original_gtf = pd.read_csv(
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
)


# function to alter boundaries
def tss_2kb(row):

    # if strand is positive, we take +-2KB around the TSS, which is at the 'start' site
    if row["strand"] == "+":
        new_start = row["start"] - 2000
        new_end = row["start"] + 2000

    # if strand is negative, we take +-2KB around the TSS, which is at the 'end' site
    elif row["strand"] == "-":
        new_start = row["start"] - 2000
        new_end = row["start"] + 2000

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


# keep only genes
filtered_gtf = original_gtf[(original_gtf["feature_type_g"] == "gene")]

# alter the gtf file by applying our tss function
tss_gtf = filtered_gtf.apply(tss_2kb, axis=1)

# save file
tss_gtf.to_csv(tss_file_path, sep="\t", header=False, index=False)
