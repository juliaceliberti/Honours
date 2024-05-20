import pandas as pd
import re


# extract gene_id from attribute column
def extract_gene_id(attributes_str):
    match = re.search(r'gene_id\s*"([^"]+)"', attributes_str)
    return match.group(1) if match else None


# utilising chunks due to memory limitations
chunk_size = 10000

# wgbs annotation path (extended genes +-2KB)
file_path = "HepG2_data/HepG2_DNAm/annotated/wgbs_extended_gene_annotations.bed"


chunk_list = []
chunk_track = 0

# read wgbs file in chunks
for chunk in pd.read_csv(
    file_path,
    chunksize=chunk_size,
    sep="\t",
    header=None,
    names=[
        "chrom",
        "chromStart",
        "chromEnd",
        "strand",
        "percentMeth",
        "chrom_g",
        "feature_type_g",
        "start_g",
        "end_g",
        "attributes_g",
    ],
    usecols=[0, 1, 2, 5, 10, 14, 16, 17, 18, 22],
):
    # extract gene_ids
    chunk["gene_id"] = chunk["attributes_g"].apply(extract_gene_id)

    # feature type is 'gene' and any NaN rows
    filtered_chunk = chunk[
        (chunk["feature_type_g"] == "gene")
        & chunk[["start_g", "end_g", "gene_id"]].notnull().all(axis=1)
    ]

    # group by gene_id and count rows for each gene
    aggregated = (
        filtered_chunk.groupby(["gene_id", "start_g", "end_g"])
        .size()
        .reset_index(name="count")
    )

    # get gene length
    aggregated["gene_length"] = aggregated["end_g"] - aggregated["start_g"]

    # Append to the list
    chunk_list.append(aggregated)
    chunk_track += 1
    if chunk_track % 1000 == 0:
        print("Completed processing of chunk {}".format(chunk_track))

final_aggregate = pd.concat(chunk_list, ignore_index=True)

# Since multiple chunks might have the same gene_id, aggregate again across all chunks
final_result = (
    final_aggregate.groupby(["gene_id", "gene_length"])
    .agg({"count": "sum"})
    .reset_index()
)

# normalise methylation count by gene length
final_result["norm_count"] = final_result["count"] / final_result["gene_length"]

# Save the final result to a new CSV file
final_result.to_csv(
    "HepG2_data/HepG2_DNAm/aggregated/wgbs_norm_aggregated_gene_counts.csv", index=False
)
