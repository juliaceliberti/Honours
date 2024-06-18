import pandas as pd
import re


# extract gene_id from attribute column
def extract_gene_id(attributes_str):
    match = re.search(r'gene_id\s*"([^"]+)"', attributes_str)
    return match.group(1) if match else None


# histone annotation path (extended genes +-2KB)
file_path = "HepG2_data/HepG2_histone/annotated/H3K27me3_pp_annot_2kb.bed"
output_file = "HepG2_data/HepG2_histone/aggregated/h3k27me3_agg_2kb_gene_counts.csv"

# utilising chunks due to memory limitations
chunk_size = 10000
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
        "peakStart",
        "peakEnd",
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
):
    # extract gene_ids
    chunk["gene_id"] = chunk["attributes_g"].apply(extract_gene_id)

    # Calculate under_peak_count as the difference between peakStart and peakEnd
    chunk["under_peak_count"] = chunk["peakEnd"] - chunk["peakStart"]

    # feature type is 'gene' and any NaN rows
    filtered_chunk = chunk[
        (chunk["feature_type_g"] == "gene") & chunk["gene_id"].notnull()
    ]

    # group by gene_id and count rows for each gene
    aggregated = (
        filtered_chunk.groupby(["gene_id"])["under_peak_count"].sum().reset_index()
    )

    # Append to the list
    chunk_list.append(aggregated)
    chunk_track += 1
    if chunk_track % 1000 == 0:
        print("Completed processing of chunk {}".format(chunk_track))

final_aggregate = pd.concat(chunk_list, ignore_index=True)

# Since multiple chunks might have the same gene_id, aggregate again across all chunks
final_result = (
    final_aggregate.groupby(["gene_id"])["under_peak_count"].sum().reset_index()
)


# Save the final result to a new CSV file
final_result.to_csv(output_file, index=False)
