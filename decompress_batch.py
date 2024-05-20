### Script to decompress .gz wgbs file in batches as too large to use gunzip in command line
import gzip

# Size of each chunk to read (bytes)
chunk_size = 1024 * 1024 * 10

# Open the gzipped file
with gzip.open("HepG2_data/HepG2_DNAm/annotated/combined_wgbs.bed.gz", "rb") as f_in:
    # Open the output file
    with open(
        "HepG2_data/HepG2_DNAm/annotated/combined_wgbs_decompressed.bed", "wb"
    ) as f_out:
        while True:
            # Read a chunk from the gzipped file
            chunk = f_in.read(chunk_size)
            if not chunk:
                break  # No more data in the file
            # Write the decompressed chunk to the output file
            f_out.write(chunk)
