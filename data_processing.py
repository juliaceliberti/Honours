import subprocess
import os
from pathlib import Path

# data folder to search
root_dir = 'HepG2_data'

# process files based on their type
def process_file(file_path):
    file_suffix = file_path.suffix
    output_dir = file_path.parent / 'processed'  # processed files saved to a 'processed' subdir
    output_dir.mkdir(exist_ok=True)  # creates the directory if it doesn't exist

    if file_suffix == '.bigBed':
        output_file = output_dir / (file_path.stem + '.bed')
        subprocess.run(['bigBedToBed', str(file_path), str(output_file)])
    elif file_suffix == '.bigWig':
        output_file = output_dir / (file_path.stem + '.bedGraph')
        subprocess.run(['bigWigToBedGraph', str(file_path), str(output_file)])
    elif file_suffix == '.bed.gz':
        output_file = output_dir / file_path.stem
        subprocess.run(['gunzip', '-c', str(file_path)], stdout=open(output_file, 'wb'))

# walk through the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        file_path = Path(subdir) / filename
        
        # filter out files not matching the bigBeg, bigWig, bed.gz
        if file_path.suffix in ['.bigBed', '.bigWig', '.bed.gz']:
            process_file(file_path)
