import subprocess
import os
from pathlib import Path

# data folder to search
root_dir = 'HepG2_data'

def process_file(file_path):
    file_suffix = file_path.suffix
    output_dir = file_path.parent / 'processed'
    output_dir.mkdir(exist_ok=True)

    if file_suffix == '.bigBed':  # Corrected to check file_suffix
        output_file = output_dir / (file_path.stem + '.bed')
        if not output_file.exists():  # Check if the output file already exists
            subprocess.run(['bigBedToBed', str(file_path), str(output_file)])
    elif file_suffix == '.bigWig':  # Corrected to check file_suffix
        output_file = output_dir / (file_path.stem + '.bedGraph')
        if not output_file.exists():  # Check if the output file already exists
            subprocess.run(['bigWigToBedGraph', str(file_path), str(output_file)])
    elif file_suffix == '.gz':
        output_file = output_dir / file_path.with_suffix('').name  # Correctly handle .bed.gz extension
        if not output_file.exists():  # Check if the output file already exists
            with open(output_file, 'wb') as f:
                subprocess.run(['gunzip', '-c', str(file_path)], stdout=f)




# walk through the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        file_path = Path(subdir) / filename
        print("Found file:", file_path)  # Verify files are being found
        
        # filter out files not matching the bigBeg, bigWig, bed.gz
        if file_path.suffix in ['.bigBed', '.bigWig', '.gz']:
            process_file(file_path)