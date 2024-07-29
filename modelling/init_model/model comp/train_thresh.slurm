#!/bin/bash -l
#SBATCH --job-name=genejc_rf_dnam
#SBATCH --partition=work
#SBATCH --nodes=2
#SBATCH --ntasks=30
#SBATCH --time=3-00:00:00
#SBATCH --mem=326G
#SBATCH --export=ALL

# Load the correct Anaconda module
module load Anaconda3/2024.06  # Replace with the correct version
conda activate genejc


# Note: SLURM_JOBID is a unique number for every job.
SCRIPT=class_rf_dnam.py

# Use MYSCRATCH space
SCRATCH=$MYSCRATCH/run_conda/$SLURM_JOBID
RESULTS=$MYGROUP/conda_results

###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then 
    mkdir -p $SCRATCH 
fi 
echo "Working SCRATCH directory is $SCRATCH"

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then 
     mkdir -p $RESULTS
fi 
echo "Results will be stored in $RESULTS/$SLURM_JOBID"

#############################################
# Copy input files to $SCRATCH
# Change directory to $SCRATCH
# cd ${SLURM_SUBMIT_DIR}

# cp ${SCRIPT} ${SCRATCH}

# cd ${SCRATCH}

# ls -al

# Check if the threshold argument is provided
if [ -z "$1" ]; then
    echo "Error: No threshold value provided"
    exit 1
fi

# Echo print the threshold value selected
echo "Threshold value selected: $1"

#############################################
# FOR MORE THAN ONE NODE 

# Start Dask MPI cluster
dask-mpi --nthreads 1 --memory-limit 0 --interface ib0 &

# Allow time for the cluster to start
sleep 60

#############################################

# Run the Python script with the threshold value as an argument
python ./${SCRIPT} $1 $2  # Pass '0' for no undersampling, '1' for undersampling


#############################################
# Move the $OUTPUT file to the unique results dir
# note this can be a copy or move  
cd $HOME
mv ${SCRATCH} ${RESULTS}

echo "Conda ML gpu job finished at `date`"
