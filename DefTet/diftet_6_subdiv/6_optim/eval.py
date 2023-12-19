import os
import subprocess

# Define your base directories for data and output
base_datadir = '/u/singularity/DefTet/data'
base_savedir = '/u/singularity/DefTet/final_output'

# List of paths for point clouds (update this list as needed)
point_cloud_paths = [
    '/u/singularity/DefTet/data/lego_t/4ab7bd2c-8/point_cloud/iteration_30000/point_cloud.ply',
    # Add more paths here
]

datasets = ['drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship']

# Change to the necessary directory
#os.chdir('diff_render/diftet_6_subdiv/6_optim')

# Function to run the command
def run_command(with_gaussian, dataset, gaussian_path=''):
    command = [
        'python', 'optim_with_mask_subdiv_from_gridmov.py',
        '--expname', dataset,
        '--datadir', base_datadir,
        '--savedir', base_savedir,
        '--remote'
    ]
    if with_gaussian:
        command.extend(['--gaussianpth',  os.path.join(os.path.join(base_datadir, dataset), 'point_cloud.ply')])

    subprocess.run(command)

# Run commands for each point cloud path
for dataset in datasets:
    # Run without gaussianpth
    run_command(False, dataset)
    
    # Run with gaussianpth
    run_command(True, dataset)