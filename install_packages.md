# create the conda env loading cineca-ai module
module load anaconda3/2023.03
module load profile/deeplrn
module av cineca-ai
module load cineca-ai/<version>
conda create -p <path>/my_env -c conda-forge --override-channels   
 
# activate the created conda env to access cineca packages and install your conda packages.
conda activate <path>/my_env
python -m pip list
python -m pip install my_packages
conda deactivate