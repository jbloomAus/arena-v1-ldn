# conda

curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
bash Miniconda3.sh

python3 -m pip install --upgrade pip
conda env create -f environment_ubuntu.yml