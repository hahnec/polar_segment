git clone git@github.com:hahnec/mm_torch

ln -s mm_torch/mm_torch ./mm

python -m venv venv
source venv/bin/activate

python -m pip install -r requirements.txt
