git clone git@github.com:hahnec/mm_torch

ln -s mm_torch/mm_torch ./mm

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
