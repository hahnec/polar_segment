git clone git@github.com:hahnec/mm_torch
ln -s mm_torch/mm_torch ./mm

python -m venv venv
source venv/bin/activate

python -m pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
