MM_TORCH_DIR="mm_torch"
if [ -d "$MM_TORCH_DIR" ]; then
    echo "Directory '$MM_TORCH_DIR' already exists. Skipping git clone."
else
    echo "Directory '$MM_TORCH_DIR' does not exist. Cloning repository..."
    git clone git@github.com:hahnec/mm_torch
fi

MM_DIR="mm"
if [ ! -d "$MM_DIR" ]; then
    ln -s mm_torch/mm_torch ./mm
fi

AUGMENT_DIR="polar_augment"
if [ -d "$AUGMENT_DIR" ]; then
    echo "Directory '$AUGMENT_DIR' already exists. Skipping git clone."
else
    echo "Directory '$AUGMENT_DIR' does not exist. Cloning repository..."
    git clone git@github.com:hahnec/polar_augment
fi

VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Directory '$VENV_DIR' already exists. Skipping git clone."
else
    python -m pip install pip --upgrade
    python -m venv venv
fi

source venv/bin/activate
python -m pip install -r requirements.txt
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
