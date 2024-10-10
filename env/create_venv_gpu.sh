./env/modules_gpu.sh

python3.11 -m venv --system-site-packages ./env/venv_gpu

source ./env/venv_gpu/bin/activate

pip install -e .