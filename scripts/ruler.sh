nvidia-smi
pip install --upgrade pybind11 onnxruntime
pip install --upgrade matplotlib
pip install html2text
pip install beautifulsoup4
pip install wonderwords
pip install flash-attn==2.6.0.post1 --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
pip install nltk
git clone https://github.com/Rya-Sanovar/RULER.git
sleep 800h
# cd RULER
# cd scripts/data/synthetic/json/
# python download_paulgraham_essay.py
# bash download_qa_dataset.sh
# Phi-3.5-mini-instruct synthetic
# /mnt/azureml/cr/j/b2269e618d4b45b6bb43054a01a747db/exe/wd/
# export HF_HOME="/mnt/azureml/cr/j/b2269e618d4b45b6bb43054a01a747db/exe/wd/hf_cache/"
# export HF_DATASETS_CACHE="/mnt/azureml/cr/j/b2269e618d4b45b6bb43054a01a747db/exe/wd/hf_cache/datasets"
# sudo fuser -v /dev/nvidia0  # List processes using the GPU
# sudo fuser -k /dev/nvidia0  # Kill all processes using the GPU
