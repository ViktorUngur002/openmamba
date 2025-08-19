## Installation

1. First of all you need to clone this repository.

    ```
    git clone https://github.com/ViktorUngur002/openmamba.git
    ```

2. Create a conda environment.

    ```
    conda create --name open_mamba python=3.9
    conda activate open_mamba
    ```

3. Install the appropriate PyTorch version that matches your CUDA version and is compatible with Detectron2.

    ```
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
    ```

4. Install Detectron2. For additional information please consult [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

    ```
    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2
    ```

5. Install the remaining requirements and mamba-ssm.

    ```
    pip install -r requirements.txt
    pip install mamba-ssm==2.0.4
    ```

6. Build the pixel decoder.

    ```
    cd maft/modeling/pixel_decoder/ops
    sh make.sh
    ```