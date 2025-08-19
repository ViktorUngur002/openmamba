## How to use OpenMamba

The following document provides instructions for training and evaluating our OpenMamba model.

### Ground-truth warmup training

First you need to set up the corresponding datasets as in [datasets/README.md](datasets/README.md). After doing so you can start the first training phase by running the script `train_net_open_mamba.py` with the following command:

```
python train_net_open_mamba.py --num-gpus N \
    --config-file configs/ground-truth-warmup/open_mamba/open_mamba_maft_convnext_base_cocostuff_eval_ade20k.yaml \
    MODEL.WEIGHTS /path/to/maftp_b.pth
```

__Mentions:__ 

* You should replace `N` with the number of GPUs available to you when training. The configuration is set for a single GPU training. Feel free to modify the learning rate and batch size in the configuration file accordingly.

* This model was trained and evaluated using only the __base__ version of CLIP. Feel free to download the corresponding MAFT+ weights from the following [link](https://drive.google.com/file/d/1BeEeKOnWWIWIH-QWK_zLhAPUzCOnHuFG/view).


### Combining OpenMamba weights with Mask2Former

The first training phase does not include the Mask2Former weights. In order to combine the weights, you will need to run the following command:

```
python tools/weight_fuse.py \
    --model_first_phase_path /path/to/first_phase_weights.pth \
    --model_sem_seg_path /path/to/maftp_b.pth \
    --output_path /path/to/merged_weights.pth
```

### Mixed-mask training

For the second phase of training we are using OpenMamba with MAFT+ model. To run this phase, use the script `train_net_maftp.py` and run it with the following command:

```
python train_net_maftp.py --num-gpus N \
    --config-file configs/mixed-mask-training/maftp/semantic/train_semantic_base_eval_a150.yaml \
    MODEL.WEIGHTS /path/to/merged_weights.pth
```

In order to evaluate the model, you can use the following command:

```
python train_net_maftp.py --num-gpus N \
    --config-file configs/mixed-mask-training/maftp/semantic/train_semantic_base_eval_a150.yaml \
    --eval-only MODEL.WEIGHTS /path/to/checkpoint.pth
```

For evaluating the model on other datasets, please modify the config file field `DATASETS.TEST` with corresponding data from the files with an __eval__ prefix, these files can be found at `configs/mixed-mask-training/maftp/semantic`.

