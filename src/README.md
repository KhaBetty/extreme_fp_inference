
# Post-Training Quantization

## Requirements

Install PyTorch. Specifically, we use version 1.5.1 with CUDA 10.1.
```pytorch
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Pick PyTorch 1.5.1 with the appropriate CUDA version from the [official PyTorch website](https://pytorch.org/).  
Then, install the other packages and our custom CUDA package:
```setup
pip install -r requirements.txt
cd cu_gemm_quant
python ./setup install
```
The ImageNet path, as well as the seeds used to achieve the paper's results, are configured in `Config.py`.  
Throughout this work, we used Ubuntu 18.04, Python 3.6.9, and NVIDIA TITAN V GPU.  

## 8-bit Model Quantization

To quantize the models, execute the following command:

```quantize
python ./main.py -a resnet18_imagenet --action QUANTIZE --x_bits 5 --w_bits 5
```
We support the following models: `resnet18_imagenet`, `resnet34_imagenet`, `resnet50_imagenet`, `resnet101_imagenet`, `googlenet_imagenet`, `inception_imagenet`, `densenet_imagenet`.

## run command in our modification

The running command same as mentioned above.
* For running in real dynamic bias setup change in file `QuantConv2d.py` change flag `max_bias_sum = False`.
* For running in dynamic bias setup change in file `QuantConv2d.py` change flag `max_bias_sum = True`.
* For changing the mantissa sizes change in file `Config.py` change dictionary X_FP, W_FP, SUM_FP.
