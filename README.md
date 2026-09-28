# SRCNN in PyTorch

A configurable PyTorch implementation of **Super-Resolution Convolutional Neural Network (SRCNN)** for single-image super-resolution. The project follows the principal SRCNN architecture proposed by Dong *et al.* [[1]](#ref-1), while providing a practical PyTorch training, evaluation, and inference pipeline.

The default experiment uses:

- **Model:** SRCNN(9,1,5)
- **Scale factor:** ×2
- **Training data:** 91-image dataset [[2]](#ref-2)
- **Evaluation data:** Set5 and Set14 [[3]](#ref-3)[[4]](#ref-4)
- **Reconstructed channel:** MATLAB-compatible luminance \(Y\)
- **Chrominance:** bicubic interpolation
- **Framework:** PyTorch

---

## SRCNN architecture
<p align="center">
<img width="850" height="341" alt="image" src="https://github.com/user-attachments/assets/97cfc304-b408-456c-b00f-38ce57ec933a" />
</p>

<p align="center"><em>Original SRCNN architecture: patch extraction and representation, non-linear mapping, and reconstruction. Figure is taken from Dong et al. [1].</em></p>

SRCNN is a three-layer pre-upsampling network. A true low-resolution image is first enlarged to the target size using bicubic interpolation. Then the interpolated luminance channel is processed by:

1. **Patch extraction and representation:** `Conv(f1, n1) + ReLU`
2. **Non-linear mapping:** `Conv(f2, n2) + ReLU`
3. **Reconstruction:** `Conv(f3, 1)` with no output activation

The default configuration is the basic **9-1-5** model with `n1=64` and `n2=32`. Training uses valid convolutions. For a `33 × 33` input patch, the default model produces an aligned `21 × 21` high-resolution target:

```text
output_size = input_size - (f1 - 1) - (f2 - 1) - (f3 - 1)
            = 33 - 8 - 0 - 4
            = 21
```

The network predicts only the **Y channel**. The Cb and Cr channels are resized using bicubic interpolation and combined with the reconstructed Y channel to produce the final RGB image.

## Configurable parameters

| Parameter | CLI option | Default | Description |
|---|---|---:|---|
| Upscaling factor | `--scale` | `2` | Supported values: ×2, ×3, and ×4. A separate model is trained for each scale. |
| First filter size | `--f1` | `9` | Patch extraction and representation kernel size. |
| Mapping filter size | `--f2` | `1` | Non-linear mapping kernel size. |
| Reconstruction filter size | `--f3` | `5` | Final reconstruction kernel size. |
| First-layer features | `--n1` | `64` | Number of feature maps produced by the first layer. |
| Mapping features | `--n2` | `32` | Number of feature maps in the mapping layer. |
| Input patch size | `--input-size` | `33` | Bicubic-upscaled luminance patch supplied to SRCNN. |
| Epochs | `--epochs` | `100` | Number of complete training epochs. |
| Repetitions | `--repeat` | `100` | Random samples drawn per training image in each epoch. |
| Batch size | `--batch-size` | `64` | Number of aligned LR/HR patch pairs per update. |
| Learning rate | `--lr` | `1e-3` | Adam learning rate. |

## Differences from the original Caffe implementation

The architecture follows SRCNN, but the training procedure is a modern PyTorch adaptation rather than a bit-exact reproduction of the released Caffe/MATLAB pipeline.

| Item | Original/released Caffe-based setup | This PyTorch project |
|---|---|---|
| **Framework** | Caffe network with MATLAB data preparation; the paper also reports cuda-convnet as its main framework | PyTorch network with Python/NumPy data preparation |
| **Loss implementation** | Caffe `EuclideanLoss`: squared L2 error divided by batch size, with a factor of 1/2 | PyTorch mean MSE averaged over batch, channel, height, and width |
| **Optimizer** | SGD with momentum `0.9` | Adam |
| **Learning rate** | Base `1e-4` with layer-wise multipliers; final-layer weights and all biases effectively use `1e-5` | `1e-3` for all trainable parameters by default |
| **Samples per training cycle** | Approximately 24,800 fixed `33 × 33` sub-images extracted from 91 images with stride 14 | `91 × 100 = 9,100` randomly cropped aligned pairs per epoch with the default `repeat=100` |
| **Interpolation implementation** | MATLAB `imresize(..., 'bicubic')` | Python implementation of MATLAB-style bicubic resizing with antialiasing for downsampling |
| **Color reconstruction** | Main experiments reconstruct Y; the paper also investigates joint three-channel models | SRCNN reconstructs Y, then the output RGB image is formed with bicubic Cb and Cr |
| **Training dataset** | 91-image dataset in the reference Caffe training setup | 91-image dataset |
| **Chroma** | Cb and Cr are not learned in the main Y-channel protocol | Cb and Cr are bicubically resized and are not learned |

## Comparison with the original paper

The following results were read from the included `checkpoints/best_psnr.pth` checkpoint. PSNR and SSIM are evaluated on the Y channel.

| Dataset | Original paper PSNR / SSIM [1] | This project PSNR / SSIM |
|---|---:|---:|
| Set5 | **36.66 / 0.9542** | **36.1882 / 0.95654** |
| Set14 | **32.45 / 0.9067** | **32.1637 / 0.91163** |

> **Comparison note:** the paper [1] values above correspond to the authors' larger 9-5-5 model trained on ImageNet, whereas the included checkpoint uses the basic 9-5-5 model trained on the 91-image dataset. The values therefore show the reported performance context, not a controlled one-to-one reproduction.


## Setup

Create and activate a PyTorch environment, then install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the datasets as follows:

```text
data/
├── 91-image/
├── Set5/
└── Set14/
```

The project reads common image formats including PNG, JPEG, BMP, and TIFF. LR inputs are generated from the HR images during training and evaluation using bicubic downsampling.

## Train

Edit dataset paths or model parameters in `scripts/train.sh`, then run:

```bash
bash scripts/train.sh
```

The default training configuration is:

```text
scale:          ×2
architecture:   9-1-5, n1=64, n2=32
epochs:         100
repeat:         100
batch size:     64
optimizer:      Adam
learning rate:  1e-3
loss:           MSE
```

Training writes checkpoints and metrics to:

```text
runs/srcnn_adam_x2_9_1_5/
├── best.pth
├── best_psnr.pth
├── best_ssim.pth
├── latest.pth
├── metrics.csv
└── epochs/
```


## Test / evaluation

The evaluation script tests the included best-PSNR checkpoint on Set5 and Set14:

```bash
bash scripts/evaluate.sh
```

By default, it uses:

```text
checkpoints/best_psnr.pth
```

Results are written to:

```text
results/Set5_srcnn_x2/
results/Set14_srcnn_x2/
```

Each output directory contains reconstructed images and a `metrics.json` file. PSNR and SSIM are computed on the Y channel.

## Inference

Edit the image path in `scripts/infer.sh`, then run:

```bash
bash scripts/infer.sh
```

For an HR ground-truth image, the script saves:

- the scale-compatible **ground-truth image**;
- the true **LR input**, whose width and height are divided by the scale factor;
- the LR input enlarged to the ground-truth size with **nearest-neighbor interpolation**;
- the LR input enlarged with **bicubic interpolation**;
- the **SRCNN output**, with the same dimensions as the ground truth.

The nearest-neighbor output is intended for visual comparison. SRCNN itself receives the bicubic-upscaled Y channel.


## References

[1] C. Dong, C. C. Loy, K. He, and X. Tang, “Image Super-Resolution Using Deep Convolutional Networks,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 38, no. 2, pp. 295–307, 2016. [doi:10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281) · [arXiv:1501.00092](https://arxiv.org/abs/1501.00092)

[2] J. Yang, J. Wright, T. S. Huang, and Y. Ma, “Image Super-Resolution via Sparse Representation,” *IEEE Transactions on Image Processing*, vol. 19, no. 11, pp. 2861–2873, 2010. This work is the source commonly associated with the 91-image/T91 training set. [doi:10.1109/TIP.2010.2050625](https://doi.org/10.1109/TIP.2010.2050625)

[3] M. Bevilacqua, A. Roumy, C. Guillemot, and M.-L. Alberi-Morel, “Low-Complexity Single-Image Super-Resolution Based on Nonnegative Neighbor Embedding,” in *BMVC*, 2012. The Set5 benchmark is attributed to this work. [doi:10.5244/C.26.135](https://doi.org/10.5244/C.26.135)

[4] R. Zeyde, M. Elad, and M. Protter, “On Single Image Scale-Up Using Sparse-Representations,” in *Curves and Surfaces*, LNCS 6920, pp. 711–730, 2012. The Set14 benchmark is attributed to this work. [doi:10.1007/978-3-642-27413-8_47](https://doi.org/10.1007/978-3-642-27413-8_47)


## Acknowledgment

This repository is a PyTorch reimplementation developed for research and educational use. The network figure is taken from the original SRCNN publication and is included with attribution to the authors.
