## ghostnet

implementation of the paper [ghostnet paper](https://arxiv.org/abs/1911.11907) 

### running locally

- start a virtual env with `python3 -m venv .` 
- run `python ghostnet.py` 

### experiments

#### ghostnet-resnet-56

| no.   | ratio(s) | kernel(d) | optimizer | lr_scheduler          | accuracy(ours) | accuracy(paper) | epochs | file          | comment |
| ----- | -----    | -----     | -----     | -----                 | -----          | -----           | -----  | ----          | -----   |
| 1     | 2        | 3         | SGD       | CosineAnnealing       | 92.20%        | 92.7%           | 200     | gr_cosine.pth | Had smooth convergence|
| 2     | 2        | 3         | SGD       | MultiStepLR(100, 500) | 92.55%        | 92.7%           | 200     | gr_multistep.pth | Had smooth convergence|

- total feature maps = s * number of intrinsic feature maps (produced by ordinary convolution filters)
- `images/ghost_visualization.png` (dog's image, cifar index=12) proves that ghost map preserves the exact same shape and pose as the intrinsic maps
- the ghost versions appear slightly shifted and smoothened. this confirms that the cheap operation (depthwise convolutions) learned useful transformations like blurring and edge enhancement to augment the feature space without the computationally heavy full convolution operation.
- computational reduction(measured using `thops`):
```
measuring standard resnet-56...
  flops: 127.93M
  params: 0.86M

measuring ghost-resnet-56...
  flops: 67.50M
  params: 0.44M

results:
  flops reduction: 1.90x (paper claims ~2x)
  param reduction: 1.95x (paper claims ~2x)
```
#### ghost-vgg-16

Experiment run on CIFAR-10 using a dynamic layer replacement strategy (swapping `nn.Conv2d` for `GhostModule`).

| ratio(s) | kernel(d) | optimizer | lr_scheduler    | accuracy(ours) | epochs | hardware | time  |
| -----    | -----     | -----     | -----           | -----          | -----  | ----     | ----  |
| 2        | 3         | SGD       | CosineAnnealing | 93.60%         | 200    | T4 GPU   | ~1.3h |

- **Implementation**: Unlike the ResNet implementation, this experiment utilized a `replace_conv_with_ghost` utility to recursively modify a standard VGG-16 architecture at runtime.
- **Convergence**: Achieved 93.60% accuracy at epoch 200. Best accuracy observed was ~93.63%.
- **Computational Reduction**:
  
Standard VGG Params: 14.99M Ghost-VGG Params: 7.65M
results: param reduction: 1.96x (48.97% reduction)
