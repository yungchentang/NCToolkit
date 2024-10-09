# Neural Clamping Toolkit
[![arXiv](https://img.shields.io/badge/arXiv-2209.11604-b31b1b.svg)](https://arxiv.org/abs/2209.11604)
![image](https://github.com/yungchentang/neural-clamping/blob/main/image/Neural_Clamping_Overview.png)


[**Paper**](https://arxiv.org/abs/2209.11604)
| [**Demonstration**](https://huggingface.co/spaces/TrustSafeAI/NCTV)
| [**Citation**](#citations)

## Overview
This repository contains code for the **TMLR Paper** "[Neural Clamping: Joint Input Perturbation and Temperature Scaling for Neural Network Calibration](https://arxiv.org/abs/2209.11604)." The research
demo of Neural Clamping can be found at [NCTV: Neural Clamping Toolkit and Visualization for Neural Network Calibration](https://huggingface.co/spaces/TrustSafeAI/NCTV), which earlier appeared in [AAAI 2023 Demo Track](https://arxiv.org/abs/2211.16274).

Authors: [Yung-Chen Tang](https://www.linkedin.com/in/yc-tang/), [Pin-Yu Chen](http://pinyuchen.com/), and [Tsung-Yi Ho](https://www.cse.cuhk.edu.hk/people/faculty/tsung-yi-ho/).

---
What the differnet between Confidence and Accuracy?
![image](https://github.com/yungchentang/neural-clamping/blob/main/image/conf_acc_demo_720p_compressed.gif)


## Uasge
Quick Start by running the following code!
```python
# !pip install -q git+https://github.com/yungchentang/NCToolkit.git
from neural_clamping.nc_wrapper import NCWrapper
from neural_clamping.utils import load_model, load_dataset, model_classes, plot_reliability_diagram

# Load model
model = load_model(name='ARCHITECTURE', data='DATASET', checkpoint_path='CHECKPOINT_PATH')
num_classes = model_classes(data='DATASET')

# Dataset loader
valloader = load_dataset(data='DATASET', split='val', batch_size="BATCH_SIZE")
testloader = load_dataset(data='DATASET', split='test', batch_size="BATCH_SIZE")

# Build Neural Clamping framework
nc = NCWrapper(model=model, num_classes=num_classes, ...)

# Calibrated using Neural Clamping
nc.train_NC(val_loader=valloader, epoch='EPOCH', ...)

# General Evaluation
nc.test_with_NC(test_loader=testloader)

# Visualization
bin_acc, conf_axis, ece_score = nc.reliability_diagram(test_loader=testloader, rd_criterion="ECE", n_bins=30)
plot_reliability_diagram(conf_axis, bin_acc)
```

## Citations
If you find this helpful for your research, please cite our papers as follows:

    @article{tang2024neural,
      title={{Neural Clamping: Joint Input Perturbation and Temperature Scaling for Neural Network Calibration}},
      author={Yung-Chen Tang and Pin-Yu Chen and Tsung-Yi Ho},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2024},
      url={https://openreview.net/forum?id=qSFToMqLcq},
    }
    
    @inproceedings{hsiung2023nctv,
      title={{NCTV: Neural Clamping Toolkit and Visualization for Neural Network Calibration}}, 
      author={Lei Hsiung and Yung-Chen Tang and Pin-Yu Chen and Tsung-Yi Ho},
      booktitle={Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence},
      publisher={Association for the Advancement of Artificial Intelligence},
      year={2023},
      month={February}
    }

