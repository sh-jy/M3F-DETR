# DOI
10.5281/zenodo.19675960
# M3F-DETR Abstract
In autonomous driving perception, real-time object detection under multi-scale and complex-background conditions remains challenging due to limited context modeling and feature coupling. This work presents a real-time detection transformer framework for complex road scenes, named M3F-DETR, which integrates Mamba-based long-range modeling, multi-scale semantic fusion, and frequency-domain-guided decoupled attention. The Long-Range Dependency Enhanced Module strengthens global context with low latency, the Multi-Scale Semantic Fusion Structure mitigates cross-scale feature inconsistency, and the Frequency-Domain-Guided Decoupled Attention enhances details and suppresses noise. Experiments on KITTI-ROD and Cityscapes datasets show the model achieves 54.2% AP, 89.28% mAP, and 97.1 FPS, balancing accuracy and efficiency effectively.

# 🔍 Overview
<img width="4479" height="1685" alt="Fig  1" src="https://github.com/user-attachments/assets/32de5fd4-a801-47ee-8ddd-d133b192fe69" />
Fig. 1 Overall architecture of M3F-DETR.

M3F-DETR is an end-to-end detection Transformer framework designed for complex road environments, with the goal of balancing detection accuracy and real-time performance. Unlike conventional detection pipelines that rely on anchor-box design and heuristic post-processing rules, M3F-DETR adopts a guided multi-branch training paradigm, in which the supervision signals and the query update process are structurally designed during training. This design improves the learnability of query representations and the stability of the decoding process without altering the inference path, thereby yielding more consistent classification and localization outputs. To address challenges in autonomous driving road scenes, including weak long-range semantics, imbalanced scale fusion, and texture-noise-induced attention bias, M3F-DETR introduces a Mamba-based Long-Range Dependency Enhanced Module (LDEM) after the backbone network, which compensates for cross-region long-range dependencies through low-latency sequence modeling and strengthens contextual discrimination in crowded and occluded scenarios. During the feature aggregation stage following encoder output, a Multi-Scale Semantic Fusion Structure (MSFS) is constructed to alleviate representational conflicts across scales through dynamically guided cross-level semantic alignment and consistency constraints, thereby improving the stable detection of distant objects. During the decoder query-feature interaction stage, a Frequency-Domain-Guided Decoupled Attention (FDGD) mechanism is designed to perform high-/low-frequency decoupled modeling and directed interactive fusion with the aid of frequency-domain priors. With modest computational overhead, it simultaneously enhances fine-grained structural details and suppresses global background noise, thereby mitigating texture-noise interference and alleviating attention bias. Benefiting from the collaborative modeling of LDEM, MSFS, and FDGD, M3F-DETR is able to maintain reliable detection performance and real-time inference capability even in distant-view scenarios and complex textured backgrounds. The overall framework is illustrated in Fig. 1.

# Dataset
KITTI is one of the most widely used computer vision benchmarks in the field of autonomous driving. In this paper, we focus on dynamic objects that are closely related to driving safety in complex road scenes and use them to evaluate the object detection performance of the model in traffic environments. We remove irrelevant categories in KITTI, such as DontCare and Misc, and select five object categories, namely car, pedestrian, tram, truck, and van, to construct an autonomous driving object detection dataset characterized by large scale variation, frequent occlusion, and complex backgrounds, which is referred to as KITTI-ROD. KITTI-ROD contains a total of 7,481 images, including 6,733 training images and 748 validation images. By preserving characteristics such as motion occlusion and dynamic road environments, KITTI-ROD is more consistent with the real-time perception requirements of autonomous driving and can be used to verify the generalization capability of detection algorithms in multi-scale and occluded scenarios.

Cityscapes is a commonly used high-resolution urban street-scene dataset in the field of autonomous driving. It contains 5,000 finely annotated images collected from road scenes across multiple cities and can be used to evaluate the robustness of detection models under multi-scale objects, dense traffic, and complex backgrounds. Since its original annotations are primarily designed for instance segmentation, this paper parses the instance masks into object instances and extracts their bounding boxes. From these, eight dynamic object categories that are most suitable for detection, namely bicycle, bus, car, motorcycle, person, rider, train, and truck, are selected and further converted into the PASCAL VOC format for object detection training and evaluation.

# Data Preparation
KITTI
Link：https://pan.baidu.com/s/1EPEV_z185GV8t-RE48lROA
Extraction code：u3zr

Link：https://pan.baidu.com/s/1w-ONPbnXAX7SSaNh_90q-g
Extraction code：osal

# Experimental setup
All experiments in this paper were conducted on the Ubuntu 22.04 operating system, with the hardware environment consisting of two NVIDIA RTX 4090 GPUs. In terms of the software environment, Python 3.9 was adopted as the programming language, and model training and inference were implemented based on the PyTorch 2.2.2 and Torchvision 0.17.2 deep learning frameworks. In addition, all experiments were adapted and reproduced under CUDA 12.1 to ensure experimental stability and portability. To guarantee fairness in the comparative experiments, all baseline methods in this paper were implemented based on the MMDetection toolbox of OpenMMLab. MMDetection provides a unified and highly modularized pipeline for training, inference, and evaluation, supports multiple vision tasks such as object detection, and integrates a rich model zoo, pretrained weights, and experiment management components. These advantages effectively reduce the experimental cost of method reproduction and ablation analysis, thereby providing reliable support for the performance evaluation and comparative experiments in this paper.

MMDetection：https://github.com/open-mmlab/mmdetection

# Environment Preparation
conda create -n your_env_name python=3.9 -y

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

pip install -e mamba-1p1p1

# Demo
<img width="6262" height="3943" alt="Fig  12" src="https://github.com/user-attachments/assets/984bff00-a7d6-4b73-99b6-87c26a86a0e9" />

# 👏 Acknowledgements
DETR：https://github.com/facebookresearch/detr

RT-DETR：https://github.com/lyuwenyu/RT-DETR

Vison Mamba：https://github.com/hustvl/Vim

Mr.DETR：https://github.com/Visual-AI/Mr.DETR

# 🪄 Notice
The code in the current repository is still under active refinement and has not yet been released as the final version. We will continue to update and improve the implementation details of M3F-DETR on an ongoing basis.
It is maintained to support the transparency and reproducibility of the reported experiments.
If you find this repository useful in your research, please consider citing the associated manuscript and the archived software release.
