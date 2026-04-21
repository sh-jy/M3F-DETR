# 🏠 M3F-DETR Abstract
在自动驾驶感知领域，由于上下文建模和特征耦合能力有限，多尺度复杂背景下的实时目标检测仍面临挑战。本研究提出了一种面向复杂道路场景的实时检测Transformer框架，名为M3F-DETR，该框架融合了基于Mamba的长距离建模、多尺度语义融合以及频域引导的解耦注意力机制。长距离依赖增强模块以低延迟强化全局上下文，多尺度语义融合结构缓解跨尺度特征不一致，频域引导的解耦注意力则增强细节并抑制噪声。在KITTI-ROD和Cityscapes数据集上的实验表明，该模型实现了54.2%的AP、89.28%的mAP以及97.1 FPS，有效兼顾了精度与效率。
# 🔍 Overview
<imgwidth="4479" height="1685" alt="图 1" src="https://github.com/user-attachments/assets/8e6eb197-fa8c-4822-a468-9a7d48220fbc" />
图1 M3F-DETR的整体架构。
M3F-DETR 是一种端到端检测 Transformer 框架，专为复杂道路环境而设计，旨在平衡检测精度与实时性能。与依赖锚框设计和启发式后处理规则的传统检测流水线不同，M3F-DETR 采用了一种引导式多分支训练范式，在训练过程中对监督信号和查询更新过程进行了结构化设计。这一设计在不改变推理路径的前提下，提升了查询表示的可学习性，并增强了解码过程的稳定性，从而带来更一致的分类与定位输出。为应对自动驾驶道路场景中面临的挑战——包括长程语义薄弱、尺度融合不平衡以及纹理噪声引起的注意力偏差——M3F-DETR在主干网络之后引入了一个基于Mamba的长程依赖增强模块（LDEM），该模块通过低延迟序列建模弥补了跨区域间的长程依赖关系，并强化了在拥挤和遮挡场景中的上下文区分能力。在编码器输出后的特征聚合阶段，构建了一种多尺度语义融合结构（MSFS），通过动态引导的跨层级语义对齐与一致性约束，缓解不同尺度间的表征冲突，从而提升远距离目标的稳定检测性能。在解码器查询-特征交互阶段，设计了一种频域引导的解耦注意力机制（FDGD），该机制借助频域先验，实现高频/低频解耦建模，并进行定向交互融合。凭借适度的计算开销，它能够同时增强细粒度的结构细节并抑制全局背景噪声，从而有效缓解纹理噪声干扰，减轻注意力偏差。得益于LDEM、MSFS和FDGD的协同建模，M3F-DETR即使在远距离视角和复杂纹理背景下，也能保持可靠的检测性能和实时推理能力。整体框架如图1所示。
#数据集
KITTI是自动驾驶领域应用最广泛的计算机视觉基准之一。在本文中，我们重点关注与复杂道路场景中驾驶安全密切相关的动态物体，并以此来评估模型在交通环境中的目标检测性能。我们去除了KITTI数据集中不相关的类别，如DontCare和Misc，并选取了五个目标类别——汽车、行人、有轨电车、卡车和厢式货车，构建了一个以大规模变化、频繁遮挡和复杂背景为特征的自动驾驶目标检测数据集，该数据集被称为KITTI-ROD。KITTI-ROD共包含7,481张图像，其中6,733张为训练图像，748张为验证图像。KITTI-ROD通过保留运动遮挡和动态道路环境等特征，更贴近自动驾驶的实时感知需求，可用于验证检测算法在多尺度和遮挡场景下的泛化能力。
Cityscapes是自动驾驶领域中一种常用的高分辨率城市街景数据集。它包含从多个城市道路场景中采集的5,000张精细标注的图像，可用于评估检测模型在多尺度目标、密集交通和复杂背景下的鲁棒性。由于其原始标注主要针对实例分割，本文将实例掩码解析为对象实例，并提取它们的边界框。在此基础上，选取了最适合检测的八个动态对象类别，即自行车、公交车、小汽车、摩托车、人、骑手、火车和卡车，并进一步将其转换为PASCAL VOC格式，以用于目标检测的训练与评估。
#数据准备
KITTI
链接：https://pan.baidu.com/s/1EPEV_z185GV8t-RE48lROA
提取码：u3zr

链接：https://pan.baidu.com/s/1w-ONPbnXAX7SSaNh_90q-g
提取码：osal

本文所有实验均在Ubuntu 22.04操作系统上进行，硬件环境配备两块NVIDIA RTX 4090显卡。软件环境方面，采用Python 3.9作为编程语言，模型训练与推理基于PyTorch 2.2.2和Torchvision 0.17.2深度学习框架实现。此外，所有实验均在CUDA 12.1下进行适配和复现，以确保实验的稳定性和可移植性。为保证对比实验的公平性，本文中所有基线方法均基于OpenMMLab的MMDetection工具箱实现。MMDetection提供了一套统一且高度模块化的训练、推理和评估流程，支持多种视觉任务，如目标检测，并集成了丰富的模型库、预训练权重以及实验管理组件。这些优势有效降低了方法复现和消融分析的实验成本，从而为本文的性能评估和对比实验提供了可靠支撑。
MMDetection：https://github.com/open-mmlab/mmdetection
#环境准备
conda create -n your_env_name python=3.9 -y
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e mamba-1p1p1
#演示
<imgwidth="6262" height="3943" alt="图 12" src="https://github.com/user-attachments/assets/1ea28b4d-b71a-4e34-9ed5-e945f0544e2d" />
#致谢
DETR：https://github.com/facebookresearch/detr
RT-DETR：https://github.com/lyuwenyu/RT-DETR
Vison Mamba：https://github.com/hustvl/Vim
Mr.DETR：https://github.com/Visual-AI/Mr.DETR
# 🪄 Notice
The code in the current repository is still under active refinement and has not yet been released as the final version. We will continue to update and improve the implementation details of M3F-DETR on an ongoing basis.
