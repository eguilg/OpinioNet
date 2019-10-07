0.运行环境
    软件：
        Ubuntu 18.04
        Python: 3.6.5
        Pytorch: 1.1.0
        CUDA: 9.0
        CUDNN: 7.1.3
    硬件：
        显卡：GTX1080 8G单卡
        内存：16G
        CPU: i7 7700

1.安装requirements.txt中的依赖
    requirements.txt所在路径执行:
    pip install -r requirements.txt

2.训练
    如需一键训练，执行　sh train_script.sh　可能耗时较长。

    或者分步运行：
    2.1 无监督laptop语料 + 有标注makeup数据 进行预训练
        在src/文件夹下运行：
        python pretrain.py --base_model roberta
        python pretrain.py --base_model wwm
        python pretrain.py --base_model ernie
        从而分别对roberta、wwm、ernie三种模型进行预训练, 权重保存在models/文件夹下, 分别为:
        pretrained_roberta, pretrained_wwm, pretrained_ernie
        预训练耗时较长，可以跳过，在下一步微调中直接使用我们提供的权重，节省时间。

    2.2 有标注laptop数据 进行交叉验证finetune
        在src/文件夹下运行：
        python finetune_cv.py --base_model roberta
        python finetune_cv.py --base_model wwm
        python finetune_cv.py --base_model ernie
        分别对三种预训练模型进行微调，权重保存至models/文件夹下，命名形如roberta_cvX, X代表cv的折数，设定为5折，即X为0~4
        微调过程中会保存各个模型在验证集中最佳的筛选阈值，在models/thresh_dict.json中。

3.测试
    如需一键测试，执行　sh eval_script.sh

    或者：
    在src/文件夹下运行：
    python eval_ensemble_round2.py
    结果保存为submit/Result.csv

    我们提供了单模型1折的权重models/roberta_cv2, 可以一键测试得到输出结果, 应该比线上成绩稍低。

4.算法原理
    模型特点：One-stage端到端，无需分别进行实体抽取和关系分类, OpinoNet Only Look Once。
    基础模型为使用bert预训练模型作为骨架的One-stage端到端实体关系抽取模型，使用了roberta-wwm、bert-wwm、ernie等初始预训练模型。
    对于复赛中数据特点，基本流程如下：
        1. 首先使用无监督laptop语料和有标注的makeup数据进行MLM和当前下游任务的双任务训练。
        2. 在数量较少的有标注laptop语料上对上一步得到的模型进行交叉验证训练微调，将模型迁移至laptop领域。
        3. 不同初始预训练模型进行结果的集成。

5.迭代说明
    复赛最高成绩inference时间：5 min
    复赛最高成绩训练时间：约 24 h

    迭代过程：
        从初赛到复赛结束有效提交一共6次：
        初赛：
            1. 0.7868 -- 初赛单模型CV    8月28
            2. 0.7793 -- 初赛单模型单折    8月29
        复赛：
            1. 0.7892 -- 复赛单模型单折    9月26
            2. 0.8109 -- 复赛单模型CV    9月27
          * 3. 0.8224 -- 复赛集成模型CV   9月28
            4. 0.8218 -- 微调使用数据扩增，效果不好  9月30
