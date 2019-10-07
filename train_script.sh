cd src
python pretrain.py --base_model roberta
python finetune_cv.py --base_model roberta
python pretrain.py --base_model wwm
python finetune_cv.py --base_model wwm
python pretrain.py --base_model ernie
python finetune_cv.py --base_model ernie