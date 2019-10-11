cd src
#python pretrain.py --base_model roberta
#python finetune_cv.py --base_model roberta
#python pretrain.py --base_model wwm
#python finetune_cv.py --base_model wwm
#python pretrain.py --base_model ernie
#python finetune_cv.py --base_model ernie

# local
############## focal #################
#python pretrain.py --base_model roberta_focal
#python finetune_cv.py --base_model roberta_focal
python pretrain.py --base_model wwm_focal
python finetune_cv.py --base_model wwm_focal
python pretrain.py --base_model ernie_focal # **
python finetune_cv.py --base_model ernie_focal


# on server
###############tiny###################
python pretrain.py --base_model roberta_tiny
python finetune_cv.py --base_model roberta_tiny
python pretrain.py --base_model wwm_tiny
python finetune_cv.py --base_model wwm_tiny # **
python pretrain.py --base_model ernie_tiny # **
python finetune_cv.py --base_model ernie_tiny
