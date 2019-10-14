cd src
python eval_ensemble_final.py --gen_label \
                              --bs 64 \
                              --rv ../data/TRAIN/Train_laptop_corpus.csv \
                              --o ../data/TRAIN/Train_laptop_corpus_labels0.csv \
                              --skipfold 0

python eval_ensemble_final.py --gen_label \
                              --bs 64 \
                              --rv ../data/TRAIN/Train_laptop_corpus.csv \
                              --o ../data/TRAIN/Train_laptop_corpus_labels1.csv \
                              --skipfold 1

python eval_ensemble_final.py --gen_label \
                              --bs 64 \
                              --rv ../data/TRAIN/Train_laptop_corpus.csv \
                              --o ../data/TRAIN/Train_laptop_corpus_labels2.csv \
                              --skipfold 2

python eval_ensemble_final.py --gen_label \
                              --bs 64 \
                              --rv ../data/TRAIN/Train_laptop_corpus.csv \
                              --o ../data/TRAIN/Train_laptop_corpus_labels3.csv \
                              --skipfold 3

python eval_ensemble_final.py --gen_label \
                              --bs 64 \
                              --rv ../data/TRAIN/Train_laptop_corpus.csv \
                              --o ../data/TRAIN/Train_laptop_corpus_labels4.csv \
                              --skipfold 4