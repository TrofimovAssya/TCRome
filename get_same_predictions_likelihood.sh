folder=$1
for t in {0..9}; do echo $t;  python predict_likelihood_same.py --data-dir ~/Emerson/processed_data --data-file binary_index_list.npy --dataset binary_hla_tcr --model allseq_bin  --emb_size 10 --save-dir $folder --epoch 559 --gpu-selection 1 --nb-tcr-to-sample 200 --cache 200 --lr 1e-3  --nb-patient 500 --mlp-layers-size 750 500 250 75 50 25 10 --load-folder $folder --tenth $t; done
