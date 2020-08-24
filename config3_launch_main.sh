nbtcr=$1
prefixdir='c3_hlabin_first'
lr='1e-3'
pt=$2
gpu=$3
savedir=`echo results/${nbtcr}_TCR/${pt}_${nbtcr}_${prefixdir}`
python main.py --data-dir ~/Emerson/processed_data --data-file binary_index_list.npy --dataset binary_hla_tcr --model allseq_bin  --emb_size 10 --save-dir $savedir --epoch 500 --gpu-selection $gpu --nb-tcr-to-sample $nbtcr --cache $nbtcr --lr $lr  --nb-patient $pt --tcr-conv-layers-sizes 20 20 5 20 20 5 20 1 10 --hla-conv-layers-sizes 20 20 3 20 20 9 20 1 15 #--mlp-layers-size 750 500 250 75 50 25 10


