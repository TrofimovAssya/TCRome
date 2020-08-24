nbtcr=$1
prefixdir='c2_hlabin_first'
lr='1e-3'
pt=$2
gpu=$3
savedir=`echo results/${nbtcr}_TCR/${pt}_${nbtcr}_${prefixdir}_${lr}`
python main.py --data-dir ~/Emerson/processed_data --data-file binary_index_list.npy --dataset binary_hla_tcr --model allseq_bin  --emb_size 10 --save-dir $savedir --epoch 500 --gpu-selection $gpu --nb-tcr-to-sample $nbtcr --cache $nbtcr --lr $lr  --nb-patient $pt --tcr-conv-layers-sizes 20 20 10 20 1 9 --hla-conv-layers-sizes 20 20 15 20 1 11


