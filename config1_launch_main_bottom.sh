nbtcr=$1
prefixdir='c1_hlabin_first'
lr='1e-3'
pt=$2
gpu=$3
savedir=`echo results/${nbtcr}_TCR/bottom_${pt}_${nbtcr}_${prefixdir}_${lr}`
python main.py --data-dir ~/Emerson/processed_data --data-file binary_index_list.npy --dataset binary_hla_tcr --model allseq_bin  --emb_size 10 --save-dir $savedir --epoch 500 --gpu-selection $gpu --nb-tcr-to-sample $nbtcr --cache bottom_${nbtcr} --lr $lr  --nb-patient $pt --mlp-layers-size 750 500 250 75 50 25 10  


