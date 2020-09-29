fdir=$1
fsave=$2

./get_predictions_likelihood.sh $fdir 0      
./get_same_predictions_likelihood.sh $fdir 0
./get_Thome_predictions_likelihood.sh $fdir 0
tar -cvf ${fsave}.tar.gz ${fdir}/*tenth*
