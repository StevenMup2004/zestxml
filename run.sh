dataset_path=$1
type=$2
# extra_args="${@:3}"
dataset=$3
extra_args="${@:4}"
#DATA_DIR=GZXML-Datasets/${dataset}
DATA_DIR=${dataset_path}
RES_DIR=${dataset_path}/Results/${dataset}
MODEL_DIR=${dataset_path}/Results/${dataset}/model
logfile=${dataset_path}/${RES_DIR}/log.txt

mkdir -p ${RES_DIR}
mkdir -p ${MODEL_DIR}

# set -x
/usr/bin/time -f "\nReal time : %E\nMax RAM : %M KB\nPercent CPU : %P" \
./run 	-trn_X_Xf ${DATA_DIR}/trn_X_Xf.txt \
		-tst_X_Xf ${DATA_DIR}/tst_X_Xf.txt \
		-Y_Yf ${DATA_DIR}/Y_Yf.txt \
		-trn_X_Y ${DATA_DIR}/trn_X_Y.txt \
		-tst_X_Y ${DATA_DIR}/tst_X_Y.txt \
		-Xf ${DATA_DIR}/Xf.txt \
		-Yf ${DATA_DIR}/Yf.txt \
		-res_dir ${RES_DIR} \
		-model_dir ${MODEL_DIR} \
		-num_thread 12 \
		-type $type \
		-propensity_A 0.55 \
		-propensity_B 1.5 \
		-bilinear_normalize 0 \
		${extra_args} \
		| tee -a ${logfile}
