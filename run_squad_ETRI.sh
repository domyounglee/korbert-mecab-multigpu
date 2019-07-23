#! /bin/bash
BERT_BASE_DIR="/root/workspace"
OUTPUT_DIR="/data2/bert_record/output"
python run_squad_ETRI_multigpu.py \
  --vocab_file=$BERT_BASE_DIR/ETRI_morp_TF/vocab.korean_morp.list \
  --bert_config_file=$BERT_BASE_DIR/ETRI_morp_TF/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/ETRI_morp_TF/model.ckpt \
  --do_train=True \
  --train_file=$BERT_BASE_DIR/eval/koquad/KorQuAD_v1.0_train.json \
  --do_predict=True \
  --predict_file=$BERT_BASE_DIR/eval/koquad/KorQuAD_v1.0_dev.json\
  --do_lower_case=False \
  --train_batch_size=8\
  --learning_rate=8.5e-5 \
  --num_train_epochs=1.5 \
  --save_checkpoints_steps=1000 \
  --max_seq_length=384 \
  --max_query_length=128 \
  --doc_stride=128 \
  --output_dir=$BERT_BASE_DIR/asdf \
  --use_tpu=False \ 
