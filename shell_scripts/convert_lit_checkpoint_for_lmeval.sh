CKPT_NAME=step-00002000.pth
CKPT_PATH="out/lit-phi-2/${CKPT_NAME}"
OUTPUT_PATH="./out/lit-phi-2/hf_${CKPT_NAME}"

# python scripts/convert_lit_checkpoint.py \
    # --checkpoint_path $CKPT_PATH \
    # --output_path $OUTPUT_PATH \
    # --config_path ./checkpoints/microsoft/phi-2/lit_config.json


HF_DIR="../hf-checkpoints"
HF_URL="https://huggingface.co/microsoft/phi-2"


mkdir -p $HF_DIR
cp $OUTPUT_PATH $HF_DIR

cd $HF_DIR
hf_model_dir=$(echo $HF_URL | awk -F/ '{print $NF}')

if [ -d $hf_model_dir ]; then
  echo "Directory for ${HF_URL} exists. Skipping git clone"
else
  git clone $HF_URL
fi 

mv "hf_${CKPT_NAME}" $hf_model_dir


