
export PATH="/home/tangyinzhou/miniforge3/envs/DisGrounding/bin:$PATH"

export CUDA_VISIBLE_DEVICES=2

cd /data5/tangyinzhou/DisGrounding/algos/finetune_clip

python finetune_clip.py
