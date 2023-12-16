torchrun --nproc_per_node=2 ./train.py \
    --optim adam --lr-base 5e-4 --weight-decay 1e-5 \
    --sched cosine \
    --epochs 100 \
    

# torchrun --nproc_per_node=2 ./train.py \
#     --optim adam --lr-base 5e-4 --weight-decay 0 \
#     --sched cosine \
#     --epochs 60 \
# --model-checkpoint ../checkpoints/model_jac-0.43297.pth
