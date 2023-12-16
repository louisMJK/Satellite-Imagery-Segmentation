torchrun --nproc_per_node=2 ./train.py \
    --optim sgd --lr-base 1e-4 --momentum 0.9 --weight-decay 1e-4 \
    --sched cosine \
    --epochs 40 \
    --model-checkpoint ../checkpoints/model_jac_0.4604.pth


torchrun --nproc_per_node=2 ./train.py \
    --optim adam --lr-base 1e-4 --weight-decay 1e-4 \
    --sched cosine \
    --epochs 40 \
    --model-checkpoint ../checkpoints/model_jac_0.4604.pth


torchrun --nproc_per_node=2 ./train.py \
    --optim sgd --lr-base 1e-4 --momentum 0.9 --weight-decay 0 \
    --sched poly \
    --epochs 40 \
    --model-checkpoint ../checkpoints/model_jac_0.4604.pth


# torchrun --nproc_per_node=2 ./train.py \
#     --optim adam --lr-base 5e-4 --weight-decay 0 \
#     --sched cosine \
#     --epochs 60 \
# --model-checkpoint ../checkpoints/model_jac-0.43297.pth
