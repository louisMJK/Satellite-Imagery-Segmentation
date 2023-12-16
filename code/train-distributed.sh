torchrun --nproc_per_node=2 ./train.py \
    --optim adam --lr-base 5e-4 --weight-decay 1e-5 \
    --epochs 60 \


torchrun --nproc_per_node=2 ./train.py \
    --optim adam --lr-base 5e-4 --weight-decay 0 \
    --sched cosine \
    --epochs 60 \
