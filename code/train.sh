torchrun ./train.py \
    --optim sgd --lr-base 1e-2 --momentum 0.9 --weight-decay 5e-4 \
    --epochs 10 \
    -b 32


# torchrun ./train.py \
#     --optim sgd --lr-base 1e-2 --momentum 0.9 --weight-decay 0 \
#     --epochs 10 \
#     -b 64


# torchrun ./train.py \
#     --optim sgd --lr-base 1e-3 --momentum 0.9 --weight-decay 0 \
#     --epochs 10 \
#     -b 64


