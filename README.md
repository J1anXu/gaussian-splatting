CUDA_VISIBLE_DEVICES=4 nohup python train.py -s /data2/jian/data/mip360/bicycle --model_path ./output/mip360/bicycle --eval   > debug/train_bicycle.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python train.py -s /data2/jian/data/mip360/kitchen --model_path ./output/mip360/kitchen --eval   > debug/train_kitchen.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python train.py -s /data2/jian/data/mip360/room --model_path ./output/mip360/room --eval   > debug/train_room.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python train.py -s /data2/jian/data/mip360/bonsai --model_path ./output/mip360/bonsai --eval   > debug/train_bonsai.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python render_p.py -m output/mip360/bicycle --skip_train > debug/render_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python render_p.py -m output/mip360/kitchen --skip_train > debug/render_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python render_p.py -m output/mip360/room --skip_train > debug/render_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python render_p.py -m output/mip360/bonsai --skip_train > debug/render_bonsai.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 python metrics_p.py -m output/mip360/bicycle > debug/metrics_bicycle.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python metrics_p.py -m output/mip360/kitchen > debug/metrics_kitchen.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python metrics_p.py -m output/mip360/room > debug/metrics_room.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 ython metrics_p.py -m output/mip360/bonsai > debug/metrics_bonsai.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python render.py -m output/mip360/bicycle --skip_train > debug/render_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python render.py -m output/mip360/kitchen --skip_train > debug/render_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python render.py -m output/mip360/room --skip_train > debug/render_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python render.py -m output/mip360/bonsai --skip_train > debug/render_bonsai.log 2>&1 &



CUDA_VISIBLE_DEVICES=4 python metrics.py -m output/mip360/bicycle > debug/metrics_bicycle.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python metrics.py -m output/mip360/kitchen > debug/metrics_kitchen.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python metrics.py -m output/mip360/room > debug/metrics_room.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 ython metrics.py -m output/mip360/bonsai > debug/metrics_bonsai.log 2>&1 &

nohup bash run_all.sh > debug/run_all.out 2>&1 &