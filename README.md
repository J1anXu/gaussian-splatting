python train.py -s /data2/jian/data/mip360/bicycle --model_path ./output/mip360/bicycle --eval
python render.py -m output/mip360/bicycle --eval
python render_p.py -m output/mip360/bicycle

CUDA_VISIBLE_DEVICES=0 nohup python train.py -s /data2/jian/data/mip360/bicycle --model_path ./output/mip360/bicycle --eval   > debug/train0.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python train.py -s /data2/jian/data/mip360/bonsai --model_path ./output/mip360/bonsai --eval   > debug/train_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python train.py -s /data2/jian/data/mip360/kitchen --model_path ./output/mip360/kitchen --eval   > debug/train_kitchen.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python train.py -s /data2/jian/data/mip360/room --model_path ./output/mip360/room --eval   > debug/train_room.log 2>&1 &
