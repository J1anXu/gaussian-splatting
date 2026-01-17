CUDA_VISIBLE_DEVICES=0 nohup python train.py -s /data2/jian/data/mip360/bicycle --model_path ./output/mip360/bicycle --eval   > debug/train_bicycle.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py -s /data2/jian/data/mip360/kitchen --model_path ./output/mip360/kitchen --eval   > debug/train_kitchen.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py -s /data2/jian/data/mip360/room --model_path ./output/mip360/room --eval   > debug/train_room.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python train.py -s /data2/jian/data/mip360/bonsai --model_path ./output/mip360/bonsai --eval   > debug/train_bonsai.log 2>&1 &


python render.py -m output/mip360/bicycle
python render_p.py -m output/mip360/kitchen --skip_train

python metrics.py -m output/mip360/bicycle


import torchvision
with torch.no_grad():
    vis = image.clamp(0, 1).cpu()
    torchvision.utils.save_image(vis, "re_image.png")