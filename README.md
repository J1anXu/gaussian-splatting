CUDA_VISIBLE_DEVICES=0 nohup python train.py -s /data2/jian/data/mip360/bicycle --model_path ./output/mip360/bicycle --eval   > debug/train0.log 2>&1 &
python render.py -m output/mip360/bicycle --eval --iteration 30000
python metrics.py -m output/mip360/bicycle


import torchvision
with torch.no_grad():
    vis = image.clamp(0, 1).cpu()
    torchvision.utils.save_image(vis, "re_image.png")