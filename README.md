python train.py -s /data2/jian/data/mip360/bicycle --model_path ./output/mip360/bicycle --eval
python render.py -m output/mip360/bicycle --eval
python render_p.py -m output/mip360/bicycle

CUDA_VISIBLE_DEVICES=0 nohup python train.py -s /data2/jian/data/mip360/bicycle --model_path ./output/mip360/bicycle --eval   > debug/train0.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python train.py -s /data2/jian/data/mip360/bonsai --model_path ./output/mip360/bonsai --eval   > debug/train_bonsai.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python train.py -s /data2/jian/data/mip360/kitchen --model_path ./output/mip360/kitchen --eval   > debug/train_kitchen.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python train.py -s /data2/jian/data/mip360/room --model_path ./output/mip360/room --eval   > debug/train_room.log 2>&1 &
# 不传则和之前一样，自动检测全部 GPU      
bash run_mip360.sh --type outdoor --gpus 0,1,2,3,4 只训练室外 5 场景
bash run_mip360.sh --type all --gpus 0,1,2,3 训练全部 9 场景  
bash run_mip360.sh --type indoor --gpus 0,1,2,3  只训练室内 4 场景



python train.py -s /data2/jian/data/cl-splats-dataset/Blender-Levels/Level-1 --model_path ./output/cl-splats-dataset/Blender-Levels/Level-1 --eval
  python render.py -m output/cl-splats-dataset/Blender-Levels/Level-1 --skip_test                                                                                                        
