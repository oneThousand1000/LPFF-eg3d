python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --conditional_camera_sample_mode=avg --camera_sample_mode=FFHQ
python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --conditional_camera_sample_mode=avg --camera_sample_mode=LPFF

python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --conditional_camera_sample_mode=FFHQ --camera_sample_mode=FFHQ
python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --conditional_camera_sample_mode=FFHQ --camera_sample_mode=LPFF
python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --conditional_camera_sample_mode=LPFF --camera_sample_mode=FFHQ
python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --conditional_camera_sample_mode=LPFF --camera_sample_mode=LPFF

python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --camera_sample_mode=FFHQ
python calc_metrics.py  --metrics=fid50k_full  --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --network=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl  --gpus=1  --camera_sample_mode=LPFF



python train.py --outdir=./training-runs/var1-64   --cfg=ffhq --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=True  --camera_sample_mode=FFHQ_LPFF --resume=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl

python train.py --outdir=./training-runs/var1-128  --cfg=ffhq --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip   --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128 --kimg=20000  --resume=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl   --camera_sample_mode=FFHQ_LPFF

python train.py --outdir=./training-runs/var2-64  --cfg=ffhq --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=True  --gpc_reg_prob=0.8 --kimg=20000   --resume=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl --camera_sample_mode=FFHQ_LPFF_rebalanced



python train.py --outdir=./training-runs/var3-64  --cfg=ffhq --data=E:\data\large_pose_face/LPFF_FFHQ_eg3d.zip  --gpus=1 --batch=4 --gamma=1   --gen_pose_cond=False --gen_pose_cond_avg=True   --kimg=20000    --resume=F:\cvpr2023\LPFF-dataset\networks\eg3d/var1-64.pkl   --camera_sample_mode=FFHQ_LPFF