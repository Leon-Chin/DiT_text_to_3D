python "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/generate.py" \
    --checkpoint "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/checkpoints/checkpoint.pth" \
    --output "generated_samples.npy" \
    --voxel_size 32 \
    --beta_start 1e-5 \
    --beta_end 0.008 \
    --time_num 1000 \
    --num_steps 50 \
    --cfg_scale 3.0 \
    --batch_size 1 \