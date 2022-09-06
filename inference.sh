python3 scripts/custom_inference.py \
        --exp_dir '/root/hyperstyle/output'\
        --checkpoint_path '/root/hyperstyle/pretrained_models/hyperstyle_ffhq.pt'\
        --data_path '/root/sample/'\
        --test_batch_size 4\
        --test_workers 4\
        --n_iters_per_batch 5\
        --load_w_encoder \
        --w_encoder_checkpoint_path /root/hyperstyle/pretrained_models/faces_w_encoder.pt