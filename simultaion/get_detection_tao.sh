python get_detection_tao.py --calibration_data ../data/million_villagers_1024_test.json \
    --album_data ../data/album_pool/ \
    --num_calibration_agents 800\
    --tolerance_fpr 7.5 \
    --vlm ../models/llava-1.5-7b-hf \
    --clip ../models/clip-vit-large-patch14 \
    --k_personas 4 \
    --persona_type heterogeneous \
    --batch_size 8 \
    --album_length 10