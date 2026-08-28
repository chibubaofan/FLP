python detection_performance.py --num_agents 800 \
        --k_personas 4 \
        --persona_type heterogeneous \
        --attack_image ../data/attack_image/border8.png \
        --vlm ../models/llava-1.5-7b-hf \
        --clip ../models/clip-vit-large-patch14 \
        --agent_data ../data/million_villagers_1024.json \
        --album_data ../data/album_pool/ \
        --th_diversity  0.1455 \
        --th_entropy 0.0 