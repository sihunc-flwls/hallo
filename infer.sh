## hallo
# python scripts/inference_hallo.py \
#     --config configs/inference/default_emo.yaml \
#     --source_image "examples/reference_images/Screenshot 2024-10-17 at 12.08.54 PM.png" \
#     --driving_audio "examples/driving_audios/1.wav" \
#     --output .cache/test_01.mp4

# python scripts/inference_hallo.py \
#     --config configs/inference/default_emo.yaml \
#     --source_image "examples/reference_images/Screenshot 2024-10-17 at 12.09.34 PM.png" \
#     --driving_audio "examples/driving_audios/1.wav" \
#     --output .cache/test_02.mp4

# python scripts/inference_hallo.py \
#     --config configs/inference/default_emo.yaml \
#     --source_image "examples/reference_images/Screenshot 2024-10-17 at 12.09.53 PM.png" \
#     --driving_audio "examples/driving_audios/1.wav" \
#     --output .cache/test_03.mp4

# python scripts/inference_hallo.py \
#     --config configs/inference/default_emo.yaml \
#     --source_image "examples/reference_images/Screenshot 2024-10-17 at 12.10.43 PM.png" \
#     --driving_audio "examples/driving_audios/1.wav" \
#     --output .cache/test_04.mp4

# python scripts/inference_hallo.py \
#     --config configs/inference/default.yaml \
#     --source_image "examples/reference_images/Screenshot 2024-10-17 at 1.38.56 PM.png" \
#     --driving_audio "examples/driving_audios/1.wav" \
#     --output .cache/test_05.mp4

# python scripts/inference_hallo.py \
#     --config configs/inference/default.yaml \
#     --source_image "examples/reference_images/Screenshot 2024-10-17 at 1.48.52 PM.png" \
#     --driving_audio "examples/driving_audios/1.wav" \
#     --output .cache/test_06.mp4

# python scripts/inference_hallo.py \
#     --config configs/inference/default.yaml \
#     --source_image "examples/reference_images/Screenshot 2024-10-21 at 12.03.48 PM.png" \
#     --driving_audio "examples/driving_audios/1.wav" \
#     --output .cache/test_07.mp4


## aniportrait
python scripts/inference_aniportrait_a2v.py \
    --config ./src/aniportrait/configs/prompts/animation_audio.yaml \
    -W 512 -H 512 -acc