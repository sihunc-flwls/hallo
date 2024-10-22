# ffmpeg -i .cache/test_02.mp4 -i .cache/test_03.mp4 -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 tmp.mp4

# ffmpeg -i tmp.mp4 -i .cache/test_05.mp4 -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 tmp2.mp4

# ffmpeg -i tmp2.mp4 -i .cache/test_06.mp4 -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 tmp.mp4

# ffmpeg -i tmp2.mp4 -i .cache/test_07.mp4 -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 tmp3.mp4


# ffmpeg -i ".cache/aniportrait-20241022/0000--seed_42-512x512/Screenshot 2024-10-17 at 12.09.34 PM_1_512x512_3_0000.mp4" -i ".cache/aniportrait-20241022/0000--seed_42-512x512/Screenshot 2024-10-17 at 12.09.53 PM_1_512x512_3_0000.mp4" -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 tmp4.mp4

# ffmpeg -i tmp4.mp4 -i ".cache/aniportrait-20241022/0000--seed_42-512x512/Screenshot 2024-10-17 at 1.38.56 PM_1_512x512_3_0000.mp4" -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 tmp_.mp4

# ffmpeg -i tmp_.mp4 -i ".cache/aniportrait-20241022/0000--seed_42-512x512/Screenshot 2024-10-17 at 1.48.52 PM_1_512x512_3_0000.mp4" -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 tmp4.mp4



ffmpeg -i ".cache/aniportrait-20241022/0000--seed_42-512x512/Screenshot 2024-10-17 at 12.08.54 PM_1_512x512_3_0000.mp4" -i ".cache/aniportrait-20241022/0000--seed_42-512x512/Screenshot 2024-10-17 at 12.10.43 PM_1_512x512_3_0000.mp4" -filter_complex \
"[0:v][1:v]hstack=inputs=2[v]; \
 [0:a][1:a]amerge[a]" \
-map "[v]" -map "[a]" -ac 2 tmp_.mp4

ffmpeg -i tmp_.mp4 -i ".cache/aniportrait-20241022/0000--seed_42-512x512/Screenshot 2024-10-21 at 12.03.48 PM_1_512x512_3_0000.mp4" -filter_complex \
"[0:v][1:v]hstack=inputs=2[v]; \
 [0:a][1:a]amerge[a]" \
-map "[v]" -map "[a]" -ac 2 tmp5.mp4
