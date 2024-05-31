CUDA_VISIBLE_DEVICES=0 \
python3.8 extract_features_fp.py \
--data_h5_dir /data4/doanhbc/camelyon_patches_20x/ \
--data_slide_dir /data3/anhnguyen/CAMELYON/ \
--csv_path ./camelyon16.csv \
--feat_dir /data4/doanhbc/camelyon_patches_20x/features_2/ \
--batch_size 512 \
--slide_ext .tif