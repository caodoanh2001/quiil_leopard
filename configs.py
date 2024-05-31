configs = {
    'online': {
        'source_path': '/input/', # Directory includes *.tif files
        'patch_path': '/workspace/Leopard_patches', # Directory that the patches would be saved
        'patch_path_advmil': '/workspace/Leopard_patches_advmil_l1', # Directory that the patches would be saved
        'feat_path': '/workspace/Leopard_feats', # Directory that features would be saved
        'patch_size': 256, # Image size
        'patch_level': 2, # Magnification
        'prediction_path': '/output/overall-survival-years.json' # Prediction output
    }, 
    'offline': {
        'source_path': '/data6/leopard/offline_datasets/Leopard', # Directory includes *.tif files
        'patch_path': '/data6/leopard/offline_datasets/Leopard_patches', # Directory that the patches would be saved
        'patch_path_advmil': '/data6/leopard/offline_datasets/Leopard_patches_advmil_l1', # Directory that the patches would be saved
        'feat_path': '/data6/leopard/offline_datasets/Leopard_feats', # Directory that features would be saved
        'patch_size': 256, # Image size
        'patch_level': 2, # Magnification
        'prediction_path': 'prediction.csv' # Prediction output
    }
}