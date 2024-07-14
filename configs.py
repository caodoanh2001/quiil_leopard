configs = {
    'online': {
        'source_path': '/input/images/prostatectomy-wsi/', # Directory includes *.tif files
        'patch_path': '/workspace/Leopard_patches', # Directory that the patches would be saved
        'patch_path_advmil': '/workspace/Leopard_patches_advmil_l1', # Directory that the patches would be saved
        'feat_path': '/workspace/Leopard_feats_uni', # Directory that features would be saved
        'patch_size': 512, # Image size
        'step_size': 512,
        'patch_level': 1, # Magnification
        'prediction_path': '/output/overall-survival-years.json', # Prediction output
        'feat_type': 'uni'
    }, 
    'offline': {
        'source_path': '/data6/leopard/offline_datasets/Leopard', # Directory includes *.tif files
        'patch_path': '/data6/leopard/offline_datasets/Leopard_patches', # Directory that the patches would be saved
        'patch_path_advmil': '/data6/leopard/offline_datasets/Leopard_patches_advmil_l1', # Directory that the patches would be saved
        'feat_path': '/data6/leopard/offline_datasets/Leopard_feats_uni', # Directory that features would be saved
        'patch_size': 512, # Image size
        'patch_level': 1, # Magnification
        'prediction_path': '/output/overall-survival-years.json', # Prediction output
        'feat_type': 'uni'
    }
}
