PRETRAINED_MODELS = {
    'roberta': {
        'name': 'roberta',
        'path': '../models/chinese_roberta_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'large'
    },
    'wwm': {
        'name': 'wwm',
        'path': '../models/chinese_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'large'
    },
    'ernie': {
        'name': 'ernie',
        'path': '../models/ERNIE',
        'lr': 8e-6,
		'version': 'large'
    },
    'roberta_tiny': {
        'name': 'roberta_tiny',
        'path': '../models/chinese_roberta_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'tiny'
    },
    'wwm_tiny': {
        'name': 'wwm_tiny',
        'path': '../models/chinese_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'tiny'
    },
    'ernie_tiny': {
        'name': 'ernie_tiny',
        'path': '../models/ERNIE',
        'lr': 8e-6,
		'version': 'tiny'
    }
}