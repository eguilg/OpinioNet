PRETRAINED_MODELS = {
    'roberta': {
        'name': 'roberta',
        'path': '../models/chinese_roberta_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'large',
		'focal': False
    },
    # 'wwm': {
    #     'name': 'wwm',
    #     'path': '../models/chinese_wwm_ext_pytorch',
    #     'lr': 6e-6,
		# 'version': 'large',
		# 'focal': False
    # },
    'ernie': {
        'name': 'ernie',
        'path': '../models/ERNIE',
        'lr': 8e-6,
		'version': 'large',
		'focal': False
    },
    'roberta_focal': {
        'name': 'roberta_focal',
        'path': '../models/chinese_roberta_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'large',
		'focal': True
    },
    'wwm_focal': {
        'name': 'wwm_focal',
        'path': '../models/chinese_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'large',
		'focal': True
    },
    'ernie_focal': {
        'name': 'ernie_focal',
        'path': '../models/ERNIE',
        'lr': 8e-6,
		'version': 'large',
		'focal': True
    },

    'roberta_tiny': {
        'name': 'roberta_tiny',
        'path': '../models/chinese_roberta_wwm_ext_pytorch',
        'lr': 6e-6,
		'version': 'tiny',
		'focal': True
    },
    # 'wwm_tiny': {
    #     'name': 'wwm_tiny',
    #     'path': '../models/chinese_wwm_ext_pytorch',
    #     'lr': 6e-6,
		# 'version': 'tiny',
		# 'focal': True
    # },
    'ernie_tiny': {
        'name': 'ernie_tiny',
        'path': '../models/ERNIE',
        'lr': 8e-6,
		'version': 'tiny',
		'focal': True
    }
}