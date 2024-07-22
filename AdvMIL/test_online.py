"""
Entry filename: main.py

Code in this file inherts from https://github.com/hugochan/IDGL as it provides a flexible way 
to configure parameters and inspect model performance. Great thanks to the author.
"""
import argparse
import yaml
import numpy as np
from collections import defaultdict, OrderedDict

from model import MyHandler
from model import BaselineHandler
from utils.func import print_config

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import configs as global_configs
import json

ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
mode = 'online'

def main(handler, config):
    model = handler(config)
    if config['semi_training']:
        metrics = model.exec_semi_sl()
    elif config['test']:
        metrics = model.exec_test()
    else:
        metrics = model.exec()
    print('[INFO] Metrics:', metrics)

def multi_run_main(handler, _config):
    # fold 1: right-left
    # fold 4: bottom-top

    models = []

    for i in range(5):
        if i == 1:
            config = _config["right-left"]
            hyperparams = []
            for k, v in config.items():
                if isinstance(v, list):
                    hyperparams.append(k)

            configs = grid(config)
            for k in hyperparams:
                configs[1]['save_path'] += '-{}_{}'.format(k, configs[1][k])
            model = handler(configs[1])
        
        elif i == 4:
            config = _config["bottom-top"]
            hyperparams = []
            for k, v in config.items():
                if isinstance(v, list):
                    hyperparams.append(k)

            configs = grid(config)
            for k in hyperparams:
                configs[4]['save_path'] += '-{}_{}'.format(k, configs[4][k])
            model = handler(configs[4])
        
        else:
            config = _config["base"]
            hyperparams = []
            for k, v in config.items():
                if isinstance(v, list):
                    hyperparams.append(k)

            configs = grid(config)
            for k in hyperparams:
                configs[i]['save_path'] += '-{}_{}'.format(k, configs[i][k])
            model = handler(configs[i])
        
        models.append(model)

    overlaps = [None, 'right-left', None, None, 'bottom-top']
    ensemble_output = 0.
    for i in range(5):
        slide_ids, results = models[i].exec_test_online(overlap=overlaps[i])
        ensemble_output += results[-1]

    # Prediction
    result = ensemble_output / len(configs)
    output_path = global_configs.configs[mode]['prediction_path']
    json.dump(float(result), open(output_path, 'w'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, help='path to the config file')
    parser.add_argument('--handler', '-d', required=True, type=str, help='model handler (adv or base)')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args

def get_config(config_path="config/config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


if __name__ == '__main__':
    cfg = get_args()
    config_base = get_config("/workspace/AdvMIL/config/leopard_uni.yaml")
    config_rightleft = get_config("/workspace/AdvMIL/config/leopard_uni_rightleft.yaml")
    config_bottomtop = get_config("/workspace/AdvMIL/config/leopard_uni_bottomtop.yaml")
    # print_config(config)
    if cfg['handler'] == 'adv':
        handler = MyHandler
    elif cfg['handler'] == 'base':
        handler = BaselineHandler
    else:
        handler = None
    if cfg['multi_run']:
        multi_run_main(handler, {"base": config_base, "right-left": config_rightleft, "bottom-top": config_bottomtop})
    else:
        main(handler, {"right-left": config_rightleft, "bottom-top": config_bottomtop})
