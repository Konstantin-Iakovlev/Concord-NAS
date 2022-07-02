"""
4 domains
1. Generate random DARTS architectures and retrain on a mixed datasets
2. Train MdDARTS on 4 domains
3. Train DARTS on mixed domains
"""
import os
from configobj import ConfigObj

os.environ['PYTHONPATH'] = '.'
os.mkdir('searchs')
os.mkdir('retrain')
os.system('pip install requirements.txt')

ds_list = ['MNIST-0', 'MNIST-90', 'MNIST-180', 'MNIST-270']

# random architectures on a mixed dataset
for seed in range(10):
    # generate random architectures
    os.system(f'python scripts/random_architecture.py --seed={seed} --save-dir=searchs/random_mnist_arc')
    # retrain random architectures
    cfg = ConfigObj('configs/md_mnist_basic/darts_retrain.cfg')
    cfg.filename = 'current_config.cfg'
    cfg['datasets'] = '+'.join(ds_list)
    cfg['architecture_path'] = f'searchs/random_mnist_arc/architecture_{seed}.json'
    cfg['folder_name'] = f'random_{seed}'
    cfg['seed'] = seed
    cfg.write()
    os.system(f'python scripts/retrain_md_darts.py --config current_config.cfg')
    os.system(f'mv current_config.cfg retrain/random_{seed}/config.cfg')

# md darts and darts
for exp_name, delimiter in zip(['md_darts', 'darts'], [';', '+']):
    for seed in range(3):
        # search
        cfg = ConfigObj('configs/md_mnist_basic/darts.cfg')
        cfg.filename = 'current_config.cfg'
        cfg['datasets'] = delimiter.join(ds_list)
        cfg['seed'] = seed
        cfg['folder_name'] = f'{exp_name}_{seed}'
        cfg.write()
        os.system(f'python scripts/train_md_darts.py --config current_config.cfg')
        os.system(f'mv current_config.cfg searchs/{exp_name}_{seed}/config.cfg')
        # retrain
        cfg = ConfigObj('config/md_mnist_basic/darts_retrain.cfg')
        cfg.filename = 'current_config.cfg'
        cfg['datasets'] = delimiter.join(ds_list)
        cfg['seed'] = seed
        cfg['folder_name'] = f'{exp_name}_{seed}'
        cfg['architecture_path'] = f'searchs/{exp_name}_{seed}/final_architecture.json'
        cfg.write()
        os.system(f'python scripts/retrain_md_darts.py --config current_config.cfg')
        os.system(f'mv current_config.cfg retrain/{exp_name}_{seed}/config.cfg')
