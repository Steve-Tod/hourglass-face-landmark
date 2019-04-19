import os, json


def parse_opt(opt_path):
    with open(opt_path, 'r') as f:
        opt = json.load(f)

    # path
    exp_name = opt['name']
    opt['path']['exp_root'] = os.path.join(opt['path']['root_path'],
                                           'experiments', exp_name)
    assert os.path.exists(opt['path']['exp_root']) == opt['resume'] # make sure no conflict
    opt['path']['checkpoint_dir'] = os.path.join(opt['path']['exp_root'],
                                                 'checkpoint')
    opt['path']['visual_dir'] = os.path.join(opt['path']['exp_root'],
                                             'visualization')
    opt['path']['tb_logger_root'] = opt['path']['exp_root'].replace(
        'experiments', 'tb_logger')
    for k, v in opt['path'].items():
        if k == 'root_path':
            continue
        elif k == 'tb_logger_root':
            if opt['use_tb_logger']:
                os.makedirs(v)
        else:
            os.makedirs(v)
    # dataset
    for k, v in opt['datasets'].items():
        opt['datasets'][k]['phase'] = k
        opt['datasets'][k]['num_keypoints'] = opt['num_keypoints']

    # network
    opt['networks']['hourglass']['num_keypoints'] = opt['num_keypoints']
    opt['networks']['ghcu']['output_dim'] = 2 * opt['num_keypoints']
    opt['networks']['ghcu']['in_channel'] = opt['num_keypoints']
    with open(os.path.join(opt['path']['exp_root'], 'opt.json'), 'w') as f:
        json.dump(opt, f, indent=4)
    return opt