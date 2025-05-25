import torch, random, argparse, os
import copy
import numpy as np
# from algorithms.engine.fedavg import *
# from algorithms.engine.byzantine_robust_fedavg import byzantine_robust_fedavg
from algorithms.engine.attack_defense_fedavg import attack_defense_fedavg
from mmengine.config import Config

def merge_config(config, args):
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="seed")
    parser.add_argument('--repeat', type=int, default=1, help='repeat index')
    parser.add_argument('--freeze_datasplit', type=int, default=0, help='freeze to save dict_users.pik or not')
    parser.add_argument('--config_name', type=str,
                        # default='attack/electricity_theft_detection/exp_noattack/electricity_theft_detection.yaml',
                        default='attack/generator_defect_classification/exp_noattack/image_defectdetection_dpfed_NM14.yaml',
                        help='configuration')

    meta_args = parser.parse_args()
    config_path = os.path.join('config/', meta_args.config_name)
    config = Config.fromfile(config_path)
    meta_args = merge_config(config, meta_args)

    meta_args.device = torch.device('cuda:{}'.format(meta_args.gpu) if torch.cuda.is_available() and meta_args.gpu != -1 else 'cpu')
    # for reproducibility
    score_box = []
    poisoned_ratio_box = []
    for r in range(meta_args.repeat):
        args = copy.deepcopy(meta_args)
        print('############ Case '+ str(r) + ' ############')
        random.seed(args.seed+r)
        torch.manual_seed(args.seed+r)
        # torch.cuda.manual_seed(args.seed+args.repeat) # avoid
        np.random.seed(args.seed+r)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # if args.method == 'fedavg':
        #     best_result = fedavg(args)
        # elif args.method == 'byzantine_robust_fedavg':
        #     best_result = byzantine_robust_fedavg(args)
        # el
        if args.method == "attack_defense_fedavg":
            best_result, poisoned_ratio = attack_defense_fedavg(args)
        score_box.append(best_result)
        poisoned_ratio_box.append(poisoned_ratio)
    print('repeated scores: ' + str(score_box))
    avg_score = np.average(score_box)
    print('avg of the scores ' + str(avg_score))
    
    print('repeated poisoned ratio: ' + str(poisoned_ratio_box))
    avg_poisoned_ratio = np.average(poisoned_ratio_box)
    print('avg of the poisoned ratios ' + str(avg_poisoned_ratio))

    # String to write
    my_string = 'config name is: ' + str(meta_args.config_name) + ', ' +\
                'repeated scores: ' + str(score_box) + ', ' +\
                'avg of the scores ' + str(avg_score) + ', ' +\
                'repeated poisoned ratios ' + str(poisoned_ratio_box) + ', ' +\
                'avg of the poisoned ratios ' + str(avg_poisoned_ratio) + ', ' +\
                '\n'

    # Open the file in write mode
    with open("auto_run.txt", "a") as file:
        # Write the string to the file
        file.write(my_string)
