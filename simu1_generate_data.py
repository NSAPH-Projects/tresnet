import os
import torch
import pickle

from data.simu1_si import simu_data1
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate simulate data')
    parser.add_argument('--save_dir', type=str, default='dataset/simu1', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=100, help='num of dataset for evaluating the methods')
    parser.add_argument('--num_tune', type=int, default=20, help='num of dataset for tuning the parameters')

    args = parser.parse_args()
    save_path = args.save_dir

    delta_vals = torch.linspace(0, 0.5, steps=10)  # standard deviation reductions from observed value

    for _ in range(args.num_tune):
        print('generating tuning set: ', _)
        data_path = os.path.join(save_path, 'tune', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        D = simu_data1(500, 200, delta_vals=delta_vals)

        data_file = os.path.join(data_path, 'sim.pkl')
        with open(data_file, 'wb') as io:
            pickle.dump(D, io, protocol=pickle.HIGHEST_PROTOCOL)

    for _ in range(args.num_eval):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        D = simu_data1(500, 200, delta_stds=delta_stds)

        data_file = os.path.join(data_path, 'sim.pkl')
        with open(data_file, 'wb') as io:
            pickle.dump(D, io, protocol=pickle.HIGHEST_PROTOCOL)