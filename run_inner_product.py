import copy
import jobs
import collections
import pathlib
import addict

def get_slurm_script_gpu(train_dir, command):
  """Returns contents of SLURM script for a gpu job."""
  return """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:tesla_p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --output={}/slurm_%j.out
#SBATCH -t 05:59:00
#module load anaconda3 cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1
#source activate yumi
{}
""".format(train_dir, command)


if __name__ == '__main__':
    commands = ["PYTHONPATH=. python inner-product.py  --train_path /scratch/gpfs/altosaar/dat/longform-data/mapped-data/train.json --test_path /scratch/gpfs/altosaar/dat/longform-data/mapped-data/test.json --eval_path /scratch/gpfs/altosaar/dat/longform-data/mapped-data/evaluation.json "]

    experiment_name = 'news-inner-product'
    log_dir = pathlib.Path("hello") / 'news-classification'

    grid = addict.Dict()
    grid.create_dicts = False
    grid.map_items = False
    grid.emb_size = 100
    grid.tokenize = False
    grid.target_publication = 0
    grid.batch_size = 2000
    grid.momentum = [0.9]
    grid.use_sparse = False
    grid.use_gpu = True
    #  grid.inner_product_checkpoint = '/scratch/gpfs/altosaar/log/food_rec/2019-05-22/adagrad_rmsprop_adam_sgd_lr_decay/model=InnerProduct_optim=sgd_learning_rate=3.0_batch_size=1024/best_state_dict'

    #RMS with all words
    grid = copy.deepcopy(grid)
    grid['optimizer_type'] = "RMS"
    grid['use_all_words'] = True
    grid['learning_rate'] = [1e-1, 1e-3, 1e-4, 1e-5]
    grid['word_embedding_type'] = ['sum', 'mean']
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    for cfg in jobs.param_grid(grid):
        cfg['train_dir'] = jobs.make_train_dir(log_dir, experiment_name, cfg, keys_for_dir_name)
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    #RMS with only unique from first 500 words
    grid = copy.deepcopy(grid)
    grid['optimizer_type'] = "RMS"
    grid['use_all_words'] = False
    grid['words_to_use'] = 500
    grid['learning_rate'] = [1e-1, 1e-3, 1e-4, 1e-5]
    grid['word_embedding_type'] = ['sum']
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    for cfg in jobs.param_grid(grid):
        cfg['train_dir'] = jobs.make_train_dir(log_dir, experiment_name, cfg, keys_for_dir_name)
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    #SGD with all words
    grid = copy.deepcopy(grid)
    grid['optimizer_type'] = "SGD"
    grid['use_all_words'] = True
    grid['learning_rate'] = [0.1,1,5,10,15]
    grid['word_embedding_type'] = 'sum'
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    for cfg in jobs.param_grid(grid):
        cfg['train_dir'] = jobs.make_train_dir(log_dir, experiment_name, cfg, keys_for_dir_name)
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    #SGD with only unique from first 500 words
    grid = copy.deepcopy(grid)
    grid['optimizer_type'] = "SGD"
    grid['use_all_words'] = False
    grid['words_to_use'] = 500
    grid['learning_rate'] = [30,300,1500,3000, 4500]
    grid['word_embedding_type'] = ['sum', 'mean']
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    for cfg in jobs.param_grid(grid):
        cfg['train_dir'] = jobs.make_train_dir(log_dir, experiment_name, cfg, keys_for_dir_name)
        jobs.submit(commands, cfg, get_slurm_script_gpu)
