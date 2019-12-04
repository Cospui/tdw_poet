import ray
from tdw_distributed.es import initialize_worker
from tdw_distributed.poet_algo import MultiESOptimizer

from argparse import ArgumentParser
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# argument
parser = ArgumentParser()

parser.add_argument('--master_seed', type=int, default=111)  # set as np.seed
parser.add_argument('--start_from', default=None)  # Json file to start from

parser.add_argument('--init', default='random')  # can only be random! ( how the parameter of model is init )
parser.add_argument('--stochastic', action='store_true', default=False)  # when activate, it becomes True

# related to the ESOptimizer
parser.add_argument('log_file', type=str, default="./tdw_logs/log_0")
parser.add_argument('--learning_rate', type=float, default=0.0001)  # lr
parser.add_argument('--lr_decay', type=float, default=0.9999)
parser.add_argument('--lr_limit', type=float, default=0.0001)
parser.add_argument('--batches_per_chunk', type=int, default=50)  # batch
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--eval_batch_size', type=int, default=1)  # eval
parser.add_argument('--eval_batches_per_step', type=int, default=50)
parser.add_argument('--noise_std', type=float, default=0.5)  # noise
parser.add_argument('--noise_decay', type=float, default=0.999)
parser.add_argument('--noise_limit', type=float, default=0.01)
parser.add_argument('--l2_coeff', type=float, default=0.01)  # l2
parser.add_argument('--normalize_grads_by_noise_std', action='store_true', default=False)
parser.add_argument('--returns_normalization', default='normal')

parser.add_argument('--max_num_envs', type=int, default=100)
parser.add_argument('--adjust_interval', type=int, default=1)

# related to optimizer
parser.add_argument('--n_iterations', type=int, default=10)
parser.add_argument('--propose_with_adam', action='store_true', default=False)  # store true: when mentioned it becomes true
parser.add_argument('--checkpointing', action='store_true', default=False)
parser.add_argument('--steps_before_transfer', type=int, default=2)

# env candidate
parser.add_argument('--repro_threshold', type=int, default=1)

parser.add_argument('--envs', nargs='+')  # open / close certain features of envs
# use like this: --envs target 

# related to the bound
# TODO: what is the score for each env?? how to measure that?
parser.add_argument('--mc_lower', type=int, default=-1)
parser.add_argument('--mc_upper', type=int, default=10)

args = parser.parse_args()
logger.info(args)

CPU_NUM = 4
# ray parallel
# ray.init(num_cpus=CPU_NUM, ignore_reinit_error=True)  # TODO change #cpu
# print("ray inited")


def run():
    print("===== START =====")

    # ray.get(initialize_worker.remote())
    initialize_worker()
    # print("RUN -> initialized")
    # from IPython import embed
    # embed()

    optimizer_zoo = MultiESOptimizer(args=args)

    print("===== OPTIMIZE =====")

    optimizer_zoo.optimize(iterations=args.n_iterations,
                           propose_with_adam=args.propose_with_adam,
                           reset_optimizer=True,
                           checkpointing=args.checkpointing,
                           steps_before_transfer=args.steps_before_transfer)

    print("=====FINISH=====")


run()
