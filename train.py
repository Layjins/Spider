"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import random
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from mmengine import Config

import spider.tasks as tasks
from spider.common.config import parse_args
from spider.common.dist_utils import get_rank, init_distributed_mode
from spider.common.logger import setup_logger
from spider.common.registry import registry
from spider.common.utils import now


def setup_seeds(seed):
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    # import pdb
    # pdb.set_trace()
    runner_cls = registry.get_runner_class(cfg.run.get("runner", "runner_base"))

    return runner_cls


def setup_output_dir(cfg, job_id):
    lib_root = Path(registry.get_path("library_root"))

    # import pdb
    # pdb.set_trace()

    output_dir = lib_root / cfg.run.output_dir / job_id
    result_dir = output_dir / "result"

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    registry.register_path("result_dir", str(result_dir))
    registry.register_path("output_dir", str(output_dir))
    setup_logger(output_dir)



def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config.fromfile(args.config)

    init_distributed_mode(cfg.run)

    setup_output_dir(cfg, job_id)

    setup_seeds(cfg.run.seed)


    # set after init_distributed_mode() to only log on master.
    logging.info(cfg.pretty_text)

    task = tasks.setup_task(cfg)
    # import pdb
    # pdb.set_trace()
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
