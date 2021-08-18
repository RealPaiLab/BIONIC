#!/usr/bin/env python3

import time
from pathlib import Path
from .train import Trainer
from .utils.common import create_time_taken_string
from pathlib import Path
import argparse




# TODO: add config params to cli
def train(config_path: Path):
    """Integrates networks using BIONIC.

    All relevant parameters for the model should be specified in a `.json` config file.

    See https://github.com/bowang-lab/BIONIC/blob/master/README.md for details on writing
    the config file, as well as usage tips.
    """
    time_start = time.time()
    trainer = Trainer(config_path)
    trainer.train()
    trainer.forward()
    time_end = time.time()
    print(create_time_taken_string(time_start, time_end))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-config', '--config_file', help='train features generated from BIONIC')

    args = vars(parser.parse_args())

    config_file = args['config_file']
    train(Path(config_file))

if __name__ == '__main__':
    main()

