#!/usr/bin/env python3

import time
from pathlib import Path
from .train import Trainer
from .utils.common import create_time_taken_string
import sys
from pathlib import Path




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
    train(Path('/Users/jyu/Documents/jenny_bionic_new/BIONIC/bionic/config/patient_similarity.json'))

if __name__ == '__main__':
    main()

