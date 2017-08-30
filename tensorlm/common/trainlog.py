# Copyright (c) 2017 Kilian Batzner All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""For logging and saving stats about the training progress."""

import datetime
import json
import os
import sys

import numpy as np
from dateutil import parser
from time import time

from tensorlm.common.log import get_logger

LOGGER = get_logger(__name__)


class Log:
    def __init__(self):
        # The key to all dicts is the step number
        self._losses_train = {}
        self._losses_valid = {}
        self._samples = {}
        self._epochs = {1: 1}
        self._interval_seconds = {}

    def log_loss_train(self, step, loss, interval_seconds):
        self._losses_train[step] = loss
        self._interval_seconds[step] = interval_seconds

    def log_loss_valid(self, step, loss):
        self._losses_valid[step] = loss

    def log_sampled(self, step, sampled):
        self._samples[step] = sampled

    def log_epoch(self, step, epoch):
        self._epochs[step] = epoch

    def get_time_since_start(self):
        total_seconds = sum(self._interval_seconds.values())
        return datetime.timedelta(seconds=total_seconds)

    def to_json(self):
        return {
            'lossesTrain': self._losses_train,
            'lossesValid': self._losses_valid,
            'samples': self._samples,
            'epochs': self._epochs,
            'intervalSeconds': self._interval_seconds
        }

    @staticmethod
    def from_json(json_dict):
        log = Log()
        log._losses_train = cast_keys(json_dict['lossesTrain'], int)
        log._losses_valid = cast_keys(json_dict['lossesValid'], int)
        log._samples = cast_keys(json_dict['samples'], int)
        log._epochs = cast_keys(json_dict['epochs'], int)
        log._interval_seconds = cast_keys(json_dict['intervalSeconds'], int)
        return log


class TrainState:
    trainstate_file_name = "trainlog.json"

    def __init__(self, global_step=0, epoch=1, step_in_epoch=0, log=None, start_time=None):
        self.global_step = global_step
        self.epoch = epoch
        self.step_in_epoch = step_in_epoch
        self._log = log if log else Log()
        self.start_time = start_time if start_time else datetime.datetime.utcnow()

        self.last_train_losses = []
        self.last_log_interval_start = time()

    def train_step_done(self, loss, log_interval, print_log=True):
        self.global_step += 1
        self.step_in_epoch += 1

        self.last_train_losses.append(loss)

        # Log the train loss
        if log_interval and self.global_step % log_interval == 0:
            avg_loss = float(np.mean(self.last_train_losses))
            self.last_train_losses = []
            interval_seconds = time() - self.last_log_interval_start
            self.last_log_interval_start = time()
            self._log.log_loss_train(self.global_step, avg_loss, interval_seconds)

            if print_log:
                LOGGER.info('Epoch %d Step %d Avg. Loss: %f Time: %s' %
                            (self.epoch, self.global_step, avg_loss,
                             self._log.get_time_since_start()))

    def epoch_done(self):
        self.epoch += 1
        self._log.log_epoch(self.global_step, self.epoch)

    def log_dev_loss(self, loss, print_log=True):
        self._log.log_loss_valid(self.global_step, loss)
        if print_log:
            LOGGER.info('Validation loss: %f' % loss)

    def log_sampled(self, text, print_log=True):
        self._log.log_sampled(self.global_step, text)
        if print_log:
            LOGGER.info('Sample: ' + text)

    def to_json(self):
        return {
            'step': self.global_step,
            'epoch': self.epoch,
            'stepInEpoch': self.step_in_epoch,
            'log': self._log,
            'startTime': self.start_time
        }

    @staticmethod
    def from_json(json_dict):
        obj = TrainState()
        obj.global_step = json_dict['step']
        obj.epoch = json_dict['epoch']
        obj.step_in_epoch = json_dict['stepInEpoch']
        obj._log = json_dict['log']
        obj.start_time = json_dict['startTime']
        return obj

    def save_to_dir(self, out_dir):
        out_path = os.path.join(out_dir, TrainState.trainstate_file_name)
        with open(out_path, "w") as f:
            json.dump(self.to_json(), f, default=json_encode)

    @staticmethod
    def try_load_from_dir(out_dir):
        if not out_dir:
            return TrainState()

        out_path = os.path.join(out_dir, TrainState.trainstate_file_name)
        if os.path.isfile(out_path):
            return TrainState.load_from_dir(out_dir)
        else:
            return TrainState()

    @staticmethod
    def load_from_dir(out_dir):
        out_path = os.path.join(out_dir, TrainState.trainstate_file_name)
        with open(out_path) as f:
            json_dict = json.load(f, object_hook=json_decode)
        return TrainState.from_json(json_dict)


def json_encode(obj):
    try:
        serial = obj.to_json()
        serial['classname'] = obj.__class__.__qualname__
        return serial
    except AttributeError:
        pass

    # Convert numpy types:
    if type(obj) in [np.int8, np.int16, np.int32, np.int64]:
        return int(obj)
    elif type(obj) in [np.float16, np.float32, np.float64, np.float128]:
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError('Type not serialisable')


def json_decode(loaded_json):
    # Transform a given value or collection to serialisable objects (creates a new dict).
    if type(loaded_json) is dict:
        # Decode the attributes before constructing the object
        decoded_items = dict([(key, json_decode(value)) for key, value in loaded_json.items()])
        if 'classname' in loaded_json:
            # Construct an object
            return json_decode_obj(decoded_items)
        else:
            # Just return the dict
            return decoded_items
    if type(loaded_json) is list:
        return map(json_decode, loaded_json)

    # Try to parse strings to dates
    if type(loaded_json) is str:
        try:
            return parser.parse(loaded_json)
        except ValueError:
            pass

    # Don't do anything for other types
    return loaded_json


def json_decode_obj(json_dict):
    # Transform a given dictionary to a serializable object. Dictionary must contain 'classname' key
    cls = str_to_class(json_dict['classname'])
    try:
        return cls.from_json(json_dict)
    except AttributeError:
        raise TypeError('The class given by the classname dict entry is not deserializable')


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


def cast_keys(dictionary, new_type):
    result = dict()
    for key, value in dictionary.items():
        result[new_type(key)] = value
    return result
