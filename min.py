import config
import models
import tasks
import networks
import plots
import argparse
import json
import time
import random
import hashlib
import numpy as np
import torch

class Task:
    def __init__(self):
        self.name = "UNTITLED"
        pass
    def generate_batch(self, batch_size=64):
        return None, None, None
    def generate_dataset(self, N, batch_size):
        data = []
        labels = []
        output_masks = []
        for i in range(N):
            d, l, o = self.generate_batch(batch_size)
            data.append(d.unsqueeze(0))
            labels.append(l.unsqueeze(0))
            output_masks.append(o.unsqueeze(0))
        return torch.cat(data).to(config.device), torch.cat(labels).to(config.device), torch.cat(output_masks).to(
            config.device)
    def assess_accuracy(self, model, batch_size):
        return 0.0

class TWO_ORIENTATIONS(Task):
    # put orientation_neurons=0 for simple sin(2theta) and cos(2theta)
    def __init__(self, orientation_neurons=32, hold_orientation_for=30, hold_cue_for=30,
                 delay0_set=torch.arange(0, 31), delay1_set=torch.arange(0, 31), delay2_set=torch.arange(0, 31),
                 simple_input=False, simple_output=False):
        super().__init__()
        self.simple_input = simple_input or (not (orientation_neurons > 0))
        self.simple_output = simple_output or (not (orientation_neurons > 0))
        self.orientation_neurons = orientation_neurons
        self.dim_input = orientation_neurons + 2 if not simple_input else 4
        self.dim_output = 2 if simple_output else orientation_neurons
        self.hold_orientation_for = hold_orientation_for
        self.hold_cue_for = hold_cue_for
        self.delay0_set = delay0_set
        self.delay1_set = delay1_set
        self.delay2_set = delay2_set
        self.name = "TWOORI"

    # Ning Quan
    def o_spikes(self, pref, stim, exponent, maxSpike, k):
        # o_spikes: spike numbers per trial for orientation tuning cells
        # r = o_spikes(pref, stim, exponent, k)
        # pref: row vec for cells' preferred orientations
        # stim: column vec for stimulus orientations
        # exponent: scalar determining the widths of tuning. larger value for sharper tuning
        # maxSpike: scalar for mean max spike number when pref = stim
        # k: scalar for determining variance = k * mean
        # spikes: different columuns for cells with different pref orintations
        #         different rows for different stim orientations
        np_ = pref.shape[0]  # number of elements in pref
        ns = stim.shape[0]  # number of elements in stim

        prefs = torch.ones((ns, 1)) @ pref[None, :]  # ns x np array, (ns x 1) @ (1 x np)
        stims = stim[:, None] @ torch.ones((1, np_))  # ns x np array, (ns x 1) @ (1 x np)

        # mean spike numbers
        meanSpike = maxSpike * (0.5 * (torch.cos(2 * (prefs - stims)) + 1)) ** exponent  # ns x np array

        # sigma for noise
        sigmaSpike = torch.sqrt(k * meanSpike)

        # spikes = normrnd(meanSpike, sigmaSpike)# ns x np array, matlab
        spikes = torch.normal(meanSpike, sigmaSpike)  # ns x np array, python

        # no negative spike numbers
        spikes[spikes < 0] = 0  # ns x np array
        return spikes

    def _orientation_representation(self, orientation, simple=False):
        # add 2 for go cue signals
        if simple:
            rates = torch.zeros(2)
            theta = 2 * orientation / 180 * math.pi
            rates[0] = math.sin(theta)
            rates[1] = math.cos(theta)
        else:
            if False:
                rates = torch.tensor(range(self.orientation_neurons)) * 180 / self.orientation_neurons
                rates -= orientation
                rates = torch.exp(-rates ** 2 / 2 / 30 / 30) - 0.2
                rates = torch.maximum(rates, torch.zeros((self.orientation_neurons,)))
                rates /= 0.8
            if False:
                rates = torch.tensor(range(self.orientation_neurons)) * 180 / self.orientation_neurons
                rates -= orientation
                rates = torch.exp(-rates ** 2 / 2 / 25 / 25) - 0.2
                rates *= 5
                rates = torch.maximum(torch.tanh(rates), torch.tensor(0).to(config.device))
                #rates /= 0.8
            if True:
                pref = math.pi * torch.arange(self.orientation_neurons) / self.orientation_neurons
                stim = torch.tensor([(orientation / 180 * math.pi)], dtype=torch.float32)
                exponent = 4
                maxSpike = 1
                k = 0
                rates = self.o_spikes(pref, stim, exponent, maxSpike, k)[0]
        return rates

    def _input_rates(self, orientation, simple=False):
        representation = self._orientation_representation(orientation, simple)
        return torch.cat((representation, torch.zeros(self.dim_input-representation.shape[0])))
    def _output_rates(self, orientation, simple=False):
        return self._orientation_representation(orientation, simple)

    def _make_trial(self, orientation1=None, orientation2=None, delay0=None, delay1=None, delay2=None, output_info=False):
        if orientation1 is None: orientation1 = random.randint(0, 179)
        if orientation2 is None: orientation2 = random.randint(0, 179)
        if delay0 is None:
            delay0 = self.delay0_set[random.randint(0, self.delay0_set.shape[0]-1)]
        if delay1 is None:
            delay1 = self.delay1_set[random.randint(0, self.delay1_set.shape[0]-1)]
        if delay2 is None:
            delay2 = self.delay2_set[random.randint(0, self.delay2_set.shape[0]-1)]

        i_orientation1 = self._input_rates(orientation1, simple=self.simple_input).repeat(self.hold_orientation_for, 1)
        i_orientation2 = self._input_rates(orientation2, simple=self.simple_input).repeat(self.hold_orientation_for, 1)
        i_delay0 = torch.zeros((delay0, self.dim_input))
        i_delay1 = torch.zeros((delay1, self.dim_input))
        i_delay2 = torch.zeros((delay2, self.dim_input))
        i_cue1 = torch.zeros((self.hold_cue_for, self.dim_input))
        i_cue1[:, -2] = 1
        i_cue2 = torch.zeros((self.hold_cue_for, self.dim_input))
        i_cue2[:, -1] = 1
        to_batch = torch.cat((i_delay0, i_orientation1, i_delay1, i_orientation2, i_delay2, i_cue1, i_cue2))

        out1 = self._output_rates(orientation1, simple=self.simple_output)
        out1 = out1.repeat(self.hold_cue_for, 1)
        out2 = self._output_rates(orientation2, simple=self.simple_output)
        out2 = out2.repeat(self.hold_cue_for, 1)

        to_batch_labels = torch.cat((torch.zeros(self.hold_orientation_for * 2 + delay0 + delay1 + delay2, self.dim_output), out1, out2))

        to_mask = torch.cat((torch.zeros((self.hold_orientation_for * 2 + delay0 + delay1 + delay2,)),
                             torch.ones((self.hold_cue_for * 2,))))

        if output_info:
            return to_batch, to_batch_labels, to_mask, (orientation1, orientation2, delay0, delay1, delay2)
        else:
            return to_batch, to_batch_labels, to_mask

    def generate_batch(self, batch_size=64):
        batch = []
        batch_labels = []
        output_masks = []
        delay0 = self.delay0_set[random.randint(0, self.delay0_set.shape[0]-1)]
        delay1 = self.delay1_set[random.randint(0, self.delay1_set.shape[0]-1)]
        delay2 = self.delay2_set[random.randint(0, self.delay2_set.shape[0]-1)]
        for j in range(batch_size):
            to_batch, to_batch_labels, to_mask = self._make_trial(delay0=delay0, delay1=delay1, delay2=delay2)
            batch.append(to_batch.unsqueeze(0))
            batch_labels.append(to_batch_labels.unsqueeze(0))
            output_masks.append(to_mask.unsqueeze(0))
        return torch.cat(batch).to(config.device), torch.cat(batch_labels).to(config.device), torch.cat(
            output_masks).to(config.device)

    def assess_accuracy(self, model, batch_size=64):
        batch = self.generate_batch(batch_size=batch_size)
        out = model(batch[0])
        return 0.0

class TWO_ORIENTATIONS_DOUBLE_OUTPUT(TWO_ORIENTATIONS):
    # put orientation_neurons=0 for simple sin(2theta) and cos(2theta)
    def __init__(self, orientation_neurons=32, hold_orientation_for=30, hold_cue_for=30,
                 delay0_set=torch.arange(0, 31), delay1_set=torch.arange(0, 31), delay2_set=torch.arange(0, 31),
                 simple_input=False, simple_output=False, hold_outputs_at_zero=False):
        super().__init__(orientation_neurons, hold_orientation_for, hold_cue_for, delay0_set, delay1_set, delay2_set,
                         simple_input=simple_input, simple_output=simple_output)
        self.dim_output = 4 if self.simple_output else orientation_neurons*2
        self.dim_input = orientation_neurons + 1 if not self.simple_input else 3
        self.name = "TWOORIDO"
        self.hold_outputs_at_zero = hold_outputs_at_zero

    def _make_trial(self, orientation1=None, orientation2=None, delay0=None, delay1=None, delay2=None, output_info=False):
        if orientation1 is None: orientation1 = random.random() * 180#random.randint(0, 179)
        if orientation2 is None: orientation2 = random.random() * 180#random.randint(0, 179)
        if delay0 is None:
            delay0 = self.delay0_set[random.randint(0, self.delay0_set.shape[0]-1)]
        if delay1 is None:
            delay1 = self.delay1_set[random.randint(0, self.delay1_set.shape[0]-1)]
        if delay2 is None:
            delay2 = self.delay2_set[random.randint(0, self.delay2_set.shape[0]-1)]

        i_orientation1 = self._input_rates(orientation1, simple=self.simple_input).repeat(self.hold_orientation_for, 1)
        i_orientation2 = self._input_rates(orientation2, simple=self.simple_input).repeat(self.hold_orientation_for, 1)
        i_delay0 = torch.zeros((delay0, self.dim_input))
        i_delay1 = torch.zeros((delay1, self.dim_input))
        i_delay2 = torch.zeros((delay2, self.dim_input))
        i_cues = torch.zeros((self.hold_cue_for, self.dim_input))
        i_cues[:, -1] = 1
        to_batch = torch.cat((i_delay0, i_orientation1, i_delay1, i_orientation2, i_delay2, i_cues))

        out = torch.cat((self._output_rates(orientation1, simple=self.simple_output),
                        self._output_rates(orientation2, simple=self.simple_output)))
        out = out.repeat(self.hold_cue_for, 1)

        to_batch_labels = torch.cat((torch.zeros(self.hold_orientation_for * 2 + delay0 + delay1 + delay2, self.dim_output), out))

        if self.hold_outputs_at_zero:
            to_mask = torch.cat((torch.ones((self.hold_orientation_for * 2 + delay0 + delay1 + delay2,)),
                                 torch.ones((self.hold_cue_for,))))
        else:
            to_mask = torch.cat((torch.zeros((self.hold_orientation_for * 2 + delay0 + delay1 + delay2,)),
                                 torch.ones((self.hold_cue_for,))))

        if output_info:
            return to_batch, to_batch_labels, to_mask, (orientation1, orientation2, delay0, delay1, delay2)
        else:
            return to_batch, to_batch_labels, to_mask