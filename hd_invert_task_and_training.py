import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn

verbose = True  # print info in console?

hyperparameters = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "random_string": "X",  # human-readable string used for random initialization (for reproducibility)
    "noise_amplitude": 0.1,  # normal noise with s.d. = noise_amplitude
    "optimizer": "Adam",  # options: Adam
    "train_for_steps": 2000,
    "save_network_every_steps": 1000,
    "note_error_every_steps": 10,  # only relevant if verbose is True
    "clip_gradients": True,  # limit gradient size (allows the network to train for a long time without diverging)
    "max_gradient_norm": 10,
    "regularization": "None",  # options: L1, L2, None
    "regularization_lambda": 1e-4
}
hyperparameters["random_seed"] = abs(hash(hyperparameters["random_string"])) % 10**8  # random initialization seed (for reproducibility)

task_parameters = {
    "task_name": "2ORI2O",
    "input_orientation_units": 32,  # how many orientation-selective input units?
    "delay0_from": 40, "delay0_to": 60,  # range (inclusive) for lengths of variable delays (in timesteps)
    "delay1_from": 40, "delay1_to": 60,
    "delay2_from": 40, "delay2_to": 60,
    "show_orientation_for": 10,  # in timesteps
    "show_cue_for": 100,  # in timesteps
    "dim_input": 32 + 1,  # plus one input for go cue signal
    "dim_output": 4
}

model_parameters = {
    "model_name": "CTRNN-IT",
    "dim_input": task_parameters["dim_input"],
    "dim_output": task_parameters["dim_output"],
    "dim_recurrent": 100,
    "tau": 10,  # defines ratio tau/dt (see continuous-time recurrent neural networks)
    "nonlinearity": "retanh",  # options: retanh, tanh
    "input_bias": True,
    "output_bias": False
}

additional_comments = [
    "Training criterion: MSE loss",
    "Noise added at all timesteps",
    "Inversion of tuning network: training is on top-level parameters + output readout layer"
]

directory = "data/"
directory += f"{model_parameters['model_name']}_{task_parameters['task_name']}_"
directory = "data/test/"  # needs to end with a slash

random.seed(1337)
r1_pref_shifts = [random.randint(-60, 60)*1 for i in range(45)]

random.seed(hyperparameters["random_seed"])
torch.manual_seed(hyperparameters["random_seed"])
np.random.seed(hyperparameters["random_seed"])

class Task:
    # outputs mask defining which timesteps noise should be applied to
    # for a given choice of (delay0, delay1, delay2)
    # output is (total_time, )
    @staticmethod
    def get_noise_mask(delay0, delay1, delay2):
        noise_from_t = delay0 + task_parameters["show_orientation_for"] * 2 + delay1
        noise_to_t = noise_from_t + delay2
        total_t = noise_to_t + task_parameters["show_cue_for"]
        mask = torch.zeros(total_t)
        mask[:] = 1
        return mask

    @staticmethod
    def get_median_delays():
        delay0 = (task_parameters["delay0_from"]+task_parameters["delay0_to"])//2
        delay1 = (task_parameters["delay1_from"]+task_parameters["delay1_to"])//2
        delay2 = (task_parameters["delay2_from"]+task_parameters["delay2_to"])//2
        return delay0, delay1, delay2

    # orientation tuning curve for input cells. Based on:
    # Andrew Teich & Ning Qian (2003) "Learning and adaptation in a recurrent model of V1 orientation selectivity"
    @staticmethod
    def _o_spikes(pref, stim, exponent, max_spike, k):
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
        mean_spike = max_spike * (0.5 * (torch.cos(2 * (prefs - stims)) + 1)) ** exponent  # ns x np array
        # sigma for noise
        sigma_spike = torch.sqrt(k * mean_spike)
        # spikes = normrnd(meanSpike, sigmaSpike)# ns x np array, matlab
        spikes = torch.normal(mean_spike, sigma_spike)  # ns x np array, python
        # no negative spike numbers
        spikes[spikes < 0] = 0  # ns x np array
        return spikes

    # convert input orientation angle (in deg) to firing rates of orientation-selective input units
    @staticmethod
    def _input_orientation_representation(orientation):
        pref = math.pi * torch.arange(task_parameters["input_orientation_units"]) / task_parameters["input_orientation_units"]
        stim = torch.tensor([(orientation / 180 * math.pi)], dtype=torch.float32)
        exponent = 4; max_spike = 1; k = 0
        rates = Task._o_spikes(pref, stim, exponent, max_spike, k)[0]
        return rates

    # convert target output orientation angles (in deg) to target firing rates of output units
    @staticmethod
    def _output_orientation_representation(orientation1, orientation2):
        rates = torch.zeros(4)
        theta = 2 * orientation1 / 180 * math.pi
        rates[0] = math.sin(theta)
        rates[1] = math.cos(theta)
        theta = 2 * orientation2 / 180 * math.pi
        rates[2] = math.sin(theta)
        rates[3] = math.cos(theta)
        return rates

    # generate parameters for a trial
    # (make choices for orientations and delay lengths)
    # can pass parameters to leave them unchanged
    @staticmethod
    def choose_trial_parameters(orientation1=None, orientation2=None, delay0=None, delay1=None, delay2=None):
        if orientation1 is None: orientation1 = random.random() * 180
        if orientation2 is None: orientation2 = random.random() * 180
        if delay0 is None: delay0 = random.randint(task_parameters["delay0_from"], task_parameters["delay0_to"])
        if delay1 is None: delay1 = random.randint(task_parameters["delay1_from"], task_parameters["delay1_to"])
        if delay2 is None: delay2 = random.randint(task_parameters["delay2_from"], task_parameters["delay2_to"])
        return orientation1, orientation2, delay0, delay1, delay2

    # generate one trial of the task (there will be batch_size of them in the batch)
    # orientation1 and orientation2 in degrees
    # output tensors: input, target, mask (which timesteps to include in the loss function)
    @staticmethod
    def _make_trial(orientation1, orientation2, delay0, delay1, delay2):
        # generate the tensor of inputs
        i_orientation1 = torch.zeros(task_parameters["dim_input"])
        i_orientation1[:task_parameters["input_orientation_units"]] = Task._input_orientation_representation(orientation1)
        i_orientation1 = i_orientation1.repeat(task_parameters["show_orientation_for"], 1)
        i_orientation2 = torch.zeros(task_parameters["dim_input"])
        i_orientation2[:task_parameters["input_orientation_units"]] = Task._input_orientation_representation(orientation2)
        i_orientation2 = i_orientation2.repeat(task_parameters["show_orientation_for"], 1)
        i_delay0 = torch.zeros((delay0, task_parameters["dim_input"]))
        i_delay1 = torch.zeros((delay1, task_parameters["dim_input"]))
        i_delay2 = torch.zeros((delay2, task_parameters["dim_input"]))
        i_cue = torch.zeros((task_parameters["show_cue_for"], task_parameters["dim_input"]))
        i_cue[:, -1] = 1
        i_full = torch.cat((i_delay0, i_orientation1, i_delay1, i_orientation2, i_delay2, i_cue))  # (total_time, dim_input)

        o_beforecue = torch.zeros(task_parameters["show_orientation_for"] * 2 + delay0 + delay1 + delay2, task_parameters["dim_output"])
        o_cue = Task._output_orientation_representation(orientation1, orientation2).repeat(task_parameters["show_cue_for"], 1)
        o_full = torch.cat((o_beforecue, o_cue))  # (total_time, dim_output)

        b_mask = torch.cat((torch.zeros((task_parameters["show_orientation_for"] * 2 + delay0 + delay1 + delay2,)),
                             torch.ones((task_parameters["show_cue_for"],))))  # (total_time,)

        return i_full, o_full, b_mask

    # generate a batch (of size batch_size)
    # all trials in batch have the same (delay0, delay1, delay2) but orientation1 and orientation2 vary (are random)
    # returns shapes (batch_size, total_time, dim_input), (batch_size, total_time, dim_output), (batch_size, total_time)
    @staticmethod
    def make_random_orientations_batch(batch_size, delay0, delay1, delay2):
        batch = []  # inputs in the batch
        batch_labels = []  # target outputs in the batch
        output_masks = []  # masks in the batch
        for j in range(batch_size):
            orientation1, orientation2, *_ = Task.choose_trial_parameters(None, None, delay0, delay1, delay2)
            i_full, o_full, b_mask = Task._make_trial(orientation1, orientation2, delay0, delay1, delay2)
            batch.append(i_full.unsqueeze(0))
            batch_labels.append(o_full.unsqueeze(0))
            output_masks.append(b_mask.unsqueeze(0))
        return torch.cat(batch), torch.cat(batch_labels), torch.cat(output_masks)

    # generate a batch (of size 180/resolution * 180/resolution)
    # all trials in batch have the same (delay0, delay1, delay2) but orientation1 and orientation2 vary (all int values, up to resolution)
    # returns shapes (batch_size, total_time, dim_input), (batch_size, total_time, dim_output), (batch_size, total_time)
    @staticmethod
    def make_all_integer_orientations_batch(delay0, delay1, delay2, resolution=1):
        batch = []  # inputs in the batch
        batch_labels = []  # target outputs in the batch
        output_masks = []  # masks in the batch
        for orientation1 in range(0, 180, resolution):
            for orientation2 in range(0, 180, resolution):
                i_full, o_full, b_mask = Task._make_trial(orientation1, orientation2, delay0, delay1, delay2)
                batch.append(i_full.unsqueeze(0))
                batch_labels.append(o_full.unsqueeze(0))
                output_masks.append(b_mask.unsqueeze(0))
        return torch.cat(batch), torch.cat(batch_labels), torch.cat(output_masks)

    # convert sin, cos outputs to the angles they represent (normalizing outputs to have sum of squares = 1)
    # converts separately for every trial and timestep
    # output o1 and o2 are (batch_size, t_to-t_from)
    @staticmethod
    def convert_sincos_to_angles(output, t_from, t_to):
        trig = output[:, t_from:t_to, :]
        o1 = torch.atan2((trig[:, :, 0] / (trig[:, :, 0] ** 2 + trig[:, :, 1] ** 2) ** 0.5),
                         (trig[:, :, 1] / (trig[:, :, 0] ** 2 + trig[:, :, 1] ** 2) ** 0.5)) / 2 * 180 / math.pi
        o2 = torch.atan2((trig[:, :, 2] / (trig[:, :, 2] ** 2 + trig[:, :, 3] ** 2) ** 0.5),
                         (trig[:, :, 3] / (trig[:, :, 2] ** 2 + trig[:, :, 3] ** 2) ** 0.5)) / 2 * 180 / math.pi
        return o1, o2

    # calculate MSE error between output and target
    # calculates raw MSE and also sqrt(MSE) in degrees (after normalizing and converting to angles)
    @staticmethod
    def calculate_errors(target, output, mask, t_from, t_to):
        error = torch.mean((output[mask == 1] - target[mask == 1]) ** 2, dim=0)
        mse_o1 = (error[0] + error[1]).item() / 2
        mse_o2 = (error[2] + error[3]).item() / 2
        o1_o, o2_o = Task.convert_sincos_to_angles(output, t_from, t_to)
        o1_t, o2_t = Task.convert_sincos_to_angles(target, t_from, t_to)
        error_o1 = torch.minimum(torch.minimum((o1_o - o1_t) ** 2, (o1_o - o1_t + 180) ** 2), (o1_o - o1_t - 180) ** 2)
        angle_error_o1 = torch.mean(error_o1).item() ** 0.5
        error_o2 = torch.minimum(torch.minimum((o2_o - o2_t) ** 2, (o2_o - o2_t + 180) ** 2), (o2_o - o2_t - 180) ** 2)
        angle_error_o2 = torch.mean(error_o2).item() ** 0.5
        return mse_o1, mse_o2, angle_error_o1, angle_error_o2


# continuous-time recurrent neural network (CTRNN)
# Tau * d(ah)/dt = -ah + W_h_ah @ f(ah) + W_ah_x @ x + b_ah
# Equation 1 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
#
# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + W_h_ah @ h[t−1] + W_x_ah @ x[t] + b_ah)
# h[t] = f(ah[t]) + noise[t], if noise_mask[t] = 1
# y[t] = W_h_y @ h[t] + b_y
#
# parameters to be learned: W_h_ah, W_x_ah, W_y_h, b_ah, b_y
# constants that are not learned: dt, Tau, noise
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim_input = model_parameters["dim_input"]
        dim_output = model_parameters["dim_output"]
        dim_recurrent = model_parameters["dim_recurrent"]
        self.dim_input, self.dim_output, self.dim_recurrent = dim_input, dim_output, dim_recurrent
        self.dt, self.tau = 1, model_parameters["tau"]
        if model_parameters["nonlinearity"] == "tanh": self.f = torch.tanh
        if model_parameters["nonlinearity"] == "retanh": self.f = lambda x: torch.maximum(torch.tanh(x), torch.tensor(0))

        #torch.nn.Parameter(torch.tensor(0.15))
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output, bias=model_parameters["output_bias"])  # y = W_h_y @ h + b_y
        for param in self.fc_h2y.parameters():
            param.requires_grad = True

        with torch.no_grad():
            R1a_N = 23
            R1b_N = 23
            R1_N = R1a_N + R1b_N
            R2_N = 44
            DT_N = 10
            R1a_i = torch.arange(0, R1a_N, dtype=int)
            R1b_i = torch.arange(R1a_N, R1a_N + R1b_N, dtype=int)
            R1_i = torch.arange(0, R1_N, dtype=int)
            DT_i = torch.arange(R1_N, R1_N + DT_N, dtype=int)
            R2_i = torch.arange(R1_N + DT_N, R1_N + DT_N + R2_N, dtype=int)
            R1a, R1b, R2, DT = R1a_i, R1b_i, R2_i, DT_i
            R1a_pref = torch.arange(len(R1a)) / len(R1a) * 180
            R1b_pref = torch.arange(len(R1b)) / len(R1b) * 180
            DT_pref = torch.arange(len(DT)) / len(DT) * 180
            R2_pref = torch.arange(len(R2)) / len(R2) * 180
            R1_pref_fixed = torch.tensor([1., 8., 16., 23., 32., 40., 47., 55., 63., 70., 78., 86.,
                                          94., 102., 110., 117., 125., 134., 140., 149., 156., 164., 172., 90.,
                                          98., 106., 114., 121., 130., 137., 145., 153., 161., 168., 177., 4.,
                                          12., 19., 28., 35., 43., 51., 59., 66., 74., 81.])
            #R1_pref_fixed = R1_pref
            self.fc_h2y.weight[1, R1a_i] = torch.cos(2 * (R1a_pref) / 180 * np.pi) * 0.0725
            self.fc_h2y.weight[0, R1a_i] = torch.sin(2 * (R1a_pref) / 180 * np.pi) * 0.0725
            self.fc_h2y.weight[1, R1b_i] = torch.cos(2 * (-R1b_pref) / 180 * np.pi) * 0.0725
            self.fc_h2y.weight[0, R1b_i] = torch.sin(2 * (-R1b_pref) / 180 * np.pi) * 0.0725
            self.fc_h2y.weight[3, R2_i] = torch.cos(2 * R2_pref / 180 * np.pi) * 0.0725 * 1.1
            self.fc_h2y.weight[2, R2_i] = torch.sin(2 * R2_pref / 180 * np.pi) * 0.0725 * 1.1

            self.fc_h2y.weight[1, R1_i] = torch.cos(2 * (R1_pref_fixed) / 180 * np.pi) * 0.0725
            self.fc_h2y.weight[0, R1_i] = torch.sin(2 * (R1_pref_fixed) / 180 * np.pi) * 0.0725

        self.in_r_m = 0.2
        self.in_dt_m = 0.3
        self.r_r_m = 0.2
        self.ra_rb_m = 0.2
        self.rb_ra_m = 0.2
        self.r1_dt_m = -0.1
        self.dt_r2_m = -0.5
        self.r1_b_m = -0.06
        self.r2_b_m = -0.1
        self.dt_b_m = -0.26

        self.in_r_m = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=False)
        self.in_dt_m = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        self.r_r_m = torch.nn.Parameter(torch.tensor([0.25]))
        self.ra_rb_m = torch.nn.Parameter(torch.tensor([-0.26]))
        self.rb_ra_m = torch.nn.Parameter(torch.tensor([-0.24]))
        self.r1_dt_m = torch.nn.Parameter(torch.tensor([-0.1]), requires_grad=True)
        self.dt_r2_m = torch.nn.Parameter(torch.tensor([-0.5]), requires_grad=True)
        self.r1_b_m = torch.nn.Parameter(torch.tensor([-0.1]))
        self.r2_b_m = torch.nn.Parameter(torch.tensor([-0.1]), requires_grad=True)
        self.dt_b_m = torch.nn.Parameter(torch.tensor([-0.1]), requires_grad=True)

        self.r1_pref_shifts = torch.nn.Parameter(torch.tensor(r1_pref_shifts, dtype=float), requires_grad=False)

    # output y and recurrent unit activations for all trial timesteps
    # input has shape (batch_size, total_time, dim_input) or (total_time, dim_input)
    # noise has shape (batch_size, total_time, dim_recurrent) or (total_time, dim_recurrent)
    def forward(self, input, noise):
        R1a_N = 23
        R1b_N = 23
        R1_N = R1a_N + R1b_N
        R2_N = 44
        DT_N = 10
        R1a_i = torch.arange(0, R1a_N, dtype=int)
        R1b_i = torch.arange(R1a_N, R1a_N + R1b_N, dtype=int)
        R1_i = torch.arange(0, R1_N, dtype=int)
        DT_i = torch.arange(R1_N, R1_N + DT_N, dtype=int)
        R2_i = torch.arange(R1_N + DT_N, R1_N + DT_N + R2_N, dtype=int)
        R1a, R1b, R2, DT = R1a_i, R1b_i, R2_i, DT_i
        R1a_pref = torch.arange(len(R1a)) / len(R1a) * 180
        R1b_pref = torch.arange(len(R1b)) / len(R1b) * 180
        DT_pref = torch.arange(len(DT)) / len(DT) * 180
        R2_pref = torch.arange(len(R2)) / len(R2) * 180

        ah0 = torch.zeros((self.dim_recurrent,))
        b_ah = torch.zeros((self.dim_recurrent,))
        b_y = torch.zeros((self.dim_output,))
        W_h_ah = torch.zeros((self.dim_recurrent, self.dim_recurrent))
        W_x_ah = torch.zeros((self.dim_recurrent, self.dim_input))
        W_h_y = torch.zeros((self.dim_output, self.dim_recurrent))

        orientation_neurons = task_parameters["input_orientation_units"]

        def weight(pref1, pref2):
            return 1 * (torch.cos(2 * (pref1 - pref2) / 180 * np.pi))
            a = 0.5
            return np.maximum(np.minimum(1 + a - torch.abs(pref1 - pref2) / 90 * 2 * (1 + a), 1.), -1.)
        def weight_ns(pref1, pref2):
            return 1
        def put_weights(units_from, units_to, units_from_pref=None, units_to_pref=None, weight_fun=weight,
                        m=None, m_s=None):
            if type(units_from) is str: units_from = units_from.upper()

            uf_N = orientation_neurons if units_from == "IN" else len(units_from)
            ut_N = len(units_to)
            if m_s is not None: m = m_s / uf_N / ut_N
            if m is None: m = 1
            magnitude = m
            if units_from_pref is None:
                units_from_pref = torch.arange(uf_N) * 180 / uf_N
            if units_to_pref is None:
                units_to_pref = torch.arange(ut_N) * 180 / ut_N

            if units_from == "IN":
                units_from = torch.arange(orientation_neurons)
                for i, uf in enumerate(units_from):
                    for j, ut in enumerate(units_to):
                        W_x_ah[ut, uf] = weight_fun(units_from_pref[i], units_to_pref[j]) * magnitude
            else:
                for i, uf in enumerate(units_from):
                    for j, ut in enumerate(units_to):
                        W_h_ah[ut, uf] = weight_fun(units_from_pref[i], units_to_pref[j]) * magnitude

        def put_bias(units_to, m=None, m_s=None):
            ut_N = len(units_to)
            if m_s is not None: m = m_s / 32 / ut_N
            if m is None: m = 1
            magnitude = m
            b_ah[units_to] = magnitude

        put_weights("IN", DT, m=self.in_dt_m, weight_fun=weight_ns)
        put_weights("IN", R1a, m=self.r_r_m)
        put_weights("IN", R1b, m=self.r_r_m)
        put_weights("IN", R2, m=self.r_r_m)

        put_weights(R2, R2, m=self.r_r_m)
        # put_weights(model, DT, DT, m=0.16)

        put_weights(R1a, R1a, m=self.r_r_m)
        put_weights(R1b, R1b, m=self.r_r_m)
        put_weights(R1a, R1b, m=self.ra_rb_m)
        put_weights(R1b, R1a, m=self.rb_ra_m)

        # put_weights(model, R1a, DT, units_to_pref=DT_pref, m=-0.3)
        put_weights(R1b, DT, units_to_pref=DT_pref, m=self.r1_dt_m, weight_fun=weight_ns)
        # put_weights(model, R2, DT, units_to_pref=DT_pref, m=-0.16/4)
        put_weights(DT, R2, m=self.dt_r2_m, weight_fun=weight_ns)  # m=0.15*R1_N/DT_N if DT_N>0 else 0)

        put_bias(R1a, m=self.r1_b_m)
        put_bias(R1b, m=self.r1_b_m)
        put_bias(DT, m=self.dt_b_m)
        put_bias(R2, m=self.r2_b_m)

        if len(input.shape) == 2:
            # if input has size (total_time, dim_input) (if there is only a single trial), add a singleton dimension
            input = input[None, :, :]  # (batch_size, total_time, dim_input)
            noise = noise[None, :, :]  # (batch_size, total_time, dim_recurrent)
        batch_size, total_time, dim_input = input.shape
        ah = ah0.repeat(batch_size, 1)
        h = self.f(ah)
        hstore = []  # store all recurrent activations at all timesteps. Shape (batch_size, total_time, dim_recurrent)
        for t in range(total_time):
            ah = ah + (self.dt / self.tau) * (-ah + (W_h_ah@h.T).T + (W_x_ah@input[:, t].T).T + b_ah)
            h = self.f(ah) + noise[:, t, :]
            hstore.append(h)
        hstore = torch.stack(hstore, dim=1)
        #output = torch.bmm(hstore, W_h_y.T.repeat(batch_size, 1, 1))
        output = self.fc_h2y(hstore)
        return output, hstore


# train the network on the task.
# outputs: error_store, error_store_o1, error_store_o2, gradient_norm_store
# error_store[j] -- the error after j parameter updates
# error_store_o1, error_store_o1 are errors in o1 and o2, respectively
# gradient_norm_store[j] -- the norm of the gradient after j parameter updates
def train_network(model):
    def save_network(model, path):
        _path = pathlib.Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, path)
    optimizer = None
    if hyperparameters["optimizer"].upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    batch_size = hyperparameters["batch_size"]
    max_steps = hyperparameters["train_for_steps"]
    error_store = torch.zeros(max_steps + 1)
    error_store_o1 = torch.zeros(max_steps + 1)
    error_store_o2 = torch.zeros(max_steps + 1)
    # error_store[0] is the error before any parameter updates have been made,
    # error_store[j] is the error after j parameter updates
    # error_store_o1, error_store_o1 are errors in o1 and o2, respectively
    gradient_norm_store = torch.zeros(max_steps + 1)
    # gradient_norm_store[0] is norm of the gradient before any parameter updates have been made,
    # gradient_norm_store[j] is the norm of the gradient after j parameter updates
    noise_amplitude = hyperparameters["noise_amplitude"]
    regularization_norm, regularization_lambda = None, None
    if hyperparameters["regularization"].upper() == "L1":
        regularization_norm = 1
        regularization_lambda = hyperparameters["regularization_lambda"]
    if hyperparameters["regularization"].upper() == "L2":
        regularization_norm = 2
        regularization_lambda = hyperparameters["regularization_lambda"]
    clip_gradients = hyperparameters["clip_gradients"]
    max_gradient_norm = hyperparameters["max_gradient_norm"]
    set_note_error = list(range(0, max_steps, hyperparameters["note_error_every_steps"]))
    if max_steps not in set_note_error: set_note_error.append(max_steps)
    set_note_error = np.array(set_note_error)
    set_save_network = list(range(0, max_steps, hyperparameters["save_network_every_steps"]))
    if max_steps not in set_save_network: set_save_network.append(max_steps)
    set_save_network = np.array(set_save_network)

    best_network_dict = None
    best_network_error = None
    for p in range(max_steps + 1):
        _, _, delay0, delay1, delay2 = Task.choose_trial_parameters()  # choose the delays for this batch
        input, target, output_mask = Task.make_random_orientations_batch(batch_size, delay0, delay1, delay2)
        noise_mask = Task.get_noise_mask(delay0, delay1, delay2)
        noise_mask = noise_mask.repeat(batch_size, 1).unsqueeze(2).repeat(1, 1, model.dim_recurrent)  # convert to (batch_size, total_time, dim_recurrent)
        noise = torch.randn_like(noise_mask) * noise_mask * noise_amplitude
        output, h = model.forward(input, noise=noise)
        # output_mask: (batch_size, total_time, dim_output) tensor, elements
        # 0 (if timestep does not contribute to this term in the error function),
        # 1 (if timestep contributes to this term in the error function)
        error = torch.sum((output[output_mask == 1] - target[output_mask == 1]) ** 2, dim=0) / torch.sum(output_mask == 1)
        error_o1 = (error[0] + error[1]).item()
        error_o2 = (error[2] + error[3]).item()
        #error = error[0] + error[1]
        error = torch.sum(error)
        if regularization_norm == 1:
            for param in model.parameters():
                if param.requires_grad is True:
                    error += regularization_lambda * torch.sum(torch.abs(param))
        if regularization_norm == 2:
            for param in model.parameters():
                if param.requires_grad is True:
                    error += regularization_lambda * torch.sum(param ** 2)
        error_store[p] = error.item()
        error_store_o1[p] = error_o1
        error_store_o2[p] = error_o2

        # don't train on step 0, just store error
        if p == 0:
            best_network_dict = model.state_dict()
            best_network_error = error.item()
            save_network(model, directory + f'model_best.pth')
            last_time = time.time()  # for estimating how long training will take
            continue
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights of the model)
        optimizer.zero_grad()
        # Backward pass: compute gradient of the error with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        error.backward()
        # clip the norm of the gradient
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        with torch.no_grad():
            #pass
            model.fc_h2y.weight[0, 45:] = 0
            model.fc_h2y.weight[1, 45:] = 0
            model.fc_h2y.weight[2, :46] = 0
            model.fc_h2y.weight[3, :46] = 0

        # store gradient norms
        gradient = []  # store all gradients
        for param in model.parameters():  # model.parameters include those defined in __init__ even if they are not used in forward pass
            if param.requires_grad is True:  # model.parameters include those defined in __init__ even if param.requires_grad is False (in this case param.grad is None)
                if param.grad is not None:
                    gradient.append(param.grad.detach().flatten().numpy())
                else:
                    print("none param")
                    print(param)
        gradient = np.concatenate(gradient)
        gradient_norm_store[p] = np.sqrt(np.sum(gradient ** 2)).item()
        # note running error in console
        if verbose and np.isin(p, set_note_error):
            error_wo_reg = torch.sum((output[output_mask == 1] - target[output_mask == 1]) ** 2) / torch.sum(output_mask == 1)
            print(f'{p} parameter updates: error = {error.item():.4g}, w/o reg {error_wo_reg.item():.4g}, o1 {error_o1:.4g}, o2 {error_o2:.4g}')
            passed_time = time.time() - last_time
            made_steps = hyperparameters["note_error_every_steps"]
            left_steps = max_steps-p
            left_time = left_steps / made_steps * passed_time
            print(f"took {int(passed_time)}s for {made_steps} steps, estimated time left {str(datetime.timedelta(seconds=int(left_time)))}")
            last_time = time.time()
        # save network
        if np.isin(p, set_save_network):
            print("SAVING", f'model_parameterupdate{p}.pth')
            save_network(model, directory + f'model_parameterupdate{p}.pth')
        if error.item() < best_network_error:
            best_network_dict = model.state_dict()
            best_network_error = error.item()
            save_network(model, directory + f'model_best.pth')

            # save parameters for hand designing
            params = {}
            params["in_r_m"] = model.in_r_m.data.item()
            params["in_dt_m"] = model.in_dt_m.data.item()
            params["r_r_m"] = model.r_r_m.data.item()
            params["ra_rb_m"] = model.ra_rb_m.data.item()
            params["rb_ra_m"] = model.rb_ra_m.data.item()
            params["r1_dt_m"] = model.r1_dt_m.data.item()
            params["dt_r2_m"] = model.dt_r2_m.data.item()
            params["r1_b_m"] = model.r1_b_m.data.item()
            params["r2_b_m"] = model.r2_b_m.data.item()
            params["dt_b_m"] = model.dt_b_m.data.item()
            params["r1_pref_shifts"] = model.r1_pref_shifts.data.tolist()
            params["w_h_y_0"] = model.fc_h2y.weight[0].tolist()
            params["w_h_y_1"] = model.fc_h2y.weight[1].tolist()
            params["w_h_y_2"] = model.fc_h2y.weight[2].tolist()
            params["w_h_y_3"] = model.fc_h2y.weight[3].tolist()
            with open(directory + "best_params.json", 'w', encoding='utf-8') as f:
                json.dump(params, f)
    return error_store, error_store_o1, error_store_o2, gradient_norm_store


if __name__ == "__main__":
    # train the network and save weights
    model = Model()
    error_store, error_store_o1, error_store_o2, gradient_norm_store = train_network(model)

    # save all parameters
    info = {
        "hyperparameters": hyperparameters,
        "task_parameters": task_parameters,
        "model_parameters": model_parameters,
        "additional_comments": additional_comments,
        "directory": directory,
    }
    with open(directory + "info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

    # save training dynamics
    training_dynamics = torch.cat((
        error_store.unsqueeze(0),
        error_store_o1.unsqueeze(0),
        error_store_o2.unsqueeze(0),
        gradient_norm_store.unsqueeze(0)
    ))
    torch.save(training_dynamics, directory + "training_dynamics.pt")

    # copy this script, analysis ipynb, and util script into the same directory
    # for easy importing in the jupyter notebook
    shutil.copy(sys.argv[0], directory+"task_and_training.py")


