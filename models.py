import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
#from torch.nn import functional as F# https://twitter.com/francoisfleuret/status/1247576431762210816
from torch import nn
import config
#device = torch.device('cpu')# device = torch.device('cuda')


class Model(nn.Module):
    def __init__(self, name="None"):
        super().__init__()
        self.name = name

    def forward(self, input, noise=None, noise_amplitude=None):
        if noise_amplitude is None:
            noise_amplitude = 0
        if noise is None:
            noise = noise_amplitude * torch.randn(self.get_noise_shape(input)).to(config.device)
        return self._forward(input.to(config.device), noise)

    def _forward(self, input, noise):
        pass

    def get_noise_shape(self, input):
        return 1

"""
class CARDS_WITH_CLUES_DT_RNN(Model):
    def __init__(self, n_cards=10, hidden_dim=50, activation="tanh",
                 Wih=None, Whh=None, b=None, name="DT_RNN"):
        super(Model, self, name=name).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_cards = n_cards
        self.activation = activation

        if Wih is None:
            pass
        if Whh is None:
            pass
        if b is None:
            pass
        self.Wih = torch.nn.Parameter(Wih)
        self.Whh = torch.nn.Parameter(Whh)
        self.b = torch.nn.Parameter(b)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, n_cards)

    # size(x) = (batch_size, T, dim_input)
    def _forward(self, x, noise):
        batch_size = x.size(0)

        # Next, using the output of CNN as input for RNN
        hidden, out = self.rnn(x)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        # out = self.fc(out)

        out = self.fc(hidden)

        # applying activation function at the end
        out = torch.sigmoid(out)

        return out, hidden
"""

#%%##############################################################################
# continuous time recurrent neural network
# Tau * dah/dt = -ah + Wahh @ f(ah) + Wahx @ x + bah
#
# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)􏰩    
# h[t] = f(ah[t]) + bhneverlearn[t], if t > 0
# y[t] = Wyh @ h[t] + by  output

# parameters to be learned: Wahh, Wahx, Wyh, bah, by, ah0(optional). In this implementation h[0] = f(ah[0]) with no noise added to h[0] except potentially through ah[0]
# constants that are not learned: dt, Tau, bhneverlearn
# Equation 1 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
class CTRNN(Model):# class CTRNN inherits from class torch.nn.Module
    def __init__(self, dim_recurrent, dim_input=None, dim_output=None,
                 Wahx=None, Wahh=None, Wyh=None, bah=None, by=None,
                 nonlinearity='retanh', ah0=None, LEARN_ah0=False,
                 dt=1, Tau=10, task=None,
                 name="CTRNN"):
        super().__init__(name=name)# super allows you to call methods of the superclass in your subclass

        if task is not None:
            if dim_input is None:
                dim_input = task.dim_input
            if dim_output is None:
                dim_output = task.dim_output
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_recurrent = dim_recurrent

        #dim_recurrent, dim_input = Wahx.shape# dim_recurrent x dim_input tensor
        #dim_output = Wyh.shape[0]# dim_output x dim_recurrent tensor  
        self.fc_x2ah = nn.Linear(dim_input, dim_recurrent).to(config.device)# Wahx @ x + bah
        self.fc_h2ah = nn.Linear(dim_recurrent, dim_recurrent, bias = False).to(config.device)# Wahh @ h
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output).to(config.device)# y = Wyh @ h + by
        self.num_parameters = dim_recurrent ** 2 + dim_recurrent * dim_input + dim_recurrent + dim_output * dim_recurrent + dim_output# number of learned parameters in model
        self.dt = dt
        self.Tau = Tau
        #------------------------------
        # initialize the biases bah and by

        if ah0 is None:
            ah0 = torch.zeros(dim_recurrent).to(config.device)
        if bah is None:
            bah = torch.zeros(dim_recurrent).to(config.device)
        if by is None:
            by = torch.zeros(dim_output).to(config.device)
        if Wahh is None:
            # Saxe at al. 2014 "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
            # We empirically show that if we choose the initial weights in each layer to be a random orthogonal matrix (satisifying W'*W = I), instead of a scaled random Gaussian matrix, then this orthogonal random initialization condition yields depth independent learning times just like greedy layerwise pre-training.
            # [u,s,v] = svd(A); A = u*s*v’; columns of u are eigenvectors of covariance matrix A*A’; rows of v’ are eigenvectors of covariance matrix A’*A; s is a diagonal matrix that has elements = sqrt(eigenvalues of A’*A and A*A’)
            Wahh = np.random.randn(dim_recurrent, dim_recurrent)
            u, s, vT = np.linalg.svd(Wahh)  # np.linalg.svd returns v transpose!
            Wahh = u @ np.diag(1.0 * np.ones(dim_recurrent)) @ vT  # make the eigenvalues large so they decay slowly
            Wahh = torch.tensor(Wahh, dtype=torch.float32).to(config.device)
        if Wahx is None:
            # Sussillo et al. 2015 "A neural network that finds a naturalistic solution for the production of muscle activity"
            Wahx = torch.randn(dim_recurrent, dim_input).to(config.device) / np.sqrt(dim_input)
        if Wyh is None:
            # Wahh = 1.5 * torch.randn(dim_recurrent,dim_recurrent) / np.sqrt(dim_recurrent); initname = '_initWahhsussillo'
            Wyh = torch.zeros(dim_output, dim_recurrent).to(config.device)

        self.fc_x2ah.bias = torch.nn.Parameter(torch.squeeze(bah)).to(config.device)# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_h2y.bias = torch.nn.Parameter(torch.squeeze(by)).to(config.device)# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_x2ah.weight = torch.nn.Parameter(Wahx).to(config.device)# Wahx @ x + bah
        self.fc_h2ah.weight = torch.nn.Parameter(Wahh.to(config.device)).to(config.device)# Wahh @ h

        print(self.fc_h2ah.weight.get_device())

        self.fc_h2y.weight = torch.nn.Parameter(Wyh).to(config.device)# y = Wyh @ h + by
        self.ah0 = torch.nn.Parameter(ah0, requires_grad=LEARN_ah0).to(config.device)# (dim_recurrent,) tensor
        if LEARN_ah0:
            self.num_parameters = self.num_parameters + dim_recurrent# number of learned parameters in model

        self.fc_h2ah = self.fc_h2ah.to(config.device)

        #------------------------------
        # set the nonlinearity for h 
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        #self.nonlinearity = lambda x: f(x, nonlinearity)
        self.nonlinearity = nonlinearity

        self.to(config.device)

    def get_noise_shape(self, input):
        return (input.shape[0], input.shape[1], self.dim_recurrent)

    # output y for all numT timesteps   
    def _forward(self, input, bhneverlearn):# nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        if len(input.shape)==2:# if input has size (numT, dim_input) because there is only a single trial then add a singleton dimension
            input = input[None,:,:]# (numtrials, numT, dim_input)
            bhneverlearn = bhneverlearn[None,:,:]# (numtrials, numT, dim_recurrent)
        
        dt = self.dt
        Tau = self.Tau
        #numtrials, numT, dim_input = input.size()# METHOD 1
        numtrials, numT, dim_input = input.shape# METHOD 2
        #dim_recurrent = self.fc_h2y.weight.size(1)# y = Wyh @ h + by, METHOD 1
        #dim_recurrent = self.fc_h2y.weight.shape[1]# y = Wyh @ h + by, METHOD 2
        ah = self.ah0.repeat(numtrials, 1).to(config.device)# (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        #if self.LEARN_ah0:
        #    ah = self.ah0.repeat(numtrials, 1)# (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        #else:
        #    ah = input.new_zeros(numtrials, dim_recurrent)# tensor.new_zeros(size) returns a tensor of size size filled with 0. By default, the returned tensor has the same torch.dtype and torch.device as this tensor. 
        #h = self.nonlinearity(ah)# h0
        h = computef(ah, self.nonlinearity).to(config.device)# h0, this implementation doesn't add noise to h0
        hstore = []# (numtrials, numT, dim_recurrent)
        for t in range(numT):

            print(self.fc_h2ah.weight)
            #print(self.fc_h2ah.bias.get_device())
            #rint(self.fc_x2ah.weight.get_device())
            #print(self.fc_x2ah.bias.get_device())

            #print(self.fc_h2ah(h))
            #print(self.fc_x2ah(input[:,t]))

            self.fc_h2ah.weight = self.fc_h2ah.weight.to(config.device)
            print(self.fc_h2ah.weight)

            ah = ah + (dt/Tau) * (-ah + self.fc_h2ah(h) + self.fc_x2ah(input[:,t]))# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)
            #h = self.nonlinearity(ah)  +  bhneverlearn[:,t,:]# bhneverlearn has shape (numtrials, numT, dim_recurrent) 
            h = computef(ah, self.nonlinearity)  +  bhneverlearn[:,t,:]# bhneverlearn has shape (numtrials, numT, dim_recurrent) 
            hstore.append(h)# hstore += [h]
        hstore = torch.stack(hstore,dim=1)# (numtrials, numT, dim_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions  
        return self.fc_h2y(hstore), hstore# (numtrials, numT, dim_output/dim_recurrent) tensor, y = Wyh @ h + by
 



'''    
# A note on broadcasting:
# multiplying a (N,) array by a (M,N) matrix with * will broadcast element-wise
torch.manual_seed(123)# set random seed for reproducible results  
numtrials = 2  
Tau = torch.randn(5); Tau[-1] = 10
ah = torch.randn(numtrials,5)
A = ah + 1/Tau * (-ah)
A_check = -700*torch.ones(numtrials,5)
for i in range(numtrials):
    A_check[i,:] = ah[i,:] + 1/Tau * (-ah[i,:])# * performs elementwise multiplication
print(f"Do A and A_check have the same shape and are element-wise equal within a tolerance? {A.shape == A_check.shape and np.allclose(A, A_check)}")
'''

#%%##############################################################################
# low pass continuous time recurrent neural network
# Tau * dr/dt = -r + f(Wrr @ r + Wrx @ x + br) 
#
# r[t] = r[t-1] + (dt/Tau) * (-r[t-1] + f(Wrr @ r[t-1] + Wrx @ x[t] + br)  +  brneverlearn[t])
# y[t] = Wyr @ r[t] + by  output

# parameters to be learned: Wrr, Wrx, Wyr, br, by, r0(optional)
# constants that are not learned: dt, Tau, brneverlearn
# Equation 2 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
# "Note that equation 2 can be written Tau*dr/dt = -r + f(v). That is, if we regard v as a voltage 
# and f(v) as a firing rate, as suggested by the "derivation" in the appendix, then r is a low-pass-filtered version of the firing rate"
class LowPassCTRNN(nn.Module):# class LowPassCTRNN inherits from class torch.nn.Module
    def __init__(self, dim_input, dim_recurrent, dim_output, Wrx=None, Wrr=None, Wyr=None, br=None, by=None, nonlinearity='retanh', r0=None, LEARN_r0=False, dt=1, Tau=10):
        super().__init__()# super allows you to call methods of the superclass in your subclass
        #dim_recurrent, dim_input = Wrx.shape# dim_recurrent x dim_input tensor
        #dim_output = Wyr.shape[0]# dim_output x dim_recurrent tensor  
        self.fc_x2r = nn.Linear(dim_input, dim_recurrent)# Wrx @ x + br
        self.fc_r2r = nn.Linear(dim_recurrent, dim_recurrent, bias = False)# Wrr @ r
        self.fc_r2y = nn.Linear(dim_recurrent, dim_output)# y = Wyr @ r + by
        self.numparameters = dim_recurrent**2 + dim_recurrent*dim_input + dim_recurrent + dim_output*dim_recurrent + dim_output# number of learned parameters in model
        self.dt = dt
        self.Tau = Tau
        #------------------------------
        # initialize the biases br and by 
        if br is not None:
            self.fc_x2r.bias = torch.nn.Parameter(torch.squeeze(br))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        if by is not None:
            self.fc_r2y.bias = torch.nn.Parameter(torch.squeeze(by))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        #------------------------------
        # initialize input(Wrx), recurrent(Wrr), output(Wyr) weights 
        if Wrx is not None:
            self.fc_x2r.weight = torch.nn.Parameter(Wrx)# Wrx @ x + br
        if Wrr is not None:
            self.fc_r2r.weight = torch.nn.Parameter(Wrr)# Wrr @ r
        if Wyr is not None:
            self.fc_r2y.weight = torch.nn.Parameter(Wyr)# y = Wyr @ r + by
        #------------------------------
        # set the nonlinearity for r 
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        #self.nonlinearity = lambda x: f(x, nonlinearity)
        self.nonlinearity = nonlinearity
        #------------------------------
        # set the initial state r0
        if r0 is None:
            self.r0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=False)# (dim_recurrent,) tensor
        else:
            self.r0 = torch.nn.Parameter(r0, requires_grad=False)# (dim_recurrent,) tensor
        if LEARN_r0:
            #self.ah0 = self.ah0.requires_grad=True# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.r0 = torch.nn.Parameter(self.r0, requires_grad=True)# (dim_recurrent,) tensor
            self.numparameters = self.numparameters + dim_recurrent# number of learned parameters in model
        #------------------------------
        
        
    # output y for all numT timesteps   
    def forward(self, input, brneverlearn):# nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        if len(input.shape)==2:# if input has size (numT, dim_input) because there is only a single trial then add a singleton dimension
            input = input[None,:,:]# (numtrials, numT, dim_input)
            brneverlearn = brneverlearn[None,:,:]# (numtrials, numT, dim_recurrent)
        dt = self.dt
        Tau = self.Tau
        #numtrials, numT, dim_input = input.size()# METHOD 1
        numtrials, numT, dim_input = input.shape# METHOD 2
        #dim_recurrent = self.fc_r2y.weight.size(1)# y = Wyr @ r + by, METHOD 1
        #dim_recurrent = self.fc_r2y.weight.shape[1]# y = Wyr @ r + by, METHOD 2
        r = self.r0.repeat(numtrials, 1)# (numtrials, dim_recurrent) tensor, all trials should have the same initial value for r, not different values for each trial
        rstore = []# (numtrials, numT, dim_recurrent)
        for t in range(numT):
            r = r + (dt/Tau) * (-r + computef( self.fc_r2r(r) + self.fc_x2r(input[:, t]), self.nonlinearity)  + brneverlearn[:,t,:])# brneverlearn has shape (numtrials, numT, dim_recurrent) 
            rstore.append(r)# rstore += [r]
        rstore = torch.stack(rstore,dim=1)# (numtrials, numT, dim_recurrent), each appended r is stored in rstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions  
        return self.fc_r2y(rstore), rstore# (numtrials, numT, dim_output/dim_recurrent) tensor, y = Wyr @ r + by
 



#%%#-----------------------------------------------------------------------------
#                      compute specified nonlinearity 
#-----------------------------------------------------------------------------
def computef(IN,string,*args):# ags[0] is the slope for string='tanhwithslope'
    if string == 'linear':
        F = IN
        return F
    elif string == 'logistic':
        F = 1 / (1 + torch.exp(-IN))
        return F
    elif string == 'smoothReLU':# smoothReLU or softplus 
        F = torch.log(1 + torch.exp(IN))# always greater than zero  
        return F
    elif string == 'ReLU':# rectified linear units
        #F = torch.maximum(IN,torch.tensor(0))
        F = torch.clamp(IN, min=0)
        return F
    elif string == 'swish':# swish or SiLU (sigmoid linear unit)
        # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        # Elfwing et al. 2017 "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"
        # Ramachandran et al. 2017 "Searching for activation functions"
        sigmoid = 1/(1+torch.exp(-IN))
        F = torch.mul(IN,sigmoid)# x*sigmoid(x), torch.mul performs elementwise multiplication
        return F
    elif string == 'mish':# Misra 2019 "Mish: A Self Regularized Non-Monotonic Neural Activation Function
        F = torch.mul(IN, torch.tanh(torch.log(1+torch.exp(IN))))# torch.mul performs elementwise multiplication
        return F
    elif string == 'GELU':# Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        F = 0.5 * torch.mul(IN, (1 + torch.tanh(torch.sqrt(torch.tensor(2/np.pi))*(IN + 0.044715*IN**3))))# fast approximating used in original paper
        #F = x.*normcdf(x,0,1);% x.*normcdf(x,0,1)  =  x*0.5.*(1 + erf(x/sqrt(2)))
        #figure; hold on; x = linspace(-5,5,100); plot(x,x.*normcdf(x,0,1),'k-'); plot(x,0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3))),'r--')           
        return F
    elif string == 'ELU':# exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1
        inegativeIN = (IN < 0)
        F = IN.clone() 
        F[inegativeIN] = alpha * (torch.exp(IN[inegativeIN]) - 1) 
        return F
    elif string == 'tanh':
        F = torch.tanh(IN)
        return F
    elif string == 'tanhwithslope':
        a = args[0]
        F = torch.tanh(a*IN)# F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)  
        return F
    elif string == 'tanhlecun':# LeCun 1998 "Efficient Backprop" 
        F = 1.7159*torch.tanh(2/3*IN)# F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)  
        return F
    elif string == 'lineartanh':
        #F = torch.minimum(torch.maximum(IN,torch.tensor(-1)),torch.tensor(1))# -1(x<-1), x(-1<=x<=1), 1(x>1)
        F = torch.clamp(IN, min=-1, max=1)
        return F
    elif string == 'retanh':# rectified tanh
        F = torch.maximum(torch.tanh(IN),torch.tensor(0).to(config.device))
        return F
    elif string == 'binarymeanzero':# binary units with output values -1 and +1
        #F = (IN>=0) - (IN<0)# matlab code
        F = 1*(IN>=0) - 1*(IN<0)# multiplying by 1 converts True to 1 and False to 0
        return F
    else:
        print('Unknown transfer function type')
        


#-----------------------------------------------------------------------------
#    compute derivative of nonlinearity with respect to its input dF(x)/dx
#-----------------------------------------------------------------------------
def computedf(F,string,*args):# input has already been passed through nonlinearity, F = f(x). ags[0] is the slope for string='tanhwithslope'
    if string == 'linear':
        dFdx = torch.ones(F.shape)
        return dFdx
    elif string == 'logistic':
        dFdx = F - F**2# dfdx = f(x)-f(x).^2 = F-F.^2
        return dFdx
    elif string == 'smoothReLU':# smoothReLU or softplus 
        dFdx = 1 - torch.exp(-F)# dFdx = 1./(1 + exp(-x)) = 1 - exp(-F)
        return dFdx
    elif string == 'ReLU':# rectified linear units
        dFdx = 1.0*(F > 0)# F > 0 is the same as x > 0 for ReLU nonlinearity, multiplying by 1 converts True to 1 and False to 0, multiplying by 1.0 versus 1 makes dFdx a float versus an integer
        return dFdx
    elif string == 'ELU':# exponential linear units, Clevert et al. 2015 "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
        alpha = 1
        inegativex = (F < 0)# F < 0 is the same as x < 0 for ELU nonlinearity
        dFdx = torch.ones(F.shape); dFdx[inegativex] = F[inegativex] + alpha
        return dFdx
    elif string == 'tanh':
        dFdx = 1 - F**2# dfdx = 1-f(x).^2 = 1-F.^2
        return dFdx
    elif string == 'tanhwithslope':
        a = args[0]
        dFdx = a - a*(F**2)# F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)  
        return dFdx
    elif string == 'tanhlecun':# LeCun 1998 "Efficient Backprop"
        dFdx = 1.7159*2/3 - 2/3*(F**2)/1.7159# F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)
        return dFdx
    elif string == 'lineartanh':
        dFdx = 1*((F>-1) * (F<1))# 0(F<=-1), 1(-1<F<1), 0(F>=1), not quite right at x=-1 and x=1, * is elementwise multiplication
        return dFdx
    elif string == 'retanh':# rectified tanh
        dFdx = (1 - F**2) * (F > 0)# dfdx = 1-f(x).^2 = 1-F.^2,  * is elementwise multiplication
        return dFdx
    elif string == 'binarymeanzero':# binary units with output values -1 and +1
        dFdx = torch.zeros(F.shape)
        return dFdx
    else:
        print('Unknown transfer function type')




#%%##############################################################################
# gated recurrent unit (GRU)
# r[t] = sigmoid( Wrx @ x[t] + Wrh @ h[t-1] + br)                 reset gate
# z[t] = sigmoid( Wzx @ x[t] + Wzh @ h[t-1] + bz)                 update gate
# n[t] = tanh( Wnx @ x[t] + bn1  +  r[t] * (Wnh @ h[t-1] + bn2))  candidate activation / new gate
# h[t] = (1-z[t]) * n[t] + z[t] * h[t-1] + bhneverlearn           hidden state
# y[t] = Wyh @ h[t] + by                                          RNN output
# where * is the elementwise product
# The candidate hidden state (n) is computed according to the pytorch convention while tensorflow uses a slightly different convention. The differences likely don't matter. See footnote 1 of https://arxiv.org/pdf/1412.3555.pdf
class GRU(nn.Module):# class LSTM inherits from class torch.nn.Module 
    def __init__(self, dim_input, dim_recurrent, dim_output, LEARN_h0=False, nonlinearity='tanh', Wnh=None):
        super().__init__()# super allows you to call methods of the superclass in your subclass
        self.fc_r_x2r = nn.Linear(dim_input, dim_recurrent, bias=False)# Wrx @ x
        self.fc_r_h2r = nn.Linear(dim_recurrent, dim_recurrent)# Wrh @ h + br
        self.fc_z_x2z = nn.Linear(dim_input, dim_recurrent, bias=False)# Wzx @ x
        self.fc_z_h2z = nn.Linear(dim_recurrent, dim_recurrent)# Wzh @ h + bz
        self.fc_n_x2n = nn.Linear(dim_input, dim_recurrent)# Wnx @ x + bn1
        self.fc_n_h2n = nn.Linear(dim_recurrent, dim_recurrent)# Wnh @ h + bn2
        self.fc_y_h2y = nn.Linear(dim_recurrent, dim_output)# y = Wyh @ h + by
        self.numparameters = 2*(dim_recurrent*dim_input + dim_recurrent**2 + dim_recurrent) + (dim_recurrent*dim_input + dim_recurrent**2 + 2*dim_recurrent) + dim_output*dim_recurrent + dim_output# number of learned parameters in model
        #------------------------------
        # initialize recurrent weight matrix (Wzh)
        if Wnh is not None:
            self.fc_n_h2n.weight = torch.nn.Parameter(Wnh)# Wnh @ h + bn2
        #------------------------------
        # initialize the biases to be 0
        self.fc_r_h2r.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_z_h2z.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_n_x2n.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_n_h2n.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_y_h2y.bias = torch.nn.Parameter(torch.zeros(dim_output))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        #------------------------------
        self.LEARN_h0 = LEARN_h0
        if LEARN_h0:
            self.h0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=True)# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.numparameters = self.numparameters + dim_recurrent# number of learned parameters in model
        #------------------------------
        self.nonlinearity = nonlinearity
        
    # output y for all numT timesteps
    # nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
    def forward(self, input, bhneverlearn):# input has size (numtrials, numT, dim_input), nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        if len(input.shape)==2:# if input has size (numT, dim_input) because there is only a single trial then add a singleton dimension
            input = input[None,:,:]# (numtrials, numT, dim_input)
            bhneverlearn = bhneverlearn[None,:,:]# (numtrials, numT, dim_recurrent)
        
        numtrials, numT, dim_input = input.shape
        dim_recurrent = self.fc_r_x2r.weight.shape[0]# Wrx @ x
        if self.LEARN_h0:
            h = self.h0.repeat(numtrials, 1)# (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        else:
            h = input.new_zeros(numtrials, dim_recurrent)# (numtrials, dim_recurrent) tensor, tensor.new_zeros(size) returns a tensor of size size filled with 0. By default, the returned tensor has the same torch.dtype and torch.device as this tensor. 
        hstore = []# (numtrials, numT, dim_recurrent)
        for t in range(numT):# t=0,1,2,...,numT-1
            r = torch.sigmoid(self.fc_r_x2r(input[:, t]) + self.fc_r_h2r(h))# input[:,t].size() is (numtrials, dim_input)
            z = torch.sigmoid(self.fc_z_x2z(input[:, t]) + self.fc_z_h2z(h))# input[:,t].size() is (numtrials, dim_input)
            n = computef(self.fc_n_x2n(input[:, t]) + r * self.fc_n_h2n(h), self.nonlinearity)# input[:,t].size() is (numtrials, dim_input)
            h = (1-z) * n + z * h + bhneverlearn[:,t,:]# bhneverlearn has shape (numtrials, numT, dim_recurrent)
            hstore.append(h)# hstore += [h]
        hstore = torch.stack(hstore,dim=1)# (numtrials, numT, dim_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions  
        return self.fc_y_h2y(hstore), hstore# (numtrials, numT, dim_output/dim_recurrent) 
    
    # to get the output we could
    # 1) apply nn.Linear to h at each timestep (out = self.fc_y_h2y(h)) and then use output.append(out), followed by a final stack (output = torch.stack(output,dim=1)) to get the final output, or
    # 2) append and stack h across timesteps and then apply nn.Linear to hstore
    
        
        
#%%##############################################################################
# vanilla LSTM with no peephole connections
# i[t] = sigmoid(  Wih @ h[t−1] + 􏰨Wix @ x[t] + bi􏰩  )  input gate
# o[t] = sigmoid(  Woh @ h[t−1] + 􏰨Wox @ x[t] + bo􏰩  )  output gate
# f[t] = sigmoid(  Wfh @ h[t−1] + 􏰨Wfx @ x[t] + bf􏰩  )  forget gate
# z[t] =    tanh(  Wzh @ h[t−1] + 􏰨Wzx @ x[t] + bz􏰩  )  cell input/candidate cell values
# c[t] = i[t]*z[t] + f[t]*c[t-1]         cell state
# h[t] = o[t]*tanh(c[t]) + bhneverlearn  cell output
# y[t] = Wyh @ h[t] + by                 RNN output
class LSTM(nn.Module):# class LSTM inherits from class torch.nn.Module 
    def __init__(self, dim_input, dim_recurrent, dim_output, LEARN_c0h0=False, nonlinearity='tanh', Wzh=None):
        super().__init__()# super allows you to call methods of the superclass in your subclass
        self.fc_i_x2i = nn.Linear(dim_input, dim_recurrent)# Wix @ x + bi
        self.fc_i_h2i = nn.Linear(dim_recurrent, dim_recurrent, bias = False)# Wih @ h
        self.fc_o_x2o = nn.Linear(dim_input, dim_recurrent)# Wox @ x + bo
        self.fc_o_h2o = nn.Linear(dim_recurrent, dim_recurrent, bias = False)# Woh @ h
        self.fc_f_x2f = nn.Linear(dim_input, dim_recurrent)# Wfx @ x + bf
        self.fc_f_h2f = nn.Linear(dim_recurrent, dim_recurrent, bias = False)# Wfh @ h
        self.fc_z_x2z = nn.Linear(dim_input, dim_recurrent)# Wzx @ x + bz
        self.fc_z_h2z = nn.Linear(dim_recurrent, dim_recurrent, bias = False)# Wzh @ h
        self.fc_y_h2y = nn.Linear(dim_recurrent, dim_output)# y = Wyh @ h + by
        self.numparameters = 4*(dim_recurrent**2 + dim_recurrent*dim_input + dim_recurrent) + dim_output*dim_recurrent + dim_output# number of learned parameters in model
        #------------------------------
        # initialize recurrent weight matrix (Wzh)
        if Wzh is not None:
            self.fc_z_h2z.weight = torch.nn.Parameter(Wzh)# Wzh @ h
        #------------------------------
        # initialize the biases to be 0
        self.fc_o_x2o.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_z_x2z.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_y_h2y.bias = torch.nn.Parameter(torch.zeros(dim_output))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        # initialize the forget gate bias (bf) to be 1 and the input gate bias (bi) to be -1
        # Jozefowicz et al. 2015 "An Empirical Exploration of Recurrent Network Architectures"
        # Gers et al. 2000 "Learning to Forget: Continual Prediction with LSTM"
        #self.fc_f_x2f.bias = torch.nn.Parameter(torch.ones(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        #self.fc_i_x2i.bias = torch.nn.Parameter(-1*torch.ones(dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        #tmin = 1; tmax = 100; self.fc_f_x2f.bias = torch.nn.Parameter(torch.log((tmax-tmin)*torch.rand(dim_recurrent)+tmin))# Tallec & Ollivier 2018 "Can recurrent neural networks warp time?"
        #tmin = -1; tmax = -100; self.fc_i_x2i.bias = torch.nn.Parameter(torch.log((tmax-tmin)*torch.rand(dim_recurrent)+tmin))# Tallec & Ollivier 2018 "Can recurrent neural networks warp time?"
        self.fc_f_x2f.bias = torch.nn.Parameter(torch.linspace(1,10,dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        self.fc_i_x2i.bias = torch.nn.Parameter(torch.linspace(-1,-10,dim_recurrent))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        
        # CAN DELETE: To replicate neural data the firing rates should decay. So initialize the forget gate bias (bf) to be 0 and the input gate bias (bi) to be 0. This is probably not a good idea computationally but might help the model look more like neural data.
        #self.fc_f_x2f.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))
        #self.fc_i_x2i.bias = torch.nn.Parameter(torch.zeros(dim_recurrent))
        #------------------------------
        self.LEARN_c0h0 = LEARN_c0h0
        if LEARN_c0h0:
            self.c0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=True)# learn initial value for c, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.h0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=True)# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.numparameters = self.numparameters + 2*dim_recurrent# number of learned parameters in model
        #------------------------------
        # set the nonlinearity for h (standard for LSTM is tanh(c[t])) so unit activity of h better matches neural data
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        '''
        if nonlinearity == 'tanh': 
            self.nonlinearity = torch.nn.Tanh()
        elif nonlinearity == 'retanh':
            class retanh(nn.Module):# https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa
                def __init__(self):
                    super().__init__()
                def forward(self, input):
                    #return torch.nn.functional.relu(torch.tanh(input))# method 1
                    return input.tanh().clamp(min=0)# method 2   
            self.nonlinearity = retanh()
        '''
        '''
        if nonlinearity == 'tanh':
            self.nonlinearity = lambda x: torch.tanh(x)
        elif nonlinearity == 'retanh':
            #self.nonlinearity = lambda x: torch.nn.functional.relu(torch.tanh(x))# method 1
            self.nonlinearity = lambda x: x.tanh().clamp(min=0)# method 2
        '''
        self.nonlinearity = nonlinearity
        #------------------------------
        # initialize recurrent weight matrix 
        #W = torch.randn(dim_recurrent,dim_recurrent)/torch.sqrt(torch.tensor(dim_recurrent))
        #U, S, Vh = torch.linalg.svd(W)
        #W = U @ torch.diag(torch.ones(S.size(0))) @ Vh
        #self.fc_h2h.weight = torch.nn.Parameter(W)
        

    # output y for all numT timesteps
    # nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
    def forward(self, input, bhneverlearn):# input has size (numtrials, numT, dim_input), nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        if len(input.shape)==2:# if input has size (numT, dim_input) because there is only a single trial then add a singleton dimension
            input = input[None,:,:]# (numtrials, numT, dim_input)
            bhneverlearn = bhneverlearn[None,:,:]# (numtrials, numT, dim_recurrent)
            
        #numtrials, numT, dim_input = input.size()# METHOD 1
        numtrials, numT, dim_input = input.shape# METHOD 2
        #dim_recurrent = self.fc_i_x2i.weight.size(0)# Wix @ x + bi, METHOD 1
        dim_recurrent = self.fc_i_x2i.weight.shape[0]# Wix @ x + bi, METHOD 2
        if self.LEARN_c0h0:
            c = self.c0.repeat(numtrials, 1)# (numtrials, dim_recurrent) tensor, all trials should have the same initial value for c, not different values for each trial
            h = self.h0.repeat(numtrials, 1)# (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        else:
            c = input.new_zeros(numtrials, dim_recurrent)# (numtrials, dim_recurrent) tensor, tensor.new_zeros(size) returns a tensor of size size filled with 0. By default, the returned tensor has the same torch.dtype and torch.device as this tensor. 
            h = input.new_zeros(numtrials, dim_recurrent)# (numtrials, dim_recurrent) tensor, tensor.new_zeros(size) returns a tensor of size size filled with 0. By default, the returned tensor has the same torch.dtype and torch.device as this tensor. 
        hstore = []# (numtrials, numT, dim_recurrent)
        for t in range(numT):# t=0,1,2,...,numT-1
            i = torch.sigmoid(self.fc_i_x2i(input[:, t]) + self.fc_i_h2i(h))# input[:,t].size() is (numtrials, dim_input)
            o = torch.sigmoid(self.fc_o_x2o(input[:, t]) + self.fc_o_h2o(h))# input[:,t].size() is (numtrials, dim_input)
            f = torch.sigmoid(self.fc_f_x2f(input[:, t]) + self.fc_f_h2f(h))# input[:,t].size() is (numtrials, dim_input)
            z =    torch.tanh(self.fc_z_x2z(input[:, t]) + self.fc_z_h2z(h))# input[:,t].size() is (numtrials, dim_input)
            c = i*z + f*c
            #h = o*torch.tanh(c)
            #h = o*self.nonlinearity(c)  +  self.noiseamplitude*torch.randn(numtrials,dim_recurrent)
            #h = o*self.nonlinearity(c) + bhneverlearn[:,t,:]# bhneverlearn has shape (numtrials, numT, dim_recurrent)
            h = o*computef(c, self.nonlinearity) + bhneverlearn[:,t,:]# bhneverlearn has shape (numtrials, numT, dim_recurrent)
            hstore.append(h)# hstore += [h]
        hstore = torch.stack(hstore,dim=1)# (numtrials, numT, dim_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions  
        return self.fc_y_h2y(hstore), hstore# (numtrials, numT, dim_output/dim_recurrent) 
    
    # to get the output we could
    # 1) apply nn.Linear to h at each timestep (out = self.fc_y_h2y(h)) and then use output.append(out), followed by a final stack (output = torch.stack(output,dim=1)) to get the final output, or
    # 2) append and stack h across timesteps and then apply nn.Linear to hstore