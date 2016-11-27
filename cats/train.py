import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from load import cats

def transform(X):
    X = [center_crop(x, npx) for x in X]
    return floatX(X).transpose(0, 3, 1, 2)/127.5 - 1.

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
    return X

k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 25        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 11230     # # of examples to train on
total_n_layers = 6 # total number of layers in the original GAN
n_transfer     = 6 # number of layers to transfer from pretrained model

tr_data, te_data, tr_stream, val_stream, te_stream = cats(ntrain=ntrain)

tr_handle = tr_data.open()
vaX, = tr_data.get_data(tr_handle, slice(0, 10000))
vaX = transform(vaX)

desc = 'uncond_dcgan'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

# load pretrained model
model_path = '../models/imagenet_gan_pretrain_128f_relu_lrelu_7l_3x3_256z/'
gen_params = [sharedX(p) for p in joblib.load(model_path+'30_gen_params.jl')]
discrim_params = [sharedX(p) for p in joblib.load(model_path+'30_discrim_params.jl')]

def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*4, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(1, 1), border_mode=(1, 1)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4))
    h5 = relu(batchnorm(deconv(h4, w5, subsample=(1, 1), border_mode=(1, 1)), g=g5, b=b5))
    h6 = relu(batchnorm(deconv(h5, w6, subsample=(2, 2), border_mode=(1, 1)), g=g6, b=b6))
    x = tanh(deconv(h6, wx, subsample=(1, 1), border_mode=(1, 1)))
    return x

def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wy):
    h = lrelu(dnn_conv(X, w, subsample=(1, 1), border_mode=(1, 1)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(1, 1), border_mode=(1, 1)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4))
    h5 = lrelu(batchnorm(dnn_conv(h4, w5, subsample=(1, 1), border_mode=(1, 1)), g=g5, b=b5))
    h6 = lrelu(batchnorm(dnn_conv(h5, w6, subsample=(2, 2), border_mode=(1, 1)), g=g6, b=b6))
    h6 = T.flatten(h6, 2)
    y = sigmoid(T.dot(h6, wy))
    return y

def gen_samples(n, nbatch=128):
    samples = []
    n_gen = 0
    for i in range(n/nbatch):
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        xmb = _gen(zmb)
        samples.append(xmb)
        n_gen += len(xmb)
    n_left = n-n_gen
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
    xmb = _gen(zmb)
    samples.append(xmb)    
    return np.concatenate(samples, axis=0)

if n_transfer < total_n_layers:
    # functions to initialize layers
    gifn = inits.Normal(scale=0.02)
    difn = inits.Normal(scale=0.02)
    gain_ifn = inits.Normal(loc=1., scale=0.02)
    bias_ifn = inits.Constant(c=0.)

    g_prev_dim = d_prev_dim = 0

    for i in range(n_transfer, total_n_layers):
        # special case if retraining entire network and iteration is on first layer
        if i == 0:
            # Generator layers:
            dim = ngf*8*4*4
            gen_params[0]   = gifn((nz, dim), 'gw')
            gen_params[1] = gain_ifn((dim), 'gg')
            gen_params[2] = bias_ifn((dim), 'gb')

            # Discriminator layers:
            discrim_params[0] = difn((ndf/4, nc, 5, 5), 'dw')

        else:
            # Generator layers:
            g_prev_dim           = 2**(12-i-1)
            dim                  = g_prev_dim/2
            g_indx               = i*3
            gen_params[g_indx]   = gifn((g_prev_dim, dim, 5, 5), 'gw'+str(i+1))
            gen_params[g_indx+1] = gain_ifn((dim), 'gg'+str(i+1))
            gen_params[g_indx+2] = bias_ifn((dim), 'gb'+str(i+1))

            # Discriminator layers:
            dim                      = 2**(i+5)
            d_prev_dim               = dim/2
            d_indx                   = (i-1)*2+i
            discrim_params[d_indx]   = difn((dim, d_prev_dim, 5, 5), 'dw'+str(i+1))
            discrim_params[d_indx+1] = gain_ifn((dim), 'dg'+str(i+1))
            discrim_params[d_indx+2] = bias_ifn((dim), 'db'+str(i+1))

    # If training, all layers are trained. If pretraining, the last layer must at least be retrained, so leave this outside of the loop.
    gen_params[18]     = gifn((ngf/4, nc, 5, 5), 'gwx')
    discrim_params[16] = difn((d_prev_dim*2*4*4, 1), 'dwy')

    print str(n_transfer)+' layers transferred.'

    X = T.tensor4()
    Z = T.matrix()

    gX = gen(Z, *gen_params)

    p_real = discrim(X, *discrim_params)
    p_gen = discrim(gX, *discrim_params)

    d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
    g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d

    cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

    lrt = sharedX(lr)
    d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    d_updates = d_updater(discrim_params, d_cost)
    g_updates = g_updater(gen_params, g_cost)
    updates = d_updates + g_updates

    print 'COMPILING'
    t = time()
    _train_g = theano.function([X, Z], cost, updates=g_updates)
    _train_d = theano.function([X, Z], cost, updates=d_updates)
    _gen = theano.function([Z], gX)
    print '%.2f seconds to compile theano functions'%(time()-t)

    vis_idxs = py_rng.sample(np.arange(len(vaX)), nvis)
    vaX_vis = inverse_transform(vaX[vis_idxs])
    color_grid_vis(vaX_vis, (14, 14), 'samples/%s_etl_test.png'%desc)

    sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))

    f_log = open('logs/%s.ndjson'%desc, 'wb')
    log_fields = [
        'n_epochs', 
        'n_updates', 
        'n_examples', 
        'n_seconds',
        '1k_va_nnd',
        '10k_va_nnd',
        '100k_va_nnd',
        'g_cost',
        'd_cost',
    ]

    vaX = vaX.reshape(len(vaX), -1)

    print desc.upper()
    n_updates = 0
    n_check = 0
    n_epochs = 0
    n_updates = 0
    n_examples = 0
    t = time()
    for epoch in range(niter):
        for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain/nbatch):
            imb = transform(imb)
            zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
            if n_updates % (k+1) == 0:
                cost = _train_g(imb, zmb)
            else:
                cost = _train_d(imb, zmb)
            n_updates += 1
            n_examples += len(imb)
        g_cost = float(cost[0])
        d_cost = float(cost[1])
        gX = gen_samples(100000)
        gX = gX.reshape(len(gX), -1)
        va_nnd_1k = nnd_score(gX[:1000], vaX, metric='euclidean')
        va_nnd_10k = nnd_score(gX[:10000], vaX, metric='euclidean')
        va_nnd_100k = nnd_score(gX[:100000], vaX, metric='euclidean')
        log = [n_epochs, n_updates, n_examples, time()-t, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost]
        print '%.0f %.2f %.2f %.2f %.4f %.4f'%(epoch, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost)
        f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
        f_log.flush()

        samples = np.asarray(_gen(sample_zmb))
        color_grid_vis(inverse_transform(samples), (14, 14), 'samples/%s/%d.png'%(desc, n_epochs))
        n_epochs += 1
        if n_epochs > niter:
            lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        if n_epochs in [1, 2, 3, 4, 5, 10, 15, 20, 25]:
            joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
            joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))
else:
    print 'All layers were transferred. No retraining required.'