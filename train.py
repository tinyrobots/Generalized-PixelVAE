"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

use_ui = False
if use_ui:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf
import scipy.misc
from skimage import transform, io, img_as_ubyte
import pandas as pd

import pixel_cnn_pp.nn as nn
from pixel_cnn_pp.model import model_spec, model_spec_encoder

# added by KS
from tensorflow import logging
logging.set_verbosity(logging.DEBUG)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# ################################################################################
# # COMMONLY SET PARAMETERS
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be cifar|imagenet|bumpworld')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint?')
parser.add_argument('-ow', '--overwrite_saves', default=True, help='Whether to overwrite checkpoints, or save separately.')
parser.add_argument('-mc', '--generate_mc_interval', type=int, default=100, help='Every how many epochs to generate MC samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model') # KS: Or does this mean "number of layers per residual block"??
parser.add_argument('-n', '--nr_filters', type=int, default=120, help='Number of filters to use across the generative model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=8, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-ae', '--use_autoencoder', default=True, dest='use_autoencoder', action='store_true', help='Use autoencoders?')
parser.add_argument('-reg', '--reg_type', type=str, default='elbo', help='Type of regularization to use for autoencoder')
parser.add_argument('-cs', '--chain_step', type=int, default=5, help='Steps to run Markov chain for sampling')
parser.add_argument('-ld', '--latent_dim', type=int, default=20, help='Dimension of latent code')
parser.add_argument('-enc', '--encoder_nr_filters', type=int, default=64, help='Number of filters in first layer of encoder (scales thereafter)')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=24, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=80, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-gid', '--gpus', type=str, default=os.environ["CUDA_VISIBLE_DEVICES"], help='Which GPUs to use')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# bumpworld data processing options
parser.add_argument('-im', '--im_size', type=int, default=64, help='Image width/height to use (square images).')
parser.add_argument('-ms', '--max_scene', type=int, default=9000, help='Scene numbers 0-max_scene will be used.')
parser.add_argument('-ts', '--test_scene', type=int, default=8500, help='Scene number at which to switch from train to test images.')
args = parser.parse_args(args=[])
args.nr_gpu = len(args.gpus.split(','))

# KS custom options
args.load_params = True

args.im_size = 128
args.nr_resnet = 3
args.nr_filters = 64
args.encoder_nr_filters = 64
args.nr_logistic_mix = 12
args.latent_dim = 10
args.batch_size = 5
args.reg_type = 'no_reg'

args.max_epochs = 200
args.save_interval = 1 # how often to save parameters and samples
args.overwrite_saves = False # False => save separate checkpoints so that we can revert (memory intensive)

# lr_factor = 1
decay_factor = 1.0
args.learning_rate = 0.001
args.lr_decay = 1-decay_factor*(args.learning_rate/args.max_epochs)

args.max_scene = 9500
args.test_scene = 9000

# args.save_dir = './continuous1Dgloss_10k_Oct2020/filters-{}_resnets-{}_logmixes-{}_lr-{}_decay-{}_01'.format(args.nr_filters, args.nr_resnet, args.nr_logistic_mix, args.learning_rate, decay_factor)
args.save_dir = './model_set_Feb2019/filters-64_resnets-3_logmixes-12_latents-10_repeat-00/'
args.data_set = 'bumpworld'
img_dir = '../../../data/original_bimodal10k/'
args.generate_mc_interval = 1000 # i.e. basically never do these (very time consuming)
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':'))) # pretty print args

# unpack some args into variables
desired_im_sz = (args.im_size, args.im_size, 3) # Square RGB images with pixel size divisible by 8, up to max real size of 800x800 for Bumpworld
scene_data = range(0,args.max_scene) # how many/which of your images you want to use here (CIFAR = 50k train)
train_cutoff = args.test_scene # index at which to switch from training to validation images
print('DEBUG: finished setting arguments.')

# ################################################################################

tic = time.time()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# save arguments of this trained net (important to do at start, so can work w unfinished net)
args_df = pd.DataFrame.from_dict(vars(args), orient='index')
args_df.to_csv(os.path.join(args.save_dir,'args_df.csv'))

# also save the details about image size and training data not in args
params_dict = {}
params_dict['im_size'] = desired_im_sz[0]
params_dict['first_train_scene'] = min(scene_data)
params_dict['last_train_scene'] = train_cutoff-1
params_dict['first_val_scene'] = train_cutoff
params_dict['last_val_scene'] = max(scene_data)
params_df = pd.DataFrame.from_dict(params_dict, orient='index')
params_df.to_csv(os.path.join(args.save_dir,'params_df.csv'))

# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if args.use_autoencoder:
            feed_dict.update({encoder_x_init: x})
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if args.use_autoencoder:
            feed_dict.update({encoder_x[i]: x[i] for i in range(args.nr_gpu)})
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

# -----------------------------------------------------------------------------

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
if args.data_set == 'imagenet' or args.data_set == 'cifar':
    DataLoader = {'cifar':cifar10_data.DataLoader, 'imagenet':imagenet_data.DataLoader}[args.data_set]
    train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
    test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
    obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
elif args.data_set == 'bumpworld':
    obs_shape = desired_im_sz
    # Load my BUMPWORLD images at the specified resolution
    splits = {'scene': scene_data}
    im_dir = os.path.join(img_dir, 'rgb/')
    _, _, all_frames = os.walk(im_dir).next() # warning: .next() doesn't work in Python3
    all_frames = sorted(all_frames)
    for split in splits:
        im_list = all_frames[splits[split][0] : splits[split][len(splits[split])-1]+1]
        source_list = [str(i) for i in splits[split]]
        X = np.zeros((len(im_list),) + desired_im_sz, np.float32) # was uint8?
        for i, im_file in enumerate(im_list):
            im = io.imread(os.path.join(im_dir, im_file)) # loads as uint8 (0-255 integers)
            im = transform.resize(im[:,:,0:-1], desired_im_sz) # explicitly cut off alpha channel here, or get weird colour behav.
            im = np.uint8(255*im) # the resize function changes dtype to float64 in range [0, 1] - not OK for make_feed_dict!
            X[i] = im

    print(X.shape)
    train_data = X[:train_cutoff, :, :]
    test_data = X[train_cutoff:, :, :]

assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]
encoder_x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
encoder_x = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = train_data.get_num_labels()
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
    h_init = tf.one_hot(y_init, num_labels)
    y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), num_labels), args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
elif args.use_autoencoder:
    # h_init = tf.placeholder(tf.float32, shape=(args.init_batch_size, args.latent_dim)) # commented out in source
    h_sample = [tf.placeholder(tf.float32, shape=(args.batch_size, args.latent_dim)) for i in range(args.nr_gpu)]
else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample

# create the model
model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity}
model = tf.make_template('model', model_spec)
if args.use_autoencoder:
    encoder_opt = {'reg_type': args.reg_type, 'latent_dim': args.latent_dim, 'encoder_nr_filters': args.encoder_nr_filters}
    encoder_model = tf.make_template('encoder', model_spec_encoder)

print('DEBUG: finished loading data and making template for models.')

# run once for data dependent initialization of parameters
if args.use_autoencoder:
    encoder = encoder_model(encoder_x_init, init=True, dropout_p=args.dropout_p, **encoder_opt)
    gen_par = model(x_init, encoder.pred, init=True, dropout_p=args.dropout_p, **model_opt)
else:
    gen_par = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

print('DEBUG: finished data-dependent initialisation.')

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params] # KS: re-added for compatibility w fast-pixel-cnn sampling

# get loss gradients over multiple GPUs
grads = []
loss_gen = []
loss_gen_reg = []
loss_gen_elbo = []
loss_gen_test = []
encoder_preds = []
decoder_preds = [] # added by KS
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        if args.use_autoencoder:
            print('DEBUG: inside training code.')
            encoder = encoder_model(encoder_x[i], ema=None, dropout_p=args.dropout_p, **encoder_opt)
            gen_par = model(xs[i], encoder.pred, ema=None, dropout_p=args.dropout_p, **model_opt)
            loss_gen_reg.append(encoder.reg_loss)
            loss_gen_elbo.append(encoder.elbo_loss)
            encoder_preds.append(encoder.pred)
            decoder_preds.append(gen_par) # added by KS
        else:
            gen_par = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))
        # gradients
        if args.use_autoencoder:
            print('DEBUG: calculating loss.')
            total_loss = loss_gen[i] + loss_gen_reg[i]
        else:
            total_loss = loss_gen[i]
        grads.append(tf.gradients(total_loss, all_params))
        # test
        if args.use_autoencoder:
            print('DEBUG: inside test code.')
            encoder = encoder_model(encoder_x[i], ema=ema, dropout_p=0., **encoder_opt)
            gen_par = model(xs[i], encoder.pred, ema=ema, dropout_p=0., **model_opt)
        else:
            gen_par = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    encoder_pred = tf.concat(values=encoder_preds, axis=0)
    decoder_pred = tf.concat(values=decoder_preds, axis=0) # added by KS
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        if args.use_autoencoder:
            loss_gen_reg[0] += loss_gen_reg[i]
            loss_gen_elbo[0] += loss_gen_elbo[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    tf.summary.scalar('ll_loss', loss_gen[0])
    if args.use_autoencoder:
        tf.summary.scalar('reg', loss_gen_reg[0])
        tf.summary.scalar('elbo', loss_gen_elbo[0])
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
tf.summary.scalar('ll_bits_per_dim', bits_per_dim)

# sample from the model
new_x_gen = []
encoder_list = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        if args.use_autoencoder:
            encoder = encoder_model(encoder_x[i], ema=ema, dropout_p=0, **encoder_opt)
            gen_par = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
            encoder_list.append(encoder)
        else:
            gen_par = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix))


def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)


def sample_from_prior(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    latent_code = [np.random.normal(size=(args.batch_size, args.latent_dim)) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            feed_dict = {xs[i]: x_gen[i] for i in range(args.nr_gpu)}
            feed_dict.update({h_sample[i]: latent_code[i] for i in range(args.nr_gpu)})
            new_x_gen_np = sess.run(new_x_gen, feed_dict)
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)


def sample_from_markov_chain(sess, initial=None):
    history = []
    if initial is None:
        encoder_current = [np.random.uniform(0.0, 1.0, (args.batch_size,) + obs_shape) for i in range(args.nr_gpu)]
    else:
        encoder_current = np.split(initial, args.nr_gpu)
    latent_op = [encoder.pred for encoder in encoder_list]
    num_steps = args.chain_step
    history.append(np.concatenate(encoder_current, axis=0))

    for step in range(num_steps):
        start_time = time.time()
        feed_dict = {encoder_x[i]: encoder_current[i] for i in range(args.nr_gpu)}
        latent_code = sess.run(latent_op, feed_dict)

        x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
        for yi in range(obs_shape[0]):
            for xi in range(obs_shape[1]):
                feed_dict = {xs[i]: x_gen[i] for i in range(args.nr_gpu)}
                feed_dict.update({h_sample[i]: latent_code[i] for i in range(args.nr_gpu)})
                new_x_gen_np = sess.run(new_x_gen, feed_dict)
                for i in range(args.nr_gpu):
                    x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
        history.append(np.concatenate(x_gen, axis=0))
        encoder_current = x_gen
        print("%d (%fs)" % (step, time.time() - start_time))
        sys.stdout.flush()
    return history


def plot_markov_chain(history):
    canvas = np.zeros((args.nr_gpu*args.batch_size*obs_shape[0], len(history)*obs_shape[1], obs_shape[2]))
    for i in range(args.nr_gpu*args.batch_size):
        for j in range(len(history)):
            canvas[i*obs_shape[0]:(i+1)*obs_shape[0], j*obs_shape[1]:(j+1)*obs_shape[1], :] = history[j][i]
    print(np.min(canvas), np.max(canvas))
    return canvas


def plot_img(images, num_img):
    canvas = np.zeros((num_img*obs_shape[0], num_img*obs_shape[1], obs_shape[2]))
    for i in range(num_img):
        for j in range(num_img):
            canvas[i*obs_shape[0]:(i+1)*obs_shape[0], j*obs_shape[1]:(j+1)*obs_shape[1], :] = images[i*num_img+j]
    print(np.min(canvas), np.max(canvas))
    return canvas

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()
all_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir=args.save_dir)

# //////////// perform training //////////////
print('starting training')
test_bpd = []
lr = args.learning_rate
global_step = 0

#fig, ax = plt.subplots()
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    for epoch in range(args.max_epochs):
        # init
        if epoch == 0:
            if args.data_set == 'bumpworld':
                # just load initialisation batch
                feed_dict = make_feed_dict(train_data[0:args.init_batch_size], init=True)
            else:
                feed_dict = make_feed_dict(train_data.next(args.init_batch_size), init=True) # manually retrieve exactly init_batch_size examples
                train_data.reset()  # rewind the iterator back to 0 to do one full epoch
            sess.run(initializer, feed_dict)
            print('initializing the model...')
            if args.load_params:
                ckpt_file = args.save_dir + 'params_' + args.data_set + '_' + args.max_epochs + '.ckpt'
                print('restoring parameters from', ckpt_file)
                try:
                    saver.restore(sess, ckpt_file)
                except:
                    print("Error: Restore file failed")

        # generate samples from the model
        if args.use_autoencoder and (epoch + 1) % args.generate_mc_interval == 0:
            print("Generating MC")
            start_time = time.time()
            initial = np.random.uniform(0.0, 1.0, (args.batch_size * args.nr_gpu,) + obs_shape)
            sample_history = sample_from_markov_chain(sess, initial)
            initial = sample_history[-1]
            sample_plot = plot_markov_chain(sample_history)
            scipy.misc.imsave(os.path.join(args.save_dir, '%s_mc%d.png' % (args.data_set, epoch)), img_as_ubyte(sample_plot))
            print("Finished, time elapsed %fs" % (time.time() - start_time))

        # generate samples from the model
        if (epoch+1) % args.save_interval == 0:
            print("Generating samples")
            start_time = time.time()
            if args.use_autoencoder:
                sample_x = sample_from_prior(sess)
            else:
                sample_x = sample_from_model(sess)
            img_tile = plot_img(sample_x, int(np.floor(np.sqrt(args.batch_size * args.nr_gpu))))
            scipy.misc.imsave(os.path.join(args.save_dir, "%s_ancestral%d.png" % (args.data_set, epoch)), img_as_ubyte(img_tile))
            print("Finished, time elapsed %fs" % (time.time() - start_time))

        begin = time.time()
        # train for one epoch
        train_losses = []
        batch_c = 10
        latents = []
        # hacky split because haven't worked out how to put own data as DataLoader object
        if args.data_set == 'bumpworld':
            batch_inds = range(0,len(train_data),args.batch_size * args.nr_gpu)
            for b in range(len(batch_inds)-1):
                feed_dict = make_feed_dict(train_data[batch_inds[b]:batch_inds[b+1]])
                # forward/backward/update model on each gpu
                lr *= args.lr_decay
                feed_dict.update({ tf_lr: lr })
                l, _, summaries, latent = sess.run([bits_per_dim, optimizer, all_summary, encoder_pred], feed_dict)
                if len(latents) < 10:
                    latents.append(latent)
                train_losses.append(l)
                if global_step % 5 == 0:
                    writer.add_summary(summaries, global_step)
                global_step += 1
        else:
            for d in train_data:
                feed_dict = make_feed_dict(d)
                # forward/backward/update model on each gpu
                lr *= args.lr_decay
                feed_dict.update({ tf_lr: lr })
                l, _, summaries, latent = sess.run([bits_per_dim, optimizer, all_summary, encoder_pred], feed_dict)
                if len(latents) < 10:
                    latents.append(latent)
                train_losses.append(l)
                if global_step % 5 == 0:
                    writer.add_summary(summaries, global_step)
                global_step += 1
        train_loss_gen = np.mean(train_losses)
        latent = np.concatenate(latents, axis=0)

        if use_ui:
            ax.cla()
            ax.scatter(latent[:, 0], latent[:, 1])
            plt.draw()
            plt.savefig(os.path.join(args.save_dir, 'latent.png'))
            plt.pause(0.5)

        # compute likelihood over test data
        test_losses = []
        # hacky split because haven't worked out how to put own data as DataLoader object
        if args.data_set == 'bumpworld':
            batch_inds = range(0,len(test_data),args.batch_size * args.nr_gpu)
            for b in range(len(batch_inds)-1):
                feed_dict = make_feed_dict(test_data[batch_inds[b]:batch_inds[b+1]])
                l = sess.run(bits_per_dim_test, feed_dict)
                test_losses.append(l)
        else:
            for d in test_data:
                feed_dict = make_feed_dict(d)
                l = sess.run(bits_per_dim_test, feed_dict)
                test_losses.append(l)
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if (epoch % args.save_interval == 0) or ((epoch+1) == args.max_epochs):
            if args.overwrite_saves:
                saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt') # save and resave params under same name
            else:
                saver.save(sess, args.save_dir + '/params_' + args.data_set + '_' + str(epoch+1).zfill(4) + '.ckpt') # save params under a new name each time
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd)) # save up-to-date training log


print('FINISHED. Total training time = {} hours'.format((time.time()-tic)/3600))
