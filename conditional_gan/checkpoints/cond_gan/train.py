#!/usr/bin/env python

import sys
import os
from shutil import copyfile
import time
import itertools
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms


from options import TrainOptions, create_sub_dirs
from dataloader import DataLoader, load_numpy_data, AlignedIterator, UnalignedIterator
from model import StochCycleGAN, AugmentedCycleGAN, StochCondGan
import numpy as np
from evaluate import eval_mse_A, eval_ubo_B, one_to_three_channels
import shutil
import random
import glob
import json

from tensorboardX import SummaryWriter
from collections import OrderedDict
import torchvision

import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['axes.facecolor'] = 'white'

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)

from skimage.transform import resize



oceanMask = resize(np.load("OceanMask.npy"), (64, 64))

def save_results(expr_dir, results_dict):
    # save to results.json (for cluster exp)
    fname = os.path.join(expr_dir, 'results.json')
    with open(fname, 'w') as f:
        json.dump(results_dict, f, indent=4)


def copy_scripts_to_folder(expr_dir):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in glob.glob("%s/*.py" % dir_path):
        shutil.copy(f, expr_dir)

def print_log(out_f, message):
    out_f.write(message+"\n")
    out_f.flush()
    print(message)

def format_log(epoch, i, errors, t, prefix=True):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message

def tbviz_cycle(opt, real_A, real_B, gen_AB, gen_BA):
    size = real_A.size()


def visualize_cycle(opt, real_A, visuals, eidx, uidx, train, writer):
    size = real_A.size()

    images = [one_to_three_channels(img.cpu()).unsqueeze(1) for img in visuals.values()]
    vis_image = torch.cat(images, dim=1).view(size[0]*len(images),size[1],size[2],size[3])
    if train:
        save_path = opt.train_vis_cycle
    else:
        save_path = opt.vis_cycle
    save_path = os.path.join(save_path, 'cycle_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=len(images))

    copyfile(save_path, os.path.join(opt.vis_latest, 'cycle.png'))
    return vis_image

def visualize_multi(opt, real_A, model, eidx, uidx):
    size = real_A.size()
    # all samples in real_A share the same prior_z_B
    multi_prior_z_B = Variable(real_A.data.new(opt.num_multi,
        opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0],1,1,1), volatile=True)
    multi_fake_B = model.generate_multi(real_A.detach(), multi_prior_z_B)
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, 1, size[2], size[3])
    print(real_A.shape, multi_fake_B.shape, real_A.data.cpu().unsqueeze(1).shape, multi_prior_z_B.shape)
    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(2), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+3),1,size[2],size[3])
    save_path = os.path.join(opt.vis_multi, 'multi_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)
    copyfile(save_path, os.path.join(opt.vis_latest, 'multi.png'))

def visualize_inference(opt, real_A, real_B, model, eidx, uidx):
    size = real_A.size()

    real_B = real_B[:opt.num_multi]
    # all samples in real_A share the same post_z_B
    multi_fake_B = model.inference_multi(real_A.detach(), real_B.detach())
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, 1, size[2], size[3])

    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(2), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+3),size[1],size[2],size[3])

    vis_multi_image = torch.cat([torch.ones(1, size[1], size[2], size[3]).cpu(), real_B.data.cpu(),
                                 vis_multi_image.cpu()], dim=0)

    save_path = os.path.join(opt.vis_inf, 'inf_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)
    copyfile(save_path, os.path.join(opt.vis_latest, 'inf.png'))

#From https://martin-mundt.com/tensorboard-figures/
def plot_to_tensorboard(writer, name, fig, step):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()
    fig.savefig('temp.png', dpi=fig.dpi)

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))


    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    #print(img)

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(name, img, step)
    plt.close(fig)

def hist_kl_div(tenA, tenB, numbins):
    min_elem = min(torch.min(tenA).item(), torch.min(tenB).item())
    max_elem = max(torch.max(tenA).item(), torch.max(tenB).item())
    print(min_elem, max_elem)
    histA = torch.histc(tenA.cpu(), numbins, min_elem, max_elem)
    histB = torch.histc(tenB.cpu(), numbins, min_elem, max_elem)
    eps = 1e-4
    histA = (histA+eps)*1.0/torch.sum(histA)
    histB = (histB+eps)*1.0/torch.sum(histB)
    return torch.sum(histA * (torch.log(histA) - torch.log(histB)))






def take_two_weeks(input_data, start_date):
    return input_data[start_date:start_date + 14, :, :, :]


#normalizing data
def arr_to_input(arr):
    arr = np.nan_to_num(arr)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=2)
    means = np.nanmean(arr, axis=(0,1,2))
    # scale and shift to [-1,1]
    arr = -1 + 2 * (arr - arr.min((1,2))[:, np.newaxis,np.newaxis]) / (arr.max((1,2))[:, np.newaxis,np.newaxis] - arr.min((1,2))[:, np.newaxis,np.newaxis])
    arr[np.isnan(arr)] = -1; arr[arr == np.inf] = -1; arr[arr == -np.inf] = -1
    if 64 is not None:
        new_arr = []
        #print "Resizing data to %d" % 64
        for x in arr:
            new_arr.append(resize(x, (64,64)))
        arr = np.stack(new_arr)
    # convert data from b,0,1,c to b,c,0,1
    arr = np.transpose(arr, (0,3,1,2))
    
    return arr.astype('float32')


def numpy_histograms(start_date, inputA, inputB, zero_mask, num_bins, model, to_plot = True):
    fake_B = model.generate_fwd(torch.from_numpy(arr_to_input(take_two_weeks(inputA, start_date))[:,0:3,:,:]).cuda()).cpu()
    real_B = torch.from_numpy(arr_to_input(take_two_weeks(inputB, start_date)))
    curr_day = 0
    #print(fake_B.shape, real_B.shape)
    
    
    mask = np.ravel(np.tile(zero_mask, (real_B.shape[0], 1, 1, 1)))
    
    real_im = np.ravel(real_B.detach().numpy())
    fake_im = np.ravel(fake_B.detach().numpy())

    

    
    real_im = real_im[~np.isnan(real_im)]
    fake_im = fake_im[~np.isnan(fake_im)]

    #print(np.sum(real_im), np.sum(fake_im))
    #print(np.mean(real_im))

    
    
    #
    
    #got everything setup, now make bins and counts
    
    real_hist_data = np.histogram(real_im, bins=num_bins)
    fake_hist_data = np.histogram(fake_im, bins=num_bins)
    
    if to_plot:
        fix, ax = plt.subplots(figsize=(10, 10))
        ax.plot(real_hist_data[1][1:], real_hist_data[0], linestyle = '--', c = '0.0', marker='s',  linewidth=3.0, label = 'Real Data')
        ax.plot(fake_hist_data[1][1:], fake_hist_data[0], marker = 'o', c = 'g', linewidth = 1.0, label = 'Generated Data')
        ax.legend(loc='upper right')
        title = "Histogram of data binned from " + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14))
        ax.set_title(title)
        ax.set_yscale('log')
        ax.set_xlabel('Normalized Pixel Values')
        ax.set_ylabel('Log Counts')


    
    return real_hist_data, fake_hist_data

def take_n_samples_random(data_A, data_B, n):
    image_indices = np.random.choice(804, n)
    return data_A[image_indices, :,:, :], data_B[image_indices, :,:, :]


def nov_aug_apr_hist(model, inputA, inputB, mask, step, writer, name):
    starting_day = datetime.date(1900, 1, 1) + datetime.timedelta(days = 37620 + 4018 - 804)
    
    
    f = plt.figure(figsize=(30,40))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(4, 3)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    ax6 = plt.subplot(gs[1, 2])
    
    ax7 = plt.subplot(gs[2, 0:3])
    ax8 = plt.subplot(gs[3, 0:3])
    
    
    start_date = 300
    real_hist_data, fake_hist_data =  numpy_histograms(start_date, inputA, inputB, mask, 100, model, False)
    ax1.plot(real_hist_data[1][1:], real_hist_data[0], linestyle = '--', c = '0.0', marker='s',  linewidth=3.0, label = 'Real Data')
    ax1.plot(fake_hist_data[1][1:], fake_hist_data[0], marker = 'o', c = 'g', linewidth = 1.0, label = 'Generated Data')
    ax1.legend(loc='upper right')
    title = "Histogram of data binned from " + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14))
    ax1.set_title(title)
    ax1.set_yscale('log')
    ax1.set_xlabel('Normalized Pixel Values')
    ax1.set_ylabel('Log Counts')
    
    temp = ax4.matshow(np.nan_to_num(pearson_coeff(start_date, inputA, inputB, model))[0,:,:], cmap='gray')
    plt.colorbar(temp, ax = ax4, boundaries=np.linspace(-1,1,101))
    ax4.set_title("Correlation Coeffs over " + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14)))

    
    start_date = 400
    real_hist_data, fake_hist_data =  numpy_histograms(start_date, inputA, inputB, mask, 100, model, False)
    ax2.plot(real_hist_data[1][1:], real_hist_data[0], linestyle = '--', c = '0.0', marker='s',  linewidth=3.0, label = 'Real Data')
    ax2.plot(fake_hist_data[1][1:], fake_hist_data[0], marker = 'o', c = 'g', linewidth = 1.0, label = 'Generated Data')
    ax2.legend(loc='upper right')
    title = "Histogram of data binned from " + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14))
    ax2.set_title(title)
    ax2.set_yscale('log')
    ax2.set_xlabel('Normalized Pixel Values')
    ax2.set_ylabel('Log Counts')
    
    temp = ax5.matshow(np.nan_to_num(pearson_coeff(start_date, inputA, inputB, model))[0,:,:], cmap = 'gray')
    plt.colorbar(temp, ax = ax5, boundaries=np.linspace(-1,1,101))
    ax5.set_title("Correlation Coeffs over "  + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14)))


    
    start_date = 530
    real_hist_data, fake_hist_data =  numpy_histograms(start_date, inputA, inputB, mask, 100, model, False)
    ax3.plot(real_hist_data[1][1:], real_hist_data[0], linestyle = '--', c = '0.0', marker='s',  linewidth=3.0, label = 'Real Data')
    ax3.plot(fake_hist_data[1][1:], fake_hist_data[0], marker = 'o', c = 'g', linewidth = 1.0, label = 'Generated Data')
    ax3.legend(loc='upper right')
    title = "Histogram of data binned from " + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14))
    ax3.set_title(title)
    ax3.set_yscale('log')
    ax3.set_xlabel('Normalized Pixel Values')
    ax3.set_ylabel('Log Counts')
    
    
    temp = ax6.matshow(np.nan_to_num(pearson_coeff(start_date, inputA, inputB, model))[0,:,:], cmap = 'gray')
    plt.colorbar(temp, ax = ax6, boundaries=np.linspace(-1,1,101))
    ax6.set_title("Correlation Coeffs over "  + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14)))
    

    num_samples = 10
    
    sampled_input, sampled_output_real = take_n_samples_random(inputA, inputB, num_samples)
    
    sampled_input = torch.from_numpy(arr_to_input(sampled_input)[:,0:3,:,:])
    
    sampled_output_fake = model.generate_fwd(sampled_input.cuda()).cpu()
    sampled_output_real = torch.from_numpy(arr_to_input(sampled_output_real))
    #print(sampled_input.shape, sampled_output_fake.shape, sampled_output_real.shape)

    
    
    img = np.hstack([image for image in sampled_input.detach().numpy().reshape(num_samples, 64*3, 64)])
    dx, dy = 64,64

    # Custom (rgb) grid color
    grid_color = 1

    # Modify the image to include the grid
    img[:,::dy] = grid_color
    img[::dx,:] = grid_color
    
    ax7.imshow(img)
    ax7.set_ylabel("INPUT CHANNELS: TMIN, TMAX, PRECIPITATION")

    
    
    
    sampled_outputs = torch.cat((sampled_output_fake, sampled_output_real, torch.abs(sampled_output_fake - sampled_output_real)), 1)
    img = np.hstack([image for image in sampled_outputs.detach().numpy().reshape(num_samples, 64*3, 64)])

    

    # Modify the image to include the grid
    img[:,::dy] = grid_color
    img[::dx,:] = grid_color

    
    
    ax8.imshow(img)
    ax8.set_ylabel("OUTPUTS: ERROR, REAL, GENERATED")
    
    plot_to_tensorboard(writer, name, f, step)


def pearson_coeff(start_date, inputA, inputB, model):
    fakeB = model.generate_fwd(torch.from_numpy(arr_to_input(take_two_weeks(inputA, start_date))[:,0:3,:,:]).cuda()).detach().cpu().numpy()
    realB = torch.from_numpy(arr_to_input(take_two_weeks(inputB, start_date))).numpy()
    
    xbar = np.mean(fakeB, axis=0)
    ybar = np.mean(realB, axis=0)
    xSTDev = np.std(fakeB, axis=0)
    ySTDev = np.std(realB, axis=0)
    r = np.mean(((fakeB - xbar)/(xSTDev)) * ((realB - ybar)/(ySTDev)), axis = 0)
    return r



def generate_nice_visualization_graph():
    '''unfinished, don't use '''
    
    #visualize current training batch
    visualize_cycle(opt, real_A, visuals, epoch, epoch_iter/opt.batchSize, train=True)

    sizeT = real_A.size()

    viz_data = viz_dataset.next()
    viz_real_A, viz_real_B = Variable(viz_data['A']), Variable(viz_data['B'])
    viz_prior_z_B = Variable(viz_real_A.data.new(viz_real_A.size(0), opt.nlatent, 1, 1).normal_(0, 1))
    if use_gpu:
        viz_real_A = viz_real_A.cuda()
        viz_real_B = viz_real_B.cuda()
        viz_prior_z_B = viz_prior_z_B.cuda()

    viz_visuals = model.generate_visuals(viz_real_A, viz_real_B, viz_prior_z_B)


    # # visuals = OrderedDict([('real_A', real_A.data), ('fake_B', fake_B.data), ('rec_A', rec_A.data),
    # #                        ('real_B', real_B.data), ('fake_A', fake_A.data), ('rec_B', rec_B.data)])
    #
    #
    viz_real_A = viz_visuals['real_A']
    viz_real_B = viz_visuals['real_B']
    viz_fake_B = viz_visuals['fake_B']
    # viz_rec_A = viz_visuals['rec_A']
    # viz_fake_A = viz_visuals['fake_A']
    # viz_rec_B = viz_visuals['rec_B']
    #
    for index in range(0, viz_visuals['real_A'].shape[0]):
        viz_real_A = viz_visuals['real_A'][index]
        viz_real_B = viz_visuals['real_B'][index]
        viz_fake_B = viz_visuals['fake_B'][index]
    #     viz_rec_A = viz_visuals['rec_A'][index]
    #     viz_fake_A = viz_visuals['fake_A'][index]
    #     viz_rec_B = viz_visuals['rec_B'][index]
    #
    #
    #
    #
    #
        viz_real_A = viz_real_A.view(-1, 3*viz_real_A.shape[1], viz_real_A.shape[2])
    #     viz_fake_A = viz_fake_A.view(-1, 3*viz_fake_A.shape[1], viz_fake_A.shape[2])
    #     viz_rec_A = viz_rec_A.view(-1, 3*viz_rec_A.shape[1], viz_rec_A.shape[2])
    #
    #     #print(viz_real_A.shape, viz_fake_A.shape, viz_rec_A.shape, viz_real_B.shape, viz_fake_B.shape, viz_rec_B.shape)
    #
    #
    #
        viz_cGen = nx.Graph()
        viz_cGen.add_node('realA', image = viz_real_A)
        viz_cGen.add_node('realB', image = viz_real_B)
        viz_cGen.add_node('fakeB', image = viz_fake_B)
    #
    #
    #
        viz_cGen.add_edge('realA','realB', r= 'temp1')
        viz_cGen.add_edge('realB', 'fakeB', r= 'temp2')
        edge_labels = nx.get_edge_attributes(viz_cGen,'r')
        pos={'realA': np.array([-1, 0]), 'realB': np.array([0, 0]), 'fakeB': np.array([ 1,0 ])}

        fig=plt.figure(figsize=(15,10))
        ax=plt.subplot(111, title="Generated visuals from cGan")
        ax.set_aspect('auto')
        nx.draw_networkx_edges(viz_cGen,pos,ax=ax)
        #nx.draw_networkx_edge_labels(viz_cGen, pos, edge_labels = edge_labels)


        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)

        trans=ax.transData.transform
        trans2=fig.transFigure.inverted().transform


        for n in viz_cGen:

            piesize=(0.135/viz_cGen.node[n]['image'].shape[2])*viz_cGen.node[n]['image'].shape[1]
            p2=piesize/2.0
            xx,yy=trans(pos[n]) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            a = plt.axes([xa-p2,ya-p2, piesize, piesize], xlabel = n)
            #a.title = "hi"
            #a.set_aspect('equal')
            image = viz_cGen.node[n]['image'].cpu().clone()
            image = image.squeeze(0).numpy()
            #print(image.shape)

            #image = np.random.rand(64,64)
            #image = unloader(image)
            #print(viz_ABA.node[n]['image'][0].shape)
            #print(image.shape)
            #savestr = "temp" + str(total_steps+index) + ".npy"
            #np.save(savestr, image)
            a.imshow(image, cmap='gray')
            #print(G.node[n]['image'].shape)
        #fig.savefig("temp1.png")
        #plt.imshow(image)
        #plt.show()
        plot_to_tensorboard(writer, "Generated Conditional Visuals pix2pix", fig, total_steps)
        #model.train()





def train_model():
    opt = TrainOptions().parse(sub_dirs=['vis_multi','vis_cycle','vis_latest','train_vis_cycle'])
    writer = SummaryWriter(opt.tbpath)
    out_f = open("%s/results.txt" % opt.expr_dir, 'w')
    copy_scripts_to_folder(opt.expr_dir)
    use_gpu = len(opt.gpu_ids) > 0

    if opt.seed is not None:
        print("using random seed:", opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(opt.seed)

    if opt.numpy_data:
        trainA, trainB, devA, devB, testA, testB = load_numpy_data(opt.dataroot, grid_size=opt.grid_size)
        train_dataset = AlignedIterator(trainA, trainB, batch_size=opt.batchSize)

        #viz_dataset = AlignedIterator(devA, devB, batch_size = 1)

        print_log(out_f, '#training images = %d' % len(train_dataset))
        vis_inf = False

        test_dataset = AlignedIterator(testA, testB, batch_size=100)
        print_log(out_f, '#test images = %d' % len(test_dataset))

        # dev_dataset = AlignedIterator(devA, devB, batch_size=100)
        # print_log(out_f, '#dev images = %d' % len(dev_dataset))

        dev_cycle = itertools.cycle(AlignedIterator(devA, devB, batch_size=25))
    else:
        train_data_loader = DataLoader(opt, subset='train', unaligned=False, batchSize=opt.batchSize)
        test_data_loader = DataLoader(opt, subset='val', unaligned=False, batchSize=200)
        dev_data_loader = DataLoader(opt, subset='dev', unaligned=False, batchSize=200)
        dev_cycle_loader = DataLoader(opt, subset='dev', unaligned=False, batchSize=25)

        train_dataset = train_data_loader.load_data()
        dataset_size = len(train_data_loader)
        print_log(out_f, '#training images = %d' % dataset_size)
        vis_inf = False

        test_dataset = test_data_loader.load_data()
        print_log(out_f, '#test images = %d' % len(test_data_loader))

        dev_dataset = dev_data_loader.load_data()
        print_log(out_f, '#dev images = %d' % len(dev_data_loader))

        dev_cycle = itertools.cycle(dev_cycle_loader.load_data())

    if opt.supervised:
        if opt.numpy_data:
            sup_size = int(len(trainA) * opt.sup_frac)
            sup_trainA = trainA[:sup_size]
            sup_trainB = trainB[:sup_size]
            sup_train_dataset = AlignedIterator(sup_trainA, sup_trainB, batch_size=opt.batchSize)
        else:
            sup_train_data_loader = DataLoader(opt, subset='train', unaligned=False,
                                               batchSize=opt.batchSize, fraction=opt.sup_frac)
            sup_train_dataset = sup_train_data_loader.load_data()
            sup_size = len(sup_train_data_loader)
        sup_train_dataset = itertools.cycle(sup_train_dataset)
        print_log(out_f, '#supervised images = %d' % sup_size)

    # create_model
    # if opt.model == 'stoch_cycle_gan':
    #     model = StochCycleGAN(opt)
    # elif opt.model == 'cycle_gan':
    #     model = StochCycleGAN(opt, ignore_noise=True)
    # elif opt.model == 'aug_cycle_gan':
    #     model = AugmentedCycleGAN(opt)
    #     create_sub_dirs(opt, ['vis_inf'])
    #     vis_inf = True
    # elif opt.model == 'cond_gan':
    #     model = StochCondGan(opt)
    # else:
    #     raise NotImplementedError('Specified model is not implemented.')

    model = StochCondGan(opt)

    print_log(out_f, "model [%s] was created" % (model.__class__.__name__))

    # visualizer = Visualizer(opt)
    total_steps = 0
    print_start_time = time.time()
    results = {
        'best_dev_mse_A'  : sys.float_info.max,
        'best_test_mse_A' : sys.float_info.max,
        'best_dev_bpp_B'  : sys.float_info.max,
        'best_test_bpp_B' : sys.float_info.max,
    }
    save_results(opt.expr_dir, results)
    history_mse_A = []
    history_ubo_B = []

    create_sub_dirs(opt, ['vis_pred_B'])
    unloader = transforms.ToPILImage()

    data_A = np.load('../datasets/livneh/testA.npz')['data']
    data_B = np.load('../datasets/livneh/testB.npz')['data']
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataset):
            real_A, real_B = Variable(data['A']), Variable(data['B'])
            if real_A.size(0) != real_B.size(0):
                continue
            prior_z_B = Variable(real_A.data.new(real_A.size(0), opt.nlatent, 1, 1).normal_(0, 1))

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            if use_gpu:
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                prior_z_B = prior_z_B.cuda()

            # print data['A'].size(), data['B'].size()

            # if opt.monitor_gnorm:
            #     losses, visuals, gnorms = model.train_instance(real_A, real_B, prior_z_B)
            # else:
            losses, visuals = model.train_instance(real_A, real_B, prior_z_B)
            for key in losses:
                temp_str = key + " loss"
                writer.add_scalar(temp_str, losses[key], total_steps)




            # supervised trainings hare tensr
            if opt.supervised:
                sup_data = sup_train_dataset.next()
                sup_real_A, sup_real_B = Variable(sup_data['A']), Variable(sup_data['B'])
                if use_gpu:
                    sup_real_A, sup_real_B = sup_real_A.cuda(), sup_real_B.cuda()
                sup_losses = model.supervised_train_instance(sup_real_A, sup_real_B, prior_z_B)

            if total_steps % opt.display_freq == 0:
                print("sending images to tensorboard")
                model.eval()



                nov_aug_apr_hist(model, data_A, data_B, oceanMask, total_steps, writer, "Dashboard Generated")



                model.train()

               
            if total_steps % opt.print_freq == 0:


                t = (time.time() - print_start_time) / opt.batchSize
                print_log(out_f, format_log(epoch, epoch_iter, losses, t))
                if opt.supervised:
                    print_log(out_f, format_log(epoch, epoch_iter, sup_losses, t, prefix=False))
                print_start_time = time.time()

        #now we're gonna get some histograms ready


        if epoch % opt.save_epoch_freq == 0:
            print_log(out_f, 'saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
            model.save('latest')

        #####################
        # evaluate mappings
        #####################
        # if epoch % opt.eval_A_freq == 0:
        #     t = time.time()
        #     dev_mse_A = eval_mse_A(dev_dataset, model)
        #     test_mse_A = eval_mse_A(test_dataset, model)
        #     t = time.time() - t
        #     history_mse_A.append((dev_mse_A, test_mse_A))
        #     np.save("%s/history_mse_A" % opt.expr_dir, history_mse_A)
        #
        #     res_str_list = ["[%d] DEV_MSE_A: %.4f, TEST_MSE_A: %.4f, TIME: %.4f" % (epoch, dev_mse_A, test_mse_A, t)]
        #     if dev_mse_A < results['best_dev_mse_A']:
        #         with open("%s/best_mse_A.txt" % opt.expr_dir, 'w') as best_mse_A_f:
        #             best_mse_A_f.write(res_str_list[0]+'\n')
        #             best_mse_A_f.flush()
        #         results['best_dev_mse_A'] = dev_mse_A
        #         results['best_test_mse_A'] = test_mse_A
        #         model.save('best_A')
        #         save_results(opt.expr_dir, results)
        #         res_str_list += ["*** BEST DEV A ***"]
        #     res_str = "\n".join(["-"*60] + res_str_list + ["-"*60])
        #     print_log(out_f, res_str)

        # if epoch % opt.eval_B_freq == 0:
        #     t = time.time()
        #     if opt.model == 'cycle_gan':
        #         steps = 1
        #     else:
        #         steps = 50
        #     dev_ubo_B, dev_bpp_B, dev_kld_B = eval_ubo_B(dev_dataset, model, steps, True, 'pred_B_%d' % epoch,
        #                                                  opt.vis_pred_B)
        #     test_ubo_B, test_bpp_B, test_kld_B = eval_ubo_B(test_dataset, model, steps, False, 'pred_B',
        #                                                     opt.vis_pred_B)
        #     t = time.time() - t
        #     history_ubo_B.append((dev_ubo_B, dev_bpp_B, dev_kld_B, test_ubo_B, test_bpp_B, test_kld_B))
        #     np.save("%s/history_ubo_B" % opt.expr_dir, history_ubo_B)
        #
        #     res_str_list = ["[%d] DEV_BPP_B: %.4f, TEST_BPP_B: %.4f, TIME: %.4f" % (epoch, dev_bpp_B, test_bpp_B, t)]
        #     if dev_bpp_B < results['best_dev_bpp_B']:
        #         with open("%s/best_bpp_B.txt" % opt.expr_dir, 'w') as best_bpp_B_f:
        #             best_bpp_B_f.write(res_str_list[0]+'\n')
        #             best_bpp_B_f.flush()
        #         results['best_dev_bpp_B'] = dev_bpp_B
        #         results['best_test_bpp_B'] = test_bpp_B
        #         save_results(opt.expr_dir, results)
        #         model.save('best_B')
        #         res_str_list += ["*** BEST BPP B ***"]
        #     res_str = "\n".join(["-"*60] + res_str_list + ["-"*60])
        #     print_log(out_f, res_str)
        #
        # print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
        #           (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()

    out_f.close()


if __name__ == "__main__":
    train_model()
