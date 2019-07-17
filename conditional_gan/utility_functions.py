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

from scipy.signal import convolve
import cv2


plt.rcParams['axes.facecolor'] = 'white'

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)

from skimage.transform import resize, downscale_local_mean



oceanMask = resize(np.load("OceanMask.npy"), (64, 64))
snotel_data = dict(np.load("snotel_values_and_index.npz", allow_pickle=True))
snotel_names = snotel_data['arr_0']
snotel_values = snotel_data['arr_1'][:,1]
snotel_indices = snotel_data['arr_1'][:,0]

valid_snotel = []
for index in range(0, 32):
    if snotel_values[index].shape[0] == 4018:
        #print(snotel_names[index], index)
        valid_snotel.append(index)



def normalize_by_peak(swe):
    return swe / np.max(swe)

def zero_one_norm_peak(swe):
    temp = 1 + swe
    return temp / np.max(temp)





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
    for k, v in list(errors.items()):
        message += '%s: %.3f ' % (k, v)
    return message

def tbviz_cycle(opt, real_A, real_B, gen_AB, gen_BA):
    size = real_A.size()


def visualize_cycle(opt, real_A, visuals, eidx, uidx, train, writer):
    size = real_A.size()

    images = [one_to_three_channels(img.cpu()).unsqueeze(1) for img in list(visuals.values())]
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
    print((real_A.shape, multi_fake_B.shape, real_A.data.cpu().unsqueeze(1).shape, multi_prior_z_B.shape))
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
    print((min_elem, max_elem))
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
    fake_B = model.generate_fwd(torch.from_numpy(arr_to_input(take_two_weeks(inputA, start_date))[:,0:6,:,:]).cuda()).cpu()
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
    range_min = min(np.min(real_im), np.min(fake_im))
    range_max = max(np.max(real_im), np.max(fake_im))
    real_hist_data = np.histogram(real_im, num_bins, (range_min, range_max))
    fake_hist_data = np.histogram(fake_im, num_bins, (range_min, range_max))

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


def plot_snotel_data(model, inputA, inputB, mask, step, writer, name, snotel):
    starting_day = datetime.date(1900, 1, 1) + datetime.timedelta(days = 37620 + 4018 - 804)
    f = plt.figure(figsize=(30,20))
    f.patch.set_facecolor('white')

def plot_station_vs_sim(model, data_A, data_B, mask, step, writer, name, index):
    f = plt.figure(figsize=(30,10))
    model_output = model.generate_fwd(torch.from_numpy(arr_to_input(data_A)).cuda()).cpu().detach().numpy()
    #print(snotel_names[index])
    plt.plot(normalize_by_peak(snotel_values[index][3214:]), 'g')
    plt.plot(normalize_by_peak(data_B[:, int(snotel_indices[index][0]*64/321), int(snotel_indices[index][1]*64/321), 0]), 'b')
    plt.plot(zero_one_norm_peak(model_output[:, 0, int(snotel_indices[index][0]*64/321), int(snotel_indices[index][1]*64/321)]), 'r:')

    plot_to_tensorboard(writer, name, f, step)


def plot_snotel_bar_chart(model, data_A, data_B, mask, step, writer, name, start, num):
    start_day = start
    end_day = start_day + num
    f = plt.figure(figsize=(100,10))


    model_output = model.generate_fwd(torch.from_numpy(arr_to_input(data_A)).cuda()).cpu().detach().numpy()
    n_groups = len(valid_snotel)
    vals_snotel = [np.mean(normalize_by_peak(snotel_values[index][3214:])[start_day:end_day]) for index in valid_snotel]
    vals_livneh = [np.mean(normalize_by_peak(data_B[:, int(snotel_indices[index][0]*64/321), int(snotel_indices[index][1]*64/321), 0])[start_day:end_day]) for index in valid_snotel]
    vals_gan = [np.mean(zero_one_norm_peak(model_output[:, 0, int(snotel_indices[index][0]*64/321), int(snotel_indices[index][1]*64/321)])[start_day:end_day]) for index in valid_snotel]
    # create plot
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, vals_snotel, bar_width,alpha=opacity,color='b',label='Snotel')
    rects2 = plt.bar(index + bar_width, vals_livneh, bar_width,alpha=opacity,color='g',label='Livneh')
    rects3 = plt.bar(index + 2*bar_width, vals_gan, bar_width,alpha=opacity,color='r',label='GAN')

    plt.xlabel('Stations')
    plt.ylabel('Normalized SWE')
    plt.title('snotel v livneh v gan')
    plt.xticks(index + bar_width, snotel_names[valid_snotel])
    plt.legend()

    plt.tight_layout()
    plot_to_tensorboard(writer, name, f, step)

def plot_total_snotel_bar_chart(model, data_A, data_B, mask, step, writer, name, start, num):
    start_day = 0
    end_day = 804
    f = plt.figure(figsize=(10,10))


    model_output = model.generate_fwd(torch.from_numpy(arr_to_input(data_A)).cuda()).cpu().detach().numpy()
    n_groups = 1
    vals_snotel = sum([np.mean(normalize_by_peak(snotel_values[index][3214:])[start_day:end_day]) for index in valid_snotel])
    vals_livneh = sum([np.mean(normalize_by_peak(data_B[:, int(snotel_indices[index][0]*64/321), int(snotel_indices[index][1]*64/321), 0])[start_day:end_day]) for index in valid_snotel])
    vals_gan = sum([np.mean(zero_one_norm_peak(model_output[:, 0, int(snotel_indices[index][0]*64/321), int(snotel_indices[index][1]*64/321)])[start_day:end_day]) for index in valid_snotel])
    # create plot
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, vals_snotel, bar_width,alpha=opacity,color='b',label='Snotel')
    rects2 = plt.bar(index + bar_width, vals_livneh, bar_width,alpha=opacity,color='g',label='Livneh')
    rects3 = plt.bar(index + 2*bar_width, vals_gan, bar_width,alpha=opacity,color='r',label='GAN')

    plt.xlabel('Stations')
    plt.ylabel('Normalized SWE')
    plt.title('snotel v livneh v gan')
    plt.xticks(index + bar_width, snotel_names[valid_snotel])
    plt.legend()

    plt.tight_layout()
    plot_to_tensorboard(writer, name, f, step)



def plot_tensorboard_histograms_corrcoeff(model, inputA, inputB, mask, step, writer, name):
    starting_day = datetime.date(1900, 1, 1) + datetime.timedelta(days = 37620 + 4018 - 804)
    f = plt.figure(figsize=(30,20))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])

    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    ax6 = plt.subplot(gs[1, 2])

    start_date = 300
    real_hist_data, fake_hist_data =  numpy_histograms(start_date, inputA, inputB, mask, 100, model, False)
    ax1.plot(real_hist_data[1][1:], real_hist_data[0], linestyle = '--', c = '0.0', marker='s',  linewidth=3.0, label = 'Real Data')
    ax1.plot(fake_hist_data[1][1:], fake_hist_data[0], marker = 'o', c = 'g', linewidth = 1.0, label = 'Generated Data')
    ax1.legend(loc='upper right')
    title = "Histogram of data binned from " + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14))
    if step % 1 == 0: log_metrics_np_hist(real_hist_data, fake_hist_data, "August Histogram", writer, step)


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
    if step % 1 == 0: log_metrics_np_hist(real_hist_data, fake_hist_data, "November Histogram", writer, step)

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
    if step % 1 == 0: log_metrics_np_hist(real_hist_data, fake_hist_data, "April Histogram", writer, step)

    ax3.set_title(title)
    ax3.set_yscale('log')
    ax3.set_xlabel('Normalized Pixel Values')
    ax3.set_ylabel('Log Counts')


    temp = ax6.matshow(np.nan_to_num(pearson_coeff(start_date, inputA, inputB, model))[0,:,:], cmap = 'gray')
    plt.colorbar(temp, ax = ax6, boundaries=np.linspace(-1,1,101))
    ax6.set_title("Correlation Coeffs over "  + str(starting_day + datetime.timedelta(days = start_date)) + " to " + str(starting_day + datetime.timedelta(days = start_date + 14)))


    plot_to_tensorboard(writer, name, f, step)



def plot_images_to_tensorboard(model, inputA, inputB, mask, step, writer, name):
    starting_day = datetime.date(1900, 1, 1) + datetime.timedelta(days = 37620 + 4018 - 804)
    f = plt.figure(figsize=(30,40))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(6, 3)


    ax7 = plt.subplot(gs[0:2, 0:3])
    ax8 = plt.subplot(gs[3, 0:3])

    ax9 = plt.subplot(gs[4, 0:3])

    ax10 = plt.subplot(gs[5, 0:3])




    num_samples = 10

    sampled_input, sampled_output_real = take_n_samples_random(inputA, inputB, num_samples)
    sampled_inp_old = sampled_input
    sampled_input = torch.from_numpy(arr_to_input(sampled_input)[:,0:6,:,:])

    sampled_output_fake = model.generate_fwd(sampled_input.cuda()).cpu()
    sampled_output_real = torch.from_numpy(arr_to_input(sampled_output_real))
    #print(sampled_input.shape, sampled_output_fake.shape, sampled_output_real.shape)



    img = np.hstack([image for image in sampled_input.detach().numpy().reshape(num_samples, 64*6, 64)])
    dx, dy = 64,64

    # Custom (rgb) grid color
    grid_color = 1

    # Modify the image to include the grid
    img[:,::dy] = grid_color
    img[::dx,:] = grid_color

    temp = ax7.imshow(img)
    plt.colorbar(temp, ax = ax7)
    ax7.set_ylabel("INPUT CHANNELS: HEIGHT, AutoSWE, NETRAD, TMIN, TMAX, PRE")




    sampled_outputs = torch.cat((sampled_output_fake, sampled_output_real, torch.abs(sampled_output_fake - sampled_output_real)), 1)
    img = np.hstack([image for image in sampled_outputs.detach().numpy().reshape(num_samples, 64*3, 64)])



    # Modify the image to include the grid
    img[:,::dy] = grid_color
    img[::dx,:] = grid_color



    temp = ax8.imshow(img)
    plt.colorbar(temp, ax = ax8)
    ax8.set_ylabel("OUTPUTS: ERROR, REAL, GENERATED")


    swe_input = torch.from_numpy(arr_to_input(sampled_inp_old)[:,4,:,:]).unsqueeze(1)

    #img = np.hstack([image for image in swe_input.detach().numpy().reshape(num_samples, 64, 64)])
    #print(swe_input.shape, sampled_output_fake.shape)

    sampled_outputs = torch.cat((sampled_output_fake, swe_input, torch.abs(sampled_output_fake - swe_input)), 1)
    img = np.hstack([image for image in sampled_outputs.detach().numpy().reshape(num_samples, 64*3, 64)])



    # Modify the image to include the grid
    img[:,::dy] = grid_color
    img[::dx,:] = grid_color



    temp = ax9.imshow(img)
    plt.colorbar(temp, ax = ax9)
    ax9.set_ylabel("OUTPUTS: ERROR, Input SWE, Output SWE")
    sampled_output_fake_down = torch.from_numpy(np.array([downscale_local_mean(i[0,:,:], (2,2)) for i in sampled_output_fake.detach().numpy()]))
    sampled_output_real_down = torch.from_numpy(np.array([downscale_local_mean(i[0,:,:], (2,2)) for i in sampled_output_real.detach().numpy()]))

    diff = torch.abs(sampled_output_fake_down - sampled_output_real_down)

    sampled_outputs = torch.cat((sampled_output_fake_down, sampled_output_real_down), 1)
    img = np.hstack([image for image in sampled_outputs.detach().numpy().reshape(num_samples, 32*2, 32)])

    img[:,::32] = grid_color
    img[::32,:] = grid_color



    # temp = ax9.imshow(img)
    # plt.colorbar(temp, ax = ax9)
    # ax9.set_ylabel("AVGPOOL: REAL, GENERATED")

    img = np.hstack([image for image in diff.numpy().reshape(num_samples, 32, 32)])

    img[:,::32] = grid_color
    img[::32,:] = grid_color



    temp = ax10.matshow(img, cmap='gray')
    plt.colorbar(temp, ax = ax10)
    ax10.set_ylabel("AVGPOOL: ERRORS")

    plot_to_tensorboard(writer, name, f, step)


def pearson_coeff(start_date, inputA, inputB, model):
    fakeB = model.generate_fwd(torch.from_numpy(arr_to_input(take_two_weeks(inputA, start_date))[:,0:6,:,:]).cuda()).detach().cpu().numpy()
    realB = torch.from_numpy(arr_to_input(take_two_weeks(inputB, start_date))).numpy()

    xbar = np.mean(fakeB, axis=0)
    ybar = np.mean(realB, axis=0)
    xSTDev = np.std(fakeB, axis=0)
    ySTDev = np.std(realB, axis=0)
    r = np.mean(((fakeB - xbar)/(xSTDev)) * ((realB - ybar)/(ySTDev)), axis = 0)
    return r

def determination_coeff(start_date, inputA, inputB, model):
    fakeB = model.generate_fwd(torch.from_numpy(arr_to_input(take_two_weeks(inputA, start_date))[:,0:6,:,:]).cuda()).detach().cpu().numpy()
    realB = torch.from_numpy(arr_to_input(take_two_weeks(inputB, start_date))).numpy()

    xbar = np.mean(fakeB, axis=0)
    ybar = np.mean(realB, axis=0)
    xSTDev = np.std(fakeB, axis=0)
    ySTDev = np.std(realB, axis=0)
    r = np.mean(((fakeB - xbar)/(xSTDev)) * ((realB - ybar)/(ySTDev)), axis = 0)
    return r




def im_corr(im1, im2):
    #return circular correlation of images 1 and 2:
    if im1.shape != im2.shape:
        print("arrays need to have same shape")
        return None
    corrmat = np.zeros(im1.shape)
    top = np.concatenate((np.zeros(im2.shape), np.zeros(im2.shape)), axis=1)
    bottom = np.concatenate((im2, np.zeros(im2.shape)), axis = 1)

    shiftmat = np.concatenate((top, bottom))
    corrmat = convolve(im1, shiftmat, 'same')

    return (corrmat - np.min(corrmat))/(np.max(corrmat) - np.min(corrmat))

def im_circcorr(im1, im2):
    #return circular correlation of images 1 and 2:
    if im1.shape != im2.shape:
        print("arrays need to have same shape")
        return None
    corrmat = np.zeros(im1.shape)

    shiftmat = np.tile(im2, (2,2))
    corrmat = convolve(im1, shiftmat, 'same')

    return (corrmat - np.min(corrmat))/(np.max(corrmat) - np.min(corrmat))

def take_radial_slice_mean(image, radius, distances):
    #print(image[(distances >= radius-.5) & (distances < radius+.5)])
    #print(((distances >= radius-.5) & (distances < radius+.5)))
    return image[(distances >= radius-.5) & (distances  < radius+.5)].mean()

def spectral_density(im, power, circ):
    #print(im.shape)
    temp = (im - np.mean(im))/np.var(im)
    if power:
        if circ:
            psd = np.log10(abs(np.fft.fftshift(np.fft.fft2(im_circcorr(temp, temp)))))*20
        else:
            psd = np.log10(abs(np.fft.fftshift(np.fft.fft2(im_corr(temp, temp)))))*20
    else:
        psd = np.log10(abs(np.fft.fftshift(np.fft.fft2(temp))))*20

    x,y = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[1]))
    center = (im.shape[0]//2, im.shape[1]//2)
    x -= center[0]
    y -= center[1]
    R = np.sqrt(x**2 + y**2)
    r  = np.linspace(0,int(np.max(R)),num=int(np.max(R))*4)
    means = []
    # plot it
    for dist in r:
        #print(dist)
        means.append(take_radial_slice_mean(psd, dist, R))
    return means


def compute_spectras(im_list, power, circ):
    radial_means_list = []
    for index in range(0, im_list.shape[0]):
        # if index % 100 == 0:
        #     print(index)
        radial_means_list.append(spectral_density(im_list[index, 0, :, :].detach().numpy(), power, circ))

    return np.array(radial_means_list)

def give_data_random(inputA, inputB, num_days, model):
    image_indices = np.random.choice(804, num_days)
    fake_B = model.generate_fwd(torch.from_numpy(arr_to_input(inputA[image_indices, :,:, :])[:,:,:,:]).cuda()).cpu()
    real_B = torch.from_numpy(arr_to_input(inputB[image_indices, :,:, :]))
    return fake_B, real_B

def give_data_range(inputA, inputB, start_date, end_date, model):
    fake_B = model.generate_fwd(torch.from_numpy(arr_to_input(inputA[start_date:end_date, :,:, :])[:,:,:,:]).cuda()).cpu()
    real_B = torch.from_numpy(arr_to_input(inputB[start_date:end_date, :,:, :]))
    return fake_B, real_B

def plot_spectra_to_tensorboard(radial_means_real, radial_means_fake, log, writer, name, step):
    f = plt.figure(figsize=(30,20))
    f.patch.set_facecolor('white')


    psd_means_real = np.mean(radial_means_real, axis = 0)
    psd_stds_real = np.var(radial_means_real, axis = 0)**(0.5)

    psd_means_fake = np.mean(radial_means_fake, axis = 0)
    psd_stds_fake = np.var(radial_means_fake, axis = 0)**(0.5)


    x = np.linspace(0, 45.254833995939045, 180)
    y_real = psd_means_real
    error = 1.96 * psd_stds_real
    if log:
        plt.semilogx(x, y_real, 'k', color='#CC4F1B', label = 'Real')
    else:
        plt.plot(x, y_real, 'k', color='#CC4F1B', label = 'Real')
    plt.fill_between(x, y_real-error, y_real+error, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    y_fake = psd_means_fake
    error = 1.96 * psd_stds_fake
    if log:
        plt.semilogx(x, y_fake, 'k', color='#1B2ACC', label = 'Generated')
    else:
        plt.plot(x, y_fake, 'k', color='#1B2ACC', label = 'Generated')
    plt.fill_between(x, y_fake-error, y_fake+error, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='dashdot', antialiased=True)

    plt.legend()

    plt.title(name)
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Power Spectrum")

    plot_to_tensorboard(writer, name, f, step)

def plot_slices_histogram(radial_means_real, radial_means_fake, name, writer, step):
    
    f = plt.figure(figsize=(180,10))
    f.patch.set_facecolor('white')


    gs = gridspec.GridSpec(1, 18)

    for index in range(0, 180, 10):
        ax = plt.subplot(gs[0, index//10])
        range_min = min(np.min(radial_means_real[:,index]), np.min(radial_means_fake[:, index]))
        range_max = max(np.max(radial_means_real[:,index]), np.max(radial_means_fake[:, index]))
        hist_dat_real = np.histogram(radial_means_real[:,index], 30, (range_min, range_max))
        hist_dat_fake = np.histogram(radial_means_fake[:,index], 30, (range_min, range_max))

        log_metrics_np_hist(hist_dat_real, hist_dat_fake, name, writer, step)

        ax.plot(hist_dat_real[1][1:], hist_dat_real[0], linestyle = '--', c = '0.0', marker='s',  linewidth=3.0, label = 'Real Data')
        ax.plot(hist_dat_fake[1][1:], hist_dat_fake[0], marker = 'o', c = 'g', linewidth = 1.0, label = 'Generated Data')
        ax.legend(loc='upper left')
        freqs = np.linspace(0, 45.254833995939045, 180)
        title = "Histogram of pixel values " + str(freqs[index])
        ax.set_title(title)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Density Values at frequency ' + str(freqs[index]))

def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')

def log_metrics_np_hist(np_hist_real, np_hist_fake, name, writer, step):
    OPENCV_METHODS = (("Correlation", cv2.HISTCMP_CORREL), ("Chi-Squared", cv2.HISTCMP_CHISQR), ("Intersection", cv2.HISTCMP_INTERSECT), ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))
    cv_hist_real = np_hist_to_cv(np_hist_real)
    cv_hist_fake = np_hist_to_cv(np_hist_fake)
    for option in OPENCV_METHODS:
        temp_name = name + " " + option[0]
        writer.add_scalar(temp_name, cv2.compareHist(cv_hist_real, cv_hist_fake, option[1]), step)
