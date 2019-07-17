

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
from model import StochCycleGAN, AugmentedCycleGAN
import numpy as np
from evaluate import eval_mse_A, eval_ubo_B, one_to_three_channels
import shutil
import random
import glob
import json

from tensorboardX import SummaryWriter
from collections import OrderedDict
import torchvision


import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

def train_model():
    opt = TrainOptions().parse(sub_dirs=['vis_multi','vis_cycle','vis_latest','train_vis_cycle'])
    writer = SummaryWriter(opt.tbpath)
    out_f = open("%s/results.txt" % opt.expr_dir, 'w')
    copy_scripts_to_folder(opt.expr_dir)
    use_gpu = len(opt.gpu_ids) > 0

    if opt.seed is not None:
        print(("using random seed:", opt.seed))
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(opt.seed)

    if opt.numpy_data:
        trainA, trainB, devA, devB, testA, testB = load_numpy_data(opt.dataroot, grid_size=opt.grid_size)
        train_dataset = UnalignedIterator(trainA, trainB, batch_size=opt.batchSize)

        viz_dataset = UnalignedIterator(trainA, trainB, batch_size = 4)

        print_log(out_f, '#training images = %d' % len(train_dataset))
        vis_inf = False

        test_dataset = AlignedIterator(testA, testB, batch_size=100)
        print_log(out_f, '#test images = %d' % len(test_dataset))

        dev_dataset = AlignedIterator(devA, devB, batch_size=100)
        print_log(out_f, '#dev images = %d' % len(dev_dataset))

        dev_cycle = itertools.cycle(AlignedIterator(devA, devB, batch_size=25))

    viz_data = next(viz_dataset)
    viz_real_A, viz_real_B = Variable(viz_data['A']), Variable(viz_data['B'])
    viz_fake_A, viz_rec_A = viz_real_A, viz_real_B
    viz_fake_B, viz_rec_B = viz_real_B, viz_real_B
    if use_gpu:
        viz_real_A = viz_real_A.cuda()
        viz_real_B = viz_real_B.cuda()
        viz_fake_A = viz_fake_A.cuda()
        viz_fake_B = viz_fake_B.cuda()
        viz_rec_A = viz_rec_A.cuda()
        viz_rec_B = viz_rec_B.cuda()
    i = 0
    viz_real_A = viz_real_A[i]
    viz_real_B = viz_real_B[i]
    viz_fake_B = viz_fake_B[i]
    viz_rec_A = viz_rec_A[i]
    viz_fake_A = viz_fake_A[i]
    viz_rec_B = viz_rec_B[i]



    # viz_real_A = viz_real_A.view(-1, viz_real_A.shape[2], 3*viz_real_A.shape[2])
    # viz_fake_A = viz_fake_A.view(-1, viz_fake_A.shape[2], 3*viz_fake_A.shape[2])
    # viz_rec_A = viz_rec_A.view(-1, viz_rec_A.shape[2], 3*viz_rec_A.shape[2])

    print((viz_real_A.shape, viz_fake_A.shape, viz_rec_A.shape, viz_real_B.shape, viz_fake_B.shape, viz_rec_B.shape))




    img1 = np.random.rand(64*3, 64)
    img2 = np.random.randn(64,64)
    img3 = np.random.rand(64*3, 64)
    #print(img)
    G=nx.Graph()
    G.add_node('realA', image = img1)
    G.add_node('fakeB', image = img2)
    G.add_node('recA', image = img3)

    G.add_edge('realA','fakeB', r= 'G_AB')
    G.add_edge('fakeB', 'recA', r= 'G_BA')
    edge_labels = nx.get_edge_attributes(G,'r')
    pos={'realA': np.array([-1, 0]),
     'fakeB': np.array([0, 0]),
     'recA': np.array([ 1,0 ])}

    fig=plt.figure(figsize=(15,10))
    ax=plt.subplot(111, title="ABA Cycle")
    #ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)


    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform


    for n in G:

        piesize=(0.135/G.node[n]['image'].shape[1])*G.node[n]['image'].shape[0]
        p2=piesize/2.0
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize], xlabel = n)
        #a.title = "hi"
        #a.set_aspect('equal')
        a.imshow(G.node[n]['image'], cmap='gray')
        #print(G.node[n]['image'].shape)
    fig.savefig('temp1.png')
    plt.show()


        #plt.imshow(image)

if __name__ == "__main__":
    train_model()



# for i in range(0, 10):
#     img1 = np.random.rand(64*3, 64)
#     img2 = np.random.randn(64,64)
#     img3 = np.random.rand(64*3, 64)
#     #print(img)
#     G=nx.Graph()
#     G.add_node('realA', image = img1)
#     G.add_node('fakeB', image = img2)
#     G.add_node('recA', image = img3)
#
#     G.add_edge('realA','fakeB', r= 'G_AB')
#     G.add_edge('fakeB', 'recA', r= 'G_BA')
#     edge_labels = nx.get_edge_attributes(G,'r')
#     pos={'realA': np.array([-1, 0]),
#      'fakeB': np.array([0, 0]),
#      'recA': np.array([ 1,0 ])}
#
#     fig=plt.figure(figsize=(15,10))
#     ax=plt.subplot(111, title="ABA Cycle")
#     #ax.set_aspect('equal')
#     nx.draw_networkx_edges(G,pos,ax=ax)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
#
#
#     plt.xlim(-1.5,1.5)
#     plt.ylim(-1.5,1.5)
#
#     trans=ax.transData.transform
#     trans2=fig.transFigure.inverted().transform
#
#
#     for n in G:
#
#         piesize=(0.135/G.node[n]['image'].shape[1])*G.node[n]['image'].shape[0]
#         p2=piesize/2.0
#         xx,yy=trans(pos[n]) # figure coordinates
#         xa,ya=trans2((xx,yy)) # axes coordinates
#         a = plt.axes([xa-p2,ya-p2, piesize, piesize], xlabel = n)
#         #a.title = "hi"
#         #a.set_aspect('equal')
#         a.imshow(G.node[n]['image'], cmap='gray', aspect=1)
#         #print(G.node[n]['image'].shape)
#     plt.show()
