
import os
import sys

import numpy as np
import torch

from PIL import Image
from matplotlib import pyplot as pl
from scipy.ndimage import uniform_filter
smooth = lambda arr: uniform_filter(arr, 3)

def transparent(img, alpha, cmap, **kw):
    from matplotlib.colors import Normalize
    colored_img = cmap(Normalize(clip=True,**kw)(img))
    colored_img[:,:,-1] = alpha
    return colored_img

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
from extract import NonMaxSuppression


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Visualize the patch detector and descriptor")
    
    parser.add_argument("--img", type=str, default="imgs/brooklyn.png")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--out", type=str, default="viz.png")

    parser.add_argument("--checkpoint", type=str, required=True, help='network path')
    parser.add_argument("--net", type=str, default="", help='network command')

    parser.add_argument("--max-kpts", type=int, default=200)
    parser.add_argument("--reliability-thr", type=float, default=0.8)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)
    parser.add_argument("--border", type=int, default=20,help='rm keypoints close to border')

    parser.add_argument("--gpu", type=int, nargs='+', required=True, help='-1 for CPU')
    parser.add_argument("--dbg", type=str, nargs='+', default=(), help='debug options')
    
    args = parser.parse_args()
    args.dbg = set(args.dbg)
    
    iscuda = common.torch_set_gpu(args.gpu)
    device = torch.device('cuda' if iscuda else 'cpu')

    # create network
    checkpoint = torch.load(args.checkpoint, lambda a,b:a)
    args.net = args.net or checkpoint['net']
    print("\n>> Creating net = " + args.net) 
    net = eval(args.net)
    net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
    if iscuda: net = net.cuda()
    print(f" ( Model size: {common.model_size(net)/1000:.0f}K parameters )")

    img = Image.open(args.img).convert('RGB')
    if args.resize: img.thumbnail((args.resize,args.resize))
    img = np.asarray(img)
        
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)

    with torch.no_grad():
        print(">> computing features...")
        res = net(imgs=[norm_RGB(img).unsqueeze(0).to(device)])
        rela = res.get('reliability')
        repe = res.get('repeatability')
        kpts = torch.transpose(detector(**res), 0, 1)[:,[1,0]]
        # For newer pytorch
        # kpts = kpts[torch.argsort(repe[0][0,0][kpts[:,1],kpts[:,0]])[-args.max_kpts:]]

        # For older pytorch
        max_index = torch.sort(-repe[0][0,0][kpts[:,1],kpts[:,0]])[1][:args.max_kpts]
        kpts  = kpts[max_index]

    print("No of points = {}".format(kpts.shape[0]))
    fig = pl.figure("viz",figsize= (10,6))
    kw = dict(cmap=pl.cm.RdYlGn, vmax=1)
    crop = (slice(args.border,-args.border or 1),)*2
    
    if 'reliability' in args.dbg:
    
        ax1 = pl.subplot(131)
        pl.imshow(img[crop], cmap=pl.cm.gray)
        pl.xticks(()); pl.yticks(())

        pl.subplot(132)
        pl.imshow(img[crop], cmap=pl.cm.gray, alpha=0)
        pl.xticks(()); pl.yticks(())

        x,y = kpts[:,0:2].cpu().numpy().T - args.border
        pl.plot(x,y,'+',c=(0,1,0),ms=10, scalex=0, scaley=0)

        ax1 = pl.subplot(133)
        rela = rela[0][0,0].cpu().numpy()
        pl.imshow(rela[crop], cmap=pl.cm.RdYlGn, vmax=1, vmin=0.9)
        pl.xticks(()); pl.yticks(())

    else:
        ax1 = pl.subplot(211)
        pl.imshow(img[crop], cmap=pl.cm.gray)
        pl.xticks(()); pl.yticks(())

        x,y = kpts[:,0:2].cpu().numpy().T - args.border
        pl.plot(x,y,'+',c=(0,1,0),ms=10, scalex=0, scaley=0)
        pl.title('Keypoints location')

        pl.subplot(212)
        pl.imshow(img[crop], cmap=pl.cm.gray)
        pl.xticks(()); pl.yticks(())
        c = repe[0][0,0].cpu().numpy()
        pl.imshow(transparent(smooth(c)[crop], 0.5, vmin=0, **kw))
        pl.title('Repeatability')

        # ax1 = pl.subplot(133)
        # pl.imshow(img[crop], cmap=pl.cm.gray)
        # pl.xticks(()); pl.yticks(())
        # rela = rela[0][0,0].cpu().numpy()
        # pl.imshow(transparent(rela[crop], 0.5, vmin=0.9, **kw))
        # pl.title('Reliability')

    # pl.gcf().set_size_inches(9, 2.73)
    pl.subplots_adjust(0.01,0.01,0.99,0.99,hspace=0.1)
    save_path = os.path.join("images_output", os.path.splitext(os.path.basename(args.img))[0] + ".png")
    print("Saving to {}".format(save_path))
    pl.savefig(save_path)
    # pdb.set_trace()
    # pl.show()

