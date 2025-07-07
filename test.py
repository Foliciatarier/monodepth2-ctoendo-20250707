from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import os
from progressbar import progressbar
import torch
from torchvision import transforms

from layers import disp_to_depth
from options import MonodepthOptions
import datasets
import networks


def visualize_depth(depth, sav=None):    
    fig = plt.figure()
    plt.imshow(depth, cmap = 'viridis', interpolation = 'nearest', aspect = 1.)
    plt.colorbar()
    plt.gca().invert_yaxis()
    if sav is not None:
        fig.savefig(sav)
        plt.close()
    return fig


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Compute errors on a model of each epoch using a specified test set
    """
    device = torch.device(opt.device)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in depth_decoder.parameters())
    print('Params:', total_params)

    dataset = datasets.CToEndoFlatten(opt.data_path)
    transform = transforms.Compose([
        transforms.CenterCrop((opt.height, opt.width)),
        transforms.ToTensor()])
    print("There are {:d} test samples\n".format(len(dataset)))

    output_dir = 'output/errors_%s.txt' % (opt.model_name)
    print(' -> Save to %s' % (output_dir))
    filerr = open(output_dir, 'w')
    print("  " + ("{:>8} | " * 8).format('epoch', "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"), file=filerr)

    for epoch in range(opt.num_epochs):

        opt.load_weights_folder = os.path.expanduser(os.path.join(opt.log_dir, opt.model_name, 'models', 'weights_%d' % epoch))
        try:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_dict = torch.load(encoder_path, map_location=device)
            decoder_dict = torch.load(decoder_path, map_location=device)
            print(" -> Loading weights from %s" % (opt.load_weights_folder))
        except Exception as e:
            print(e)
            continue
        
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(decoder_dict)
        encoder.to(device)
        encoder.eval()
        depth_decoder.to(device)
        depth_decoder.eval()

        errors, ratios = [], []
        print(" -> Computing predictions of epoch %d" % (epoch + 1))

        with torch.no_grad():
            for idx in progressbar(range(len(dataset))):
                img, dep = dataset[idx]
                input_color: torch.Tensor = transform(datasets.loadImg(img)).to(device)
                input_color = input_color.unsqueeze(0)

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_depth = 1 / pred_disp
                pred_depth = pred_depth.squeeze()

                gt_depth = dep
                gt_depth = gt_depth[8:-8, 8:-8]

                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
                pred_depth[pred_depth < opt.min_depth] = opt.min_depth
                pred_depth[pred_depth > opt.max_depth] = opt.max_depth
                errors.append(compute_errors(gt_depth, pred_depth))

        ratios = np.array(ratios)
        mean_errors = np.array(errors).mean(0)

        print(("&{: 8d}  ").format(epoch + 1) + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\", file=filerr)
    filerr.close()


def evaluate_depth(opt):
    """Evaluates a pretrained model using a specified test set
    """
    device = torch.device(opt.device)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in depth_decoder.parameters())
    print('Params:', total_params)

    dataset = datasets.CToEndoFlatten(opt.data_path)
    transform = transforms.Compose([
        transforms.CenterCrop((opt.height, opt.width)),
        transforms.ToTensor()])
    print("There are {:d} test samples\n".format(len(dataset)))

    output_dir = 'output/CTo%s_%s_%d' % (opt.data_path[-9:-4], opt.model_name, opt.num_epochs+1)
    output_gts = 'output/CTo%s_GT' % (opt.data_path[-9:-4])
    print(' -> Save to %s and %s' % (output_dir, output_gts))
    os.makedirs(output_gts, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    epoch = opt.num_epochs

    opt.load_weights_folder = os.path.expanduser(os.path.join(opt.log_dir, opt.model_name, 'models', 'weights_%d' % epoch))
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)
    print(" -> Loading weights from %s" % (opt.load_weights_folder))

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(decoder_dict)
    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    with torch.no_grad():
        for idx in progressbar(range(len(dataset))):
            img, dep = dataset[idx]
            input_color: torch.Tensor = transform(datasets.loadImg(img)).to(device)
            input_color = input_color.unsqueeze(0)

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_depth = 1 / pred_disp
            pred_depth = pred_depth.squeeze()
            visualize_depth(pred_depth, sav=output_dir+'/D%03d.png' % (idx))

            gt_depth = dep
            gt_depth = gt_depth[8:-8, 8:-8]
            if not os.path.exists(output_gts+'/G%03d.png' % (idx)):
                visualize_depth(gt_depth, sav=output_gts+'/G%03d.png' % (idx))


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
    evaluate_depth(options.parse())
