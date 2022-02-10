import os.path
import logging
import time
from collections import OrderedDict
from copy import deepcopy
from basicsr.utils import get_root_logger
import PIL.Image
import numpy
import torch


# --------------------------------
# basic settings
# --------------------------------
logger = get_root_logger()
model_names = ['msrresnet', 'san', 'han', 'pan', 'rcan', 'ignn', 'rnan', # 6
               'san_ablation_wNL', 'san_ablation_woNL', 'han_ablation_wLA', 'han_ablation_woLA', # 10
               'rnan_ablation_wNL', 'rnan_ablation_woNL', 'pan_ablation', 'rcan_ablation', # 14s
               'mdsr', 'swinir']
model_id = 16                # set the model name
sf = 4
model_name = model_names[model_id]
logger.info('{:>16s} : {:s}'.format('Model Name', model_name))
border = 0

testsets = 'testsets'         # set path of testsets
testset_L = 'DIV2K_valid_HR'  # set current testing dataset; 'DIV2K_test_LR'
testset_L = 'GTmod8'

save_results = True
print_modelsummary = True     # set False when calculating `Max Memery` and `Runtime`

torch.cuda.set_device(0)      # set GPU ID
logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# --------------------------------
# define network and load model
# --------------------------------
model_path = None
if model_name == 'pan':
    from basicsr.archs.pan_arch import PAN as net
    model = net(num_in_ch=3, num_out_ch=3, num_feat=40, unf=24, num_block=16, upscale=4)
    model_path = os.path.join('model_zoo', 'PANx4_DF2K.pth')
elif model_name == 'san':
    from basicsr.archs.san_arch import SAN as net
    model = net(factor=4, num_channels=3)
    model_path = os.path.join('model_zoo', 'SAN_BI4X.pt')
elif model_name == 'han':
    from basicsr.archs.han_arch import HAN as net
    model = net(factor=4, num_channels=3)
    model_path = os.path.join('model_zoo', 'HAN_BIX4.pt')
elif model_name == 'rcan':
    from basicsr.archs.rcan_arch import RCAN as net
    model = net( num_in_ch=3, num_out_ch=3, num_feat=64, num_group=10, num_block=10, squeeze_factor=16, upscale=4,
                 res_scale=1, img_range=255., rgb_mean=[0.4488, 0.4371, 0.4040])
    model_path = os.path.join('model_zoo', 'RCAN_basicSR.pth')
elif model_name == 'rnan':
    from basicsr.archs.rnan_arch import RNAN as net
    model = net(factor=4, num_channels=3)
    model_path = os.path.join('model_zoo', 'RNAN_SR_F64G10P48BIX4.pt')
elif model_name == 'ignn':
    from basicsr.archs.ignn_arch import IGNN as net
    model = net()
    model_path = os.path.join('model_zoo', 'IGNN_x4.pth')
elif model_name == 'han_ablation_wLA':
    from basicsr.archs.hanablation_arch import HANABLATION as net
    model = net(factor=4, num_channels=3, n_resgroups=10, n_resblocks=9, wLA=True)
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_HANx4_f64G3B10_wLA/models/'
                              , 'net_g_5000.pth')
elif model_name == 'han_ablation_woLA':
    from basicsr.archs.hanablation_arch import HANABLATION as net
    model = net(factor=4, num_channels=3, n_resgroups=10, n_resblocks=10, wLA=False)
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_HANx4_f64G3B10_woLA/models/'
                              , 'net_g_5000.pth')
elif model_name == 'san_ablation_wNL':
    from basicsr.archs.sanablation_arch import SANABLATION as net
    model = net(factor=4, num_channels=3, n_resgroups=10, n_resblocks=10, wNonLocal=True)
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_SAN_f64G3B10/models/'
                              , 'net_g_5000.pth')
elif model_name == 'san_ablation_woNL':
    from basicsr.archs.sanablation_arch import SANABLATION as net
    model = net(factor=4, num_channels=3, n_resgroups=10, n_resblocks=10, wNonLocal=False)
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_SAN_f64G3B10_woNL/models/'
                              , 'net_g_5000.pth')
elif model_name == 'rnan_ablation_wNL':
    from basicsr.archs.rnanablation_arch import RNANABLATION as net
    model = net(factor=4, num_channels=3, n_resgroup=3, n_resblock=2,  wNL=True)
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_RNANx4_f64G3B10_wNL/models/'
                              , 'net_g_5000.pth')
elif model_name == 'rnan_ablation_woNL':
    from basicsr.archs.rnanablation_arch import RNANABLATION as net
    model = net(factor=4, num_channels=3, n_resgroup=4, n_resblock=2, wNL=False)
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_RNANx4_f64G3B10_woNL/models/'
                              , 'net_g_5000.pth')
elif model_name == 'pan_ablation':
    from basicsr.archs.panablation_arch import PANABLATION as net
    model = net(num_in_ch=3, unf=24, num_groups=20, num_blocks=10)
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_PAN_x4_f64G3b10/models/'
                              , 'net_g_5000.pth')
elif model_name == 'rcan_ablation':
    from basicsr.archs.rcan_arch import RCAN as net
    model = net(num_in_ch=3, num_out_ch=3, num_feat=64, num_group=10, num_block=10, squeeze_factor=16, upscale=4,
                res_scale=1, img_range=255., rgb_mean=[0.4488, 0.4371, 0.4040])
    model_path = os.path.join('/data/zyli/projects/BasicSR/experiments/ablation_RCANx4_f64G3B10/models/'
                              , 'net_g_5000.pth')
elif model_name == 'mdsr':
    from basicsr.archs.mdsr_arch import MDSR as net
    model = net(factor=4, n_resblocks=80)
elif model_name == 'swinir':
    from basicsr.archs.xmswinir_arch import SwinIR as net
    model = net(upscale=4, in_chans=3, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

print(model_name)
# if model_name == 'srcnn' or model_name == 'fsrcnn':
#     state_dict = model.state_dict()
#     for n, p in torch.load(model_path, map_location=lambda storage, loc: storage).items():
#         if n in state_dict.keys():
#             state_dict[n].copy_(p)
#         else:
#             raise KeyError(n)
# elif model_name == 'msrresnet':
#     model.load_state_dict(torch.load(model_path), strict=True)
# elif model_name == 'ignn':
#     checkpoint = torch.load(model_path)
#     net_state_dict = OrderedDict()
#     for k, v in checkpoint['net_state_dict'].items():
#         name = k[7:]
#         net_state_dict[name] = v
#     model.load_state_dict(net_state_dict, False)
# elif model_name == 'san' or model_name == 'pan' or model_name == 'rnan' or model_name == 'hans':
#     load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
#     model.load_state_dict(load_net, strict=False)
# else:
#     load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
#     param_key = 'params' #RCAN
#     if param_key is not None:
#         if param_key not in load_net and 'params' in load_net:
#             param_key = 'params'
#             logger.info('Loading: params_ema does not exist, use params.')
#         load_net = load_net[param_key]
#     model.load_state_dict(load_net, strict=True)
#     # remove unnecessary 'module.'
#     # for k, v in deepcopy(load_net).items():
#     #     if k.startswith('module.'):
#     #         load_net[k[7:]] = v
#     #         load_net.pop(k)
#     # model.load_state_dict(torch.load(model_path), strict=True)


model.eval()
print(model.named_parameters())
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

# --------------------------------
# print model summary
# --------------------------------
if print_modelsummary:
    from utils_modelsummary import get_model_activation, get_model_flops, get_model_activation_ignn, get_model_flops_ignn
    activations = None
    num_conv2d = None
    input_dim = (3, 256, 256)  # set the input dimension
    input_dim_s = (3, 128, 128)
    if model_name == 'srcnn' or model_name == 'fsrcnn':
        input_dim = (1, 256, 256)
    if model_name == 'ignn':
        activations, num_conv2d = get_model_activation_ignn(model, input_dim_s, input_dim)
    else:
        activations, num_conv2d = get_model_activation(model, input_dim)
    print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
    print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

    if model_name == 'ignn':
        flops = get_model_flops_ignn(model, input_dim_s, input_dim, False)
    else:
        flops = get_model_flops(model, input_dim, False)
    print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

