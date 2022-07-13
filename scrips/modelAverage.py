import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
from basicsr.archs.rfdnfinalB5_arch import RFDNFINALB5 as net
from copy import deepcopy


def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net


def save_network(net, path, param_key='params'):
    """Save networks.

    Args:
        net (nn.Module | list[nn.Module]): Network(s) to be saved.
        net_label (str): Network label.
        current_iter (int): Current iter number.
        param_key (str | list[str]): The parameter key(s) to save network.
            Default: 'params'.
    """
    save_path = path

    net = net if isinstance(net, list) else [net]
    param_key = param_key if isinstance(param_key, list) else [param_key]
    assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

    save_dict = {}
    for net_, param_key_ in zip(net, param_key):
        net_ = get_bare_model(net_)
        state_dict = net_.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            state_dict[key] = param.cpu()
        save_dict[param_key_] = state_dict

    # avoid occasional writing errors
    retry = 3
    while retry > 0:
        try:
            torch.save(save_dict, save_path)
        except Exception as e:
            pass
        else:
            break
        finally:
            retry -= 1



model = net(num_block=5, conv='BSConvU', num_feat=48)
save_dir = '/home/yqliu/projects/BasicSR/model_average/output_models/'
save_name = 'test.pth'
load_dir = '/home/yqliu/projects/BasicSR/model_average/load_dir/'
model_paths = [os.path.join(load_dir, name) for name in os.listdir(load_dir) if name.endswith('.pth')] #这里的'.tif'可以换成任意的文件后缀
num_models = len(model_paths)
beta = 1. / num_models
param_key= 'params'
target_state_dict = model.state_dict()

for key in target_state_dict:
    if target_state_dict[key].data.dtype == torch.float32:
        target_state_dict[key].data.fill_(0.)
        for model_path in model_paths:
            load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
            if param_key is not None:
                if param_key not in load_net and 'params' in load_net:
                    param_key = 'params'
                load_net = load_net[param_key]
            for k, v in deepcopy(load_net).items():
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)
            target_state_dict[key].data += load_net[key].data.clone() * beta

model.load_state_dict(target_state_dict, strict=True)
save_network(model, os.path.join(save_dir, save_name))




# load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
# if param_key is not None:
#     if param_key not in load_net and 'params' in load_net:
#         param_key = 'params'
#     load_net = load_net[param_key]
# logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
# # remove unnecessary 'module.'
# for k, v in deepcopy(load_net).items():
#     if k.startswith('module.'):
#         load_net[k[7:]] = v
#         load_net.pop(k)
# self._print_different_keys_loading(net, load_net, True)