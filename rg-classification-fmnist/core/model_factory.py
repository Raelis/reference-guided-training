# Model construction.
from copy import deepcopy

def model_class(name):
    if name == 'CNN':
        from models.cnn2d import MyModel
    elif name == 'MLP':
        from models.mlp_model import MyModel
    elif name == 'Transformer':
        from models.transformer import MyModel
    elif name == 'ResNet20':
        from models.resnet20 import MyModel
    elif name == 'MLP-guide':
        from models.mlp_model_guide import MyModel
    else:
        raise ValueError(f'Unsupported model: {name}')
    return MyModel


def architecture_from_study(model, selected):
    # Map selected Optuna architecture parameters to model constructor keys.
    params = dict(selected)
    if model == 'CNN' and 'cnn_channels' not in params:
        names = [f'cnn_channel_l{i}' for i in range(3)]
        if all(name in params for name in names):
            params['cnn_channels'] = [params.pop(name) for name in names]
    if model == 'MLP' and 'hidden_dims' not in params:
        if 'mlp_layers' in params:
            n = int(params.pop('mlp_layers'))
            names = [f'mlp_hidden_{i}' for i in range(n)]
            if not all(name in params for name in names):
                raise ValueError(f'Missing selected Stage-1 MLP dimensions: {names}')
            params['hidden_dims'] = tuple(params.pop(name) for name in names)
        if 'mlp_dropout' in params:
            params['dropout'] = params.pop('mlp_dropout')
    return params


def construct(model, arch, device):
    cls = model_class(model)
    kwargs, _ = cls.prepare_model_params(deepcopy(arch))
    kwargs['num_features'] = 784
    kwargs['output_dim'] = 10
    instance = cls(**kwargs).to(device)
    cls.model_postprocess(instance)
    return instance


