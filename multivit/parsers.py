

def parse_adaptor_arguments(
        adaptor,
):
    """
    Adaptor strings are of the form:
     - 'linear', 'TR' | 'TCn', Flabel_name
     - 'mlp', S128, L2, D10, 'TR' | 'TCn', Flabel_name

    'mlp' takes a hidden size (S), number of hidden layers (L) and dropout (D)

    It outputs a dictionary as follows:
    Given ('mlp', 'S128', 'L2', 'D10', 'TR', 'Fthing'), it outputs:
    {
        'head': {
            head: 'mlp',
            kwargs: {'hidden_size': 128, 'hidden_layers': 2, 'dropout': 0.1}
        },
        'loss': {
            loss: 'XEnt',
            label: 'thing',
            kwargs: {'nclasses': 5}
        }
    }

    Given ('adaptor', 'mlp',):
    {
        'adaptor': 'mlp',
        'input_size': <input_size>,
        'hidden_size: <input_size> // 2,
        'output_size: <output_size>,
        'dropout': 0,
    }
    """
    def fetch_field(fields, field_start):
        field = [f for f in fields[1:] if f[0] == field_start]
        if len(field) == 0:
            raise ValueError(f"'adaptor' ({fields}) must contain a '{field_start}' field")
        if len(field) > 1:
            raise ValueError(f"'adaptor' ({fields}) must contain only a single '{field_start}' field")
        return field[0][1:]


    if not isinstance(adaptor, (tuple, list)):
        raise TypeError(f"'adaptor' must be a tuple or list but is of type '{type(adaptor)}'")
    if len(adaptor) == 0:
        raise ValueError("'adaptor' must contain at least one element")

    task = fetch_field(adaptor, 'T')
    label = fetch_field(adaptor, 'F')

    if adaptor[0] == 'linear':
        if len(adaptor) > 3:
            raise ValueError(f"'adaptor' value of 'linear'({adaptor}) should be three elements e.g. ['linear', 'Fpasi', 'R'])")

        head_config = {
            'head': adaptor[0],
        }
    elif adaptor[0] == 'mlp':
        if len(adaptor) > 6:
            raise ValueError(f"'adaptor' value of 'mlp' takes up to five additional arguments: received {adaptor}")
        if any(a[0] not in ('S', 'L', 'D', 'F', 'T') for a in adaptor[1:]):
            raise ValueError(f"'adaptor' arguments must start with ('S', 'L', 'D', 'F', or 'T'): received {adaptor}")
        if any(adaptor[1:].count(s[0]) > 1 for s in adaptor[1:]):
            raise ValueError(f"'adaptor' arguments must only appear once: received {adaptor}")
        head_config = {
            'head': adaptor[0],
            'kwargs': {
                'hidden_layers': 1,
                'dropout': 0,
            }
        }
        # override the default config with any user-specified values
        for arg in adaptor[1:]:
            if arg[0] == 'S':
               head_config['kwargs']['hidden_channels'] = int(arg[1:])
            elif arg[0] == 'L':
               head_config['kwargs']['hidden_layers'] = int(arg[1:])
            elif arg[0] == 'D':
               head_config['kwargs']['dropout'] = float(arg[1:])
    else:
        raise ValueError(f"'adaptor' ({adaptor[0]}) must be one of ('linear', 'mlp')")

    loss_map = {
        'R': 'mse',
        'C': 'xent',
    }

    loss_config = {
        'loss': loss_map[task[0]],
        'label': label,
    }

    if task[0] == 'C':
        classes = int(task[1:])
        head_config['classes'] = classes

    return {
        'name': ' '.join(adaptor),
        'head': head_config,
        'loss': loss_config,
    }


def set_adaptor_sizes(
        input_size,
        output_size,
        adaptor,
):
    out_adaptor = dict(adaptor)
    out_adaptor_head = out_adaptor['head']
    if out_adaptor_head['head'] == 'linear':
        if 'kwargs' not in out_adaptor_head:
            out_adaptor_head['kwargs'] = {}
        if 'in_features' not in out_adaptor_head['kwargs']:
            out_adaptor_head['kwargs']['in_features'] = input_size
        if 'out_features' not in out_adaptor_head['kwargs']:
            out_adaptor_head['kwargs']['out_features'] = output_size

    elif out_adaptor_head['head'] == 'mlp':
        if 'kwargs' not in out_adaptor_head:
            out_adaptor_head['kwargs'] = {}
        if 'input_channels' not in out_adaptor_head:
            out_adaptor_head['kwargs']['input_channels'] = input_size
        if 'hidden_channels' not in out_adaptor_head:
            out_adaptor_head['kwargs']['hidden_channels'] = input_size // 2
        if 'output_channels' not in out_adaptor_head:
            out_adaptor_head['kwargs']['output_channels'] = output_size

    else:
        raise ValueError(f"adaptor['head']['head'] '({out_adaptor['head']['head']})' must be one of 'linear' or 'mlp')")

    return out_adaptor
