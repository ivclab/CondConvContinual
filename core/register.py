
registered_optimizers = {}
registered_schedulers = {}
registered_backbones = {}


def register_module(*args, **kwargs):

    def register(cls):
        if args[0] == 'optimizers':
            registered_optimizers[cls.__name__] = cls
        elif args[0] == 'schedulers':
            registered_schedulers[cls.__name__] = cls
        elif args[0] == 'backbones':
            registered_backbones[cls.__name__] = cls
        else:
            raise NotImplementedError(
                'Unrecognized module_type: {}'.format(module_type))
        return cls

    return register
