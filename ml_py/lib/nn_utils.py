import torch


def get_scaled_random_weights(shape, min_=-0.5, max_=0.5):
    return torch.FloatTensor(*shape).uniform_(min_, max_).requires_grad_()


def fire_and_forget(f):
    def wrapped(*args, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)

    return wrapped
