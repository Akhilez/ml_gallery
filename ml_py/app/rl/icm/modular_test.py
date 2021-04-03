from typing import List, Union, Callable, Iterable

from omegaconf import DictConfig


class ProcessModule(DictConfig):
    required_keys = []

    def __init__(self, content=None, **kwargs):
        super().__init__(content or {}, **kwargs)

    def run(self):
        pass

    def validate_required_keys(self):
        for key in self.required_keys:
            if isinstance(key, dict):
                for dict_key in key:
                    assert dict_key in self
                    for item in key[dict_key]:
                        assert item in self[dict_key]
            else:
                assert key in self

    def __call__(self, additional_config=None):
        if additional_config:
            self.merge_config(additional_config)
        self.validate_required_keys()
        self.run()
        return self

    def merge_config(self, config):
        for key in config:
            if key[0] != "_":
                self[key] = config[key]

    def then(self, next_module):
        return next_module(self)


class Compose(ProcessModule):
    def __init__(self, *modules: ProcessModule):
        super().__init__()
        self._modules = modules

    def run(self):
        config = self
        for module in self._modules:
            config = module(config)
        self.merge_config(config)


class Loop(ProcessModule):
    def __init__(self, *modules: ProcessModule):
        super().__init__()
        self._modules = modules

    def run(self):
        # start infinite loop
        while True:
            # Go through all modules
            config = self
            for module in self._modules:
                config = module(config)
            self.merge_config(config)

            # check if terminate
            if self.terminate():
                break

    def terminate(self) -> bool:
        return True


class DataInit(ProcessModule):
    def run(self):
        pass


class PrintModule(ProcessModule):
    required_keys = ProcessModule.required_keys + ["boo"]

    def run(self):
        print(self.name)
        print(self.boo)


class WAndBModule(ProcessModule):
    def __init__(self, content, config_key: str = None, **kwargs):
        super().__init__(content)
        import wandb

        self.wb = wandb.init(**kwargs)
        self._add_wb_config(config_key)

    def _add_wb_config(self, config_key):
        config = self[config_key] if config_key else self
        self.wb.config = config


if __name__ == "__main__":

    ProcessModule({"name": "yo"})().then(PrintModule({"boo": 9}))

    initial_data = {"name": "akhil", "boo": 9}

    Compose(
        PrintModule(initial_data),
        PrintModule(),
        PrintModule(),
    )()
