from omegaconf import DictConfig


class ProcessModule(DictConfig):
    required_keys = ["name"]

    def __init__(self, content=None, **kwargs):
        super().__init__(content or {}, **kwargs)

    def run(self):
        pass

    def validate_required_keys(self):
        for key in self.required_keys:
            assert key in self

    def __call__(self, additional_config=None):
        if additional_config:
            self.merge_config(additional_config)
        self.validate_required_keys()
        self.run()
        return self

    def merge_config(self, config):
        for key in config:
            self[key] = config[key]

    def then(self, next_module):
        return next_module(self)


class PrintModule(ProcessModule):
    required_keys = ProcessModule.required_keys + ["boo"]

    def run(self):
        print(self.name)
        print(self.boo)


class Sequential:
    def __init__(self, *args):
        self.processes = args

    def run(self):
        context = None
        for process in self.processes:
            context = process(context)


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

    Sequential(
        PrintModule(initial_data),
        PrintModule(),
        PrintModule(),
    ).run()
