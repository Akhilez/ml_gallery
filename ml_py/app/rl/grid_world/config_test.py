from omegaconf import OmegaConf

conf = OmegaConf.create({"foo": "bar", "foo2": "${foo}"})

print(conf.foo)
