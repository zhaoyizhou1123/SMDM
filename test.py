from dataclasses import dataclass

@dataclass
class Config:
    a: int

config = Config(a=10)
config.b = 5
print(config)