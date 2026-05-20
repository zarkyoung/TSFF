import yaml
from yaml import SafeLoader

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.loader = SafeLoader
            self.loader.add_constructor('!include', self._include)
            config_dict = yaml.safe_load(f)
        
        # 处理包含的配置
        if 'base' in config_dict and isinstance(config_dict['base'], dict):
            base_config = config_dict.pop('base')
            base_config.update(config_dict)
            config_dict = base_config

        for key, value in config_dict.items():
            setattr(self, key, value)

    def _include(self, loader, node):
        filename = loader.construct_scalar(node)
        with open(f'configs/{filename}', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)