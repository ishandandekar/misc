from pathlib import Path
from loader import DataLoaderStrategyFactory

if __name__ == "__main__":
    import yaml
    from box import Box

    params_filepath = Path("loader_params.yaml")
    with open(params_filepath, "r") as f_in:
        load_params = Box(yaml.safe_load(f_in)).data.load

    data = DataLoaderStrategyFactory.get(load_params.strategy)(**load_params.args)()
    print(data.head(4))
