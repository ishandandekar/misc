from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import typing as t
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import requests
import boto3


class DataLoaderStrategy(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load data
        """


@dataclass
class DirDataLoaderStrategy(DataLoaderStrategy):
    dir: t.Union[Path, str]
    columns: list[str]

    def load_data(self) -> pd.DataFrame:
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        if self.dir.is_dir():
            data = pd.DataFrame(columns=self.columns)
            paths = list(self.dir.glob("*.csv"))
            if len(paths) == 0:
                raise Exception(f"No `.csv` present in the directory {dir}")
            for path in paths:
                df = pd.read_csv(path)
                data = pd.concat([data, df])
            return data
        else:
            raise Exception("Path provided is not a directory")


@dataclass
class UrlDataLoaderStrategy(DataLoaderStrategy):
    url: str
    columns: list[str]

    def load_data(self) -> DataFrame:
        if not self.url.endswith(".csv"):
            raise Exception(f"The url is not of a `.csv`: {self.url}")
        response = requests.get(self.url)
        response.raise_for_status()
        csv_file = StringIO(response.text)
        df = pd.read_csv(csv_file)
        if df.columns.to_list() != self.columns:
            raise Exception(f"Column mismatch error for `.csv`: {self.url}")
        return df


@dataclass
class AwsS3DataLoaderStrategy(DataLoaderStrategy):
    bucket_name: str
    folder_path: str
    session: boto3.Session

    def load_data(self) -> DataFrame:
        s3_client = self.session.client("s3")
        if self.folder_path == "":
            s3_url = f"s3://{self.bucket_name}"
            objects = s3_client.list_objects_v2(Bucket=self.bucket_name)
        else:
            s3_url = f"s3://{self.bucket_name}/{self.folder_path}"
            objects = s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.folder_path
            )

        dataframes: list = list()
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv"):
                file_url = f"{s3_url}/{key}"
                print(file_url)
                try:
                    dataframe = pd.read_csv(file_url)
                    dataframes.append(dataframe)
                    print(f"Loaded: {key}")
                except Exception as e:
                    print(f"Error loading {key}: {e}")

        if len(dataframes) == 0:
            raise Exception(
                "No data found. Check the contents in the bucket and folder"
            )
        return pd.concat(dataframes, ignore_index=True)


StrategyMap: t.Dict[str, t.Type[DataLoaderStrategy]] = {
    "dir": DirDataLoaderStrategy,
    "url": UrlDataLoaderStrategy,
    "aws_s3": AwsS3DataLoaderStrategy,
}


if __name__ == "__main__":
    import yaml
    from box import Box

    params_filepath = Path("loader_params.yaml")
    with open(params_filepath, "r") as f_in:
        load_params = Box(yaml.safe_load(f_in)).data.load

    dataloader = StrategyMap.get(load_params.strategy)(**load_params.args)
    data = dataloader.load_data()
    print(data.head(4))
