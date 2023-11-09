import prefect
from prefect import task, flow


@task(name="Task1")
def hello():
    return {"str": "Hello", "integer": 69}


@task(name="intermediate")
def intermediate():
    _ = hello()
    return _.get("integer")


@flow(name="GH_Action_flow_test")
def flower():
    _ = intermediate()
    return None


if __name__ == "__main__":
    flower()
