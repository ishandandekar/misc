import prefect
from prefect import task, flow


@task(name="Task1")
def hello():
    return {"str": "Hello", "int": 69}


@flow(name="GH_Action_flow_test")
def flower():
    boom = hello()
    return None


if __name__ == "__main__":
    flower()
