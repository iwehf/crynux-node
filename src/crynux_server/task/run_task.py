from typing import Protocol, Optional

from celery.result import AsyncResult
from crynux_server.models import TaskType
from crynux_server.config import TaskConfig
from crynux_worker.task import inference
from crynux_server.celery_app import get_celery


class RunTaskProto(Protocol):
    def __call__(
        self,
        distributed: bool,
        task_id: int,
        task_type: TaskType,
        task_args: str,
        task_config: Optional[TaskConfig] = None,
        stream: bool = False,
    ):
        ...


def _run_distributed_task(
    task_id: int,
    task_type: TaskType,
    task_args: str,
    stream: bool,
):
    celery = get_celery()
    kwargs = {
        "task_id": task_id,
        "task_type": int(task_type),
        "task_args": task_args,
        "distributed": True,
    }
    res: AsyncResult = celery.send_task(
        "inference",
        kwargs=kwargs,
    )
    res.get()


def _run_local_task(
    task_id: int,
    task_type: TaskType,
    task_args: str,
    task_config: TaskConfig,
    stream: bool,
):
    proxy = None
    if task_config.proxy is not None:
        proxy = task_config.proxy.model_dump()

    inference(
        task_id=task_id,
        task_type=int(task_type),
        task_args=task_args,
        distributed=False,
        result_url=task_config.result_url,
        output_dir=task_config.output_dir,
        hf_cache_dir=task_config.hf_cache_dir,
        external_cache_dir=task_config.external_cache_dir,
        script_dir=task_config.script_dir,
        inference_logs_dir=task_config.inference_logs_dir,
        proxy=proxy,
    )


def run_task(
    distributed: bool,
    task_id: int,
    task_type: TaskType,
    task_args: str,
    task_config: Optional[TaskConfig] = None,
    stream: bool = False,
):
    if distributed:
        _run_distributed_task(
            task_id=task_id, task_type=task_type, task_args=task_args, stream=stream
        )
    else:
        assert (
            task_config is not None
        ), "Task config is None when run task in non distributed mode"

        _run_local_task(
            task_id=task_id,
            task_type=task_type,
            task_args=task_args,
            task_config=task_config,
            stream=stream,
        )
