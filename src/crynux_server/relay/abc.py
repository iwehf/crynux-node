from abc import ABC, abstractmethod
from typing import List, BinaryIO

from crynux_server.models import RelayTask, GPTTaskResponse


class Relay(ABC):
    @abstractmethod
    async def create_task(self, task_id: int, task_args: str) -> RelayTask:
        ...

    @abstractmethod
    async def get_task(self, task_id: int) -> RelayTask:
        ...

    @abstractmethod
    async def upload_sd_task_result(self, task_id: int, file_paths: List[str]):
        ...

    @abstractmethod
    async def upload_gpt_task_result(self, task_id: int, response: GPTTaskResponse):
        ...

    @abstractmethod
    async def get_sd_result(self, task_id: int, index: int, dst: BinaryIO):
        ...

    @abstractmethod
    async def get_gpt_result(self, task_id: int) -> GPTTaskResponse:
        ...

    @abstractmethod
    async def close(self):
        ...
