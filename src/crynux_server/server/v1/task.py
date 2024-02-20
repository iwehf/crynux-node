import logging
import os
import shutil
import time
from datetime import datetime
from string import hexdigits
from typing import List, Literal

from anyio import create_memory_object_stream, create_task_group
from fastapi import (APIRouter, File, Form, HTTPException, Path, UploadFile,
                     WebSocket, WebSocketDisconnect, WebSocketException,
                     status)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing_extensions import Annotated

from crynux_server.models import NodeStatus, TaskResultReady, TaskStatus, TaskType

from ..depends import (ConfigDep, EventQueueDep, ManagerStateCacheDep,
                       TaskStateCacheDep, RelayDep)
from .utils import CommonResponse

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks")


def store_result_files(task_dir: str, task_id: str, files: List[UploadFile]):
    result_dir = os.path.join(task_dir, task_id)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    dsts: List[str] = []
    for file in files:
        assert file.filename is not None
        dst_filename = os.path.join(result_dir, file.filename)
        with open(dst_filename, mode="wb") as dst:
            shutil.copyfileobj(file.file, dst)
        dsts.append(dst_filename)
    return dsts


def is_valid_hexstr(h: str) -> bool:
    if not h.startswith("0x"):
        return False
    return all(c in hexdigits for c in h[2:])


@router.post("/{task_id}/result", response_model=CommonResponse)
async def upload_result(
    task_id: Annotated[int, Path(description="The task id")],
    hashes: Annotated[List[str], Form(description="The task result file hashes")],
    files: Annotated[List[UploadFile], File(description="The task result files")],
    *,
    config: ConfigDep,
    task_state_cache: TaskStateCacheDep,
    event_queue: EventQueueDep,
) -> CommonResponse:
    if event_queue is None or task_state_cache is None:
        raise HTTPException(status_code=400, detail="Node is not running.")
    if not (await task_state_cache.has(task_id=task_id)):
        raise HTTPException(status_code=400, detail=f"Task {task_id} does not exist.")

    for h in hashes:
        if not is_valid_hexstr(h):
            raise HTTPException(
                status_code=400, detail="Hash is not a valid hex string."
            )

    for file in files:
        if file.filename is None:
            raise HTTPException(status_code=400, detail="Filename is missing.")

    dsts = await run_in_threadpool(
        store_result_files, config.task_dir, str(task_id), files
    )

    event = TaskResultReady(task_id=task_id, hashes=hashes, files=dsts)
    await event_queue.put(event)

    return CommonResponse()


@router.websocket("/{task_id}/result/stream")
async def upload_gpt_result_stream(
    websocket: WebSocket,
    task_id: Annotated[int, Path(description="The task id")],
    *,
    task_state_cache: TaskStateCacheDep,
    relay: RelayDep
):
    if task_state_cache is None or relay is None:
        raise WebSocketException(
            status.WS_1008_POLICY_VIOLATION, "Node is not running."
        )
    if not (await task_state_cache.has(task_id=task_id)):
        raise WebSocketException(
            status.WS_1008_POLICY_VIOLATION, f"Task {task_id} does not exist."
        )

    state = await task_state_cache.load(task_id=task_id)
    if state.task_type != TaskType.LLM:
        raise WebSocketException(
            status.WS_1008_POLICY_VIOLATION, f"Task {task_id} is not a gpt task."
        )

    await websocket.accept()

    tx, rx = create_memory_object_stream(1, item_type=str)

    async def _send_token():
        async with tx:
            try:
                while True:
                    token = await websocket.receive_text()
                    _logger.debug(f"Task {task_id} gpt result stream token: {token}")
                    await tx.send(token)
            except WebSocketDisconnect:
                _logger.debug(f"Task {task_id} upload gpt result stream complete")

    async with create_task_group() as tg:
        tg.start_soon(_send_token)
        async with rx:
            await relay.upload_gpt_task_result_stream(task_id, rx)


class TaskStats(BaseModel):
    status: Literal["running", "idle", "stopped"]
    num_today: int
    num_total: int


@router.get("", response_model=TaskStats)
async def get_task_stats(
    *, task_state_cache: TaskStateCacheDep, state_cache: ManagerStateCacheDep
):
    node_status = (await state_cache.get_node_state()).status
    if task_state_cache is None or node_status not in [
        NodeStatus.Running,
        NodeStatus.PendingPause,
        NodeStatus.PendingStop,
    ]:
        return TaskStats(status="stopped", num_today=0, num_total=0)
    running_status = [
        TaskStatus.Pending,
        TaskStatus.Executing,
        TaskStatus.ResultUploaded,
        TaskStatus.Disclosed,
    ]
    running_states = await task_state_cache.find(status=running_status)
    if len(running_states) > 0:
        status = "running"
    else:
        status = "idle"

    now = datetime.now().astimezone()
    date = now.date()
    today_ts = time.mktime(date.timetuple())
    today = datetime.fromtimestamp(today_ts)

    today_states = await task_state_cache.find(start=today, status=[TaskStatus.Success])

    total_states = await task_state_cache.find(status=[TaskStatus.Success])

    return TaskStats(
        status=status, num_today=len(today_states), num_total=len(total_states)
    )
