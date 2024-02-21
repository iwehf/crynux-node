from typing import Union

from fastapi import APIRouter, FastAPI

from .account import router as account_router
from .node import router as node_router
from .system import router as system_router
from .task import router as task_router


def include_router(app: Union[FastAPI, APIRouter], headless: bool, prefix: str = ""):
    router = APIRouter(prefix="/v1")
    router.include_router(task_router)
    if not headless:
        router.include_router(account_router)
        router.include_router(node_router)
        router.include_router(system_router)

    app.include_router(router, prefix=prefix)
