import argparse
import json
import os
from datetime import datetime
from contextlib import ExitStack
from typing import Optional

import pydantic
from websockets.sync.client import connect as websocket_connect
from websockets.sync.connection import Connection
from gpt_task.inference import run_task as gpt_run_task
from gpt_task.models import GPTTaskArgs
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from sd_task.inference_task_runner.inference_task import run_task as sd_run_task


def sd_inference(output_dir: str, task_args_str: str):
    print(f"Inference task starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    task_args = InferenceTaskArgs.model_validate_json(task_args_str)
    imgs = sd_run_task(task_args)
    for i, img in enumerate(imgs):
        dst = os.path.join(output_dir, f"{i}.png")
        img.save(dst)
    print(f"Inference task finishes at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


class WebsocketStreamHandler(object):
    def __init__(self, url: str) -> None:
        self.url = url

        self._exit_stack = ExitStack()

        self._sock: Optional[Connection] = None

    def __enter__(self):
        assert self._sock is None, "Websocket has already connected"
        self._sock = self._exit_stack.enter_context(websocket_connect(self.url))

        return self

    def __exit__(self, *exc_details):
        res = self._exit_stack.__exit__(*exc_details)
        self._sock = None
        return res

    def handle(self, token: str):
        assert self._sock is not None, "Websocket has not connected"

        self._sock.send(token)

    def connect(self):
        assert self._sock is None, "Websocket has already connected"

        self._sock = websocket_connect(self.url)

    def close(self):
        assert self._sock is not None, "Websocket has not connected"

        self._sock.close()
        self._sock = None


def gpt_inference(
    output_dir: str, task_args_str: str, stream: bool = False, stream_url: str = ""
):
    print(f"Inference task starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    task_args = GPTTaskArgs.model_validate_json(task_args_str)
    if stream:
        assert stream_url != "", "Stream url is empty"
        with WebsocketStreamHandler(url=stream_url) as handler:
            resp = gpt_run_task(task_args, stream=True, stream_handle=handler.handle)
    else:
        resp = gpt_run_task(task_args, stream=False)

    dst = os.path.join(output_dir, "response.json")
    with open(dst, mode="w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False)
    print(f"Inference task finishes at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_type", type=int, choices=[0, 1], required=True)
    parser.add_argument("output_dir", type=str, required=True)
    parser.add_argument("task_args", type=str, required=True)
    parser.add_argument("-stream", action="store_true", required=False)
    parser.add_argument("-stream_url", type=str, required=False, default="")
    args = parser.parse_args()

    task_type: int = args.task_type
    output_dir: str = args.output_dir
    task_args_str: str = args.task_args
    stream: bool = args.stream
    stream_url = ""
    if stream:
        stream_url: str = args.stream_url
        if stream_url == "":
            raise ValueError("Missing argument -stream_url")

    try:
        print(
            f"Inference task starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if task_type == 0:
            # sd task
            sd_inference(output_dir, task_args_str)
        elif task_type == 1:
            # gpt task
            gpt_inference(output_dir, task_args_str, stream=stream, stream_url=stream_url)
        else:
            raise ValueError(f"unknown task type {task_type}")

        print(
            f"Inference task finishes at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except pydantic.ValidationError:
        print("Task args invalid")
        raise


if __name__ == "__main__":
    main()
