import json
import secrets
import os
import shutil
from io import BytesIO
from typing import List, Callable, Awaitable, Optional
from functools import partial

import pytest
from anyio import create_task_group, Event, from_thread
from eth_account import Account
from PIL import Image
from web3 import Web3

from crynux_server import models
from crynux_server.config import Config, TxOption, set_config, TaskConfig
from crynux_server.contracts import Contracts
from crynux_server.event_queue import EventQueue, MemoryEventQueue
from crynux_server.node_manager import (
    NodeManager,
    NodeStateManager,
)
from crynux_server.node_manager.state_cache import (
    MemoryNodeStateCache,
    MemoryTxStateCache,
    ManagerStateCache,
)
from crynux_server.relay import MockRelay, Relay
from crynux_server.task import InferenceTaskRunner, MemoryTaskStateCache, TaskSystem
from crynux_server.task.state_cache import TaskStateCache
from crynux_server.utils import get_task_hash
from crynux_server.watcher import EventWatcher, MemoryBlockNumberCache
from crynux_worker.task.utils import get_image_hash, get_gpt_resp_hash


@pytest.fixture
def tx_option():
    return {}


@pytest.fixture
def privkeys():
    return [
        "0xa627246a109551432ac5db6535566af34fdddfaa11df17b8afd53eb987e209a2",
        "0xb171f296622b98cbdc08dcdcb0696f738c3a22d9d367c657117cd3c8d0b71d42",
        "0x8fb2fc9862b93b5b75cda8202f583711201e4cba5459eefe442b8c5dcc4bdab9",
    ]


@pytest.fixture
def gpu_name():
    return "NVIDIA GeForce GTX 1070 Ti"


@pytest.fixture
def gpu_vram():
    return 8


@pytest.fixture
async def root_contracts(tx_option, privkeys):
    from web3.providers.eth_tester import AsyncEthereumTesterProvider

    provider = AsyncEthereumTesterProvider()
    c0 = Contracts(provider=provider, default_account_index=0)

    await c0.init(option=tx_option)

    waiter = await c0.node_contract.update_task_contract_address(
        c0.task_contract.address, option=tx_option
    )
    await waiter.wait()

    for privkey in privkeys:
        provider.ethereum_tester.add_account(privkey)
        account = Account.from_key(privkey)
        amount = Web3.to_wei(1000, "ether")
        await c0.transfer(account.address, amount, option=tx_option)

    return c0


@pytest.fixture
def config():
    test_config = Config.model_validate(
        {
            "log": {"dir": "logs", "level": "INFO"},
            "ethereum": {
                "privkey": "",
                "provider": "",
                "chain_id": None,
                "gas": None,
                "gas_price": None,
                "contract": {"token": "", "node": "", "task": ""},
            },
            "task_dir": "task",
            "db": "",
            "relay_url": "",
            "celery": {"broker": "", "backend": ""},
            "distributed": False,
            "task_config": {
                "output_dir": "build/data/images",
                "hf_cache_dir": "build/data/huggingface",
                "external_cache_dir": "build/data/external",
                "inference_logs_dir": "build/data/inference-logs",
                "script_dir": "stable-diffusion-task",
                "result_url": "",
            },
        }
    )
    set_config(test_config)
    return test_config


@pytest.fixture
async def node_contracts(
    root_contracts: Contracts, tx_option: TxOption, privkeys: List[str]
):
    token_contract_address = root_contracts.token_contract.address
    node_contract_address = root_contracts.node_contract.address
    task_contract_address = root_contracts.task_contract.address

    cs = []
    for privkey in privkeys:
        contracts = Contracts(provider=root_contracts.provider, privkey=privkey)
        await contracts.init(
            token_contract_address=token_contract_address,
            node_contract_address=node_contract_address,
            task_contract_address=task_contract_address,
            option=tx_option,
        )
        amount = Web3.to_wei(1000, "ether")
        if (await contracts.token_contract.balance_of(contracts.account)) < amount:
            waiter = await root_contracts.token_contract.transfer(
                contracts.account, amount, option=tx_option
            )
            await waiter.wait()
        task_amount = Web3.to_wei(400, "ether")
        if (
            await contracts.token_contract.allowance(task_contract_address)
        ) < task_amount:
            waiter = await contracts.token_contract.approve(
                task_contract_address, task_amount, option=tx_option
            )
            await waiter.wait()
        node_amount = Web3.to_wei(400, "ether")
        if (
            await contracts.token_contract.allowance(node_contract_address)
        ) < node_amount:
            waiter = await contracts.token_contract.approve(
                node_contract_address, node_amount, option=tx_option
            )
            await waiter.wait()

        cs.append(contracts)
    return cs


@pytest.fixture
def relay():
    return MockRelay()


def mock_run_task(
    queue: EventQueue,
    distributed: bool,
    task_id: int,
    task_type: models.TaskType,
    task_args: str,
    task_config: Optional[TaskConfig] = None,
    stream: bool = False,
):
    assert task_config is not None
    result_dir = os.path.abspath(os.path.join(task_config.output_dir, str(task_id)))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    if task_type == models.TaskType.SD:
        img = Image.new("RGB", (512, 512), (255, 255, 255))
        dst = os.path.join(result_dir, "test.png")
        img.save(dst, "PNG")
        result_hash = get_image_hash(dst)
    else:
        res = {
            "model": "gpt2",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": '\n\nI have a chat bot, called "Eleanor" which was developed by my team on Skype. '
                        "The only thing I will say is this",
                    },
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 30, "total_tokens": 41},
        }
        dst = os.path.join(result_dir, "test.json")
        with open(dst, mode="w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        result_hash = get_gpt_resp_hash(dst)
    
    event = models.TaskResultReady(
        task_id=task_id,
        hashes=[result_hash],
        files=[dst]
    )

    from_thread.run(queue.put, event)


@pytest.fixture
async def create_node_managers(
    privkeys: List[str],
    node_contracts: List[Contracts],
    relay: Relay,
    config: Config,
    gpu_name: str,
    gpu_vram: int,
):
    new_data_dirs = []

    async def make_node_managers(fail_step: int):
        managers = []

        for i, (privkey, contracts) in enumerate(zip(privkeys, node_contracts)):
            queue = MemoryEventQueue()

            watcher = EventWatcher.from_contracts(contracts)
            block_number_cache = MemoryBlockNumberCache()
            watcher.set_blocknumber_cache(block_number_cache)

            def make_callback(queue):
                async def _push_event(event_data):
                    event = models.load_event_from_contracts(event_data)
                    await queue.put(event)

                return _push_event

            watcher.watch_event(
                "task",
                "TaskCreated",
                callback=make_callback(queue),
                filter_args={"selectedNode": contracts.account},
            )

            task_state_cache = MemoryTaskStateCache()
            system = TaskSystem(
                task_state_cache,
                queue=queue,
                distributed=config.distributed,
                retry=(fail_step > 0),
            )

            assert config.task_config is not None
            local_config = config.task_config.model_copy()
            data_dir = os.path.join(local_config.output_dir, f"node_{fail_step}_{i}")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            local_config.output_dir = data_dir
            new_data_dirs.append(data_dir)

            def make_runner_cls(contracts, queue, relay, watcher, local_config, fail_step):
                _run_task = partial(mock_run_task, queue=queue)

                class _InferenceTaskRunner(InferenceTaskRunner):
                    def __init__(
                        self,
                        task_id: int,
                        distributed: bool,
                        state_cache: TaskStateCache,
                        queue: EventQueue,
                    ) -> None:
                        super().__init__(
                            task_id=task_id,
                            distributed=distributed,
                            state_cache=state_cache,
                            queue=queue,
                            task_func=_run_task,
                            contracts=contracts,
                            relay=relay,
                            watcher=watcher,
                            local_config=local_config,
                        )
                        self._fail_count = 0

                    async def task_created(self, event, finish_callback):
                        if self._fail_count == 0 and fail_step == 1:
                            self._fail_count += 1
                            raise ValueError("mock fail")
                        return await super().task_created(event, finish_callback)

                    async def result_ready(self, event, finish_callback):
                        if self._fail_count == 0 and fail_step == 2:
                            self._fail_count += 1
                            raise ValueError("mock fail")
                        return await super().result_ready(event, finish_callback)

                    async def commitment_ready(self, event, finish_callback):
                        if self._fail_count == 0 and fail_step == 3:
                            self._fail_count += 1
                            raise ValueError("mock fail")
                        return await super().commitment_ready(event, finish_callback)

                    async def task_success(self, event, finish_callback):
                        if self._fail_count == 0 and fail_step == 4:
                            self._fail_count += 1
                            raise ValueError("mock fail")
                        return await super().task_success(event, finish_callback)

                    async def task_aborted(self, event, finish_callback):
                        if self._fail_count == 0 and fail_step == 5:
                            self._fail_count += 1
                            raise ValueError("mock fail")
                        return await super().task_aborted(event, finish_callback)

                return _InferenceTaskRunner

            system.set_runner_cls(
                make_runner_cls(contracts, queue, relay, watcher, local_config, fail_step)
            )

            state_cache = ManagerStateCache(
                node_state_cache_cls=MemoryNodeStateCache,
                tx_state_cache_cls=MemoryTxStateCache,
            )
            # set init state to stopped to bypass prefetch stage
            await state_cache.set_node_state(models.NodeStatus.Stopped)

            state_manager = NodeStateManager(
                state_cache=state_cache,
                contracts=contracts,
            )

            manager = NodeManager(
                config=config,
                gpu_name=gpu_name,
                gpu_vram=gpu_vram,
                manager_state_cache=state_cache,
                node_state_manager=state_manager,
                privkey=privkey,
                event_queue=queue,
                contracts=contracts,
                relay=relay,
                watcher=watcher,
                task_system=system,
                retry=False,
            )
            managers.append(manager)

        return managers

    try:
        yield make_node_managers
    finally:
        for data_dir in new_data_dirs:
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)


async def create_task(
    task_type: models.TaskType, contracts: Contracts, relay: Relay, tx_option: TxOption
):
    if task_type == models.TaskType.SD:
        prompt = (
            "best quality, ultra high res, photorealistic++++, 1girl, off-shoulder sweater, smiling, "
            "faded ash gray messy bun hair+, border light, depth of field, looking at "
            "viewer, closeup"
        )

        negative_prompt = (
            "paintings, sketches, worst quality+++++, low quality+++++, normal quality+++++, lowres, "
            "normal quality, monochrome++, grayscale++, skin spots, acnes, skin blemishes, "
            "age spot, glans"
        )

        args = {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "task_config": {"num_images": 9, "safety_checker": False},
        }
        task_args = json.dumps(args)

        task_hash = get_task_hash(task_args)
        data_hash = bytes([0] * 32)
    else:
        messages = [
            {"role": "user", "content": "I want to create a chat bot. Any suggestions?"}
        ]
        args = {
            "model": "gpt2",
            "messages": messages,
            "generation_config": {
                "max_new_tokens": 30,
            },
            "seed": 42,
        }

        task_args = json.dumps(args)

        task_hash = get_task_hash(task_args)
        data_hash = bytes([0] * 32)

    waiter = await contracts.task_contract.create_task(
        task_type=task_type,
        task_hash=task_hash,
        data_hash=data_hash,
        vram_limit=8,
        option=tx_option,
    )
    receipt = await waiter.wait()

    events = await contracts.task_contract.get_events(
        "TaskCreated",
        from_block=receipt["blockNumber"],
    )
    event = events[0]
    task_id = event["args"]["taskId"]
    await relay.create_task(task_id=task_id, task_args=task_args)

    round_map = {
        event["args"]["selectedNode"]: event["args"]["round"] for event in events
    }
    return task_id, round_map, receipt["blockNumber"]


async def start_nodes(
    node_managers: List[NodeManager], gpu_name: str, gpu_vram: int, tx_option
):
    waits = []
    for m in node_managers:
        assert m._node_state_manager is not None
        waits.append(
            await m._node_state_manager.start(
                gpu_name=gpu_name, gpu_vram=gpu_vram, option=tx_option
            )
        )
    for n in node_managers:
        assert (await n.state_cache.get_tx_state()).status == models.TxStatus.Pending
    async with create_task_group() as sub_tg:
        for w in waits:
            sub_tg.start_soon(w)
    for n in node_managers:
        assert (
            await n.state_cache.get_node_state()
        ).status == models.NodeStatus.Running


@pytest.mark.parametrize("fail_step", [0, 1, 2, 3])
@pytest.mark.parametrize("task_type", [models.TaskType.SD, models.TaskType.LLM])
async def test_node_manager(
    create_node_managers: Callable[[int], Awaitable[List[NodeManager]]],
    node_contracts: List[Contracts],
    relay: Relay,
    tx_option,
    gpu_name: str,
    gpu_vram: int,
    fail_step: int,
    task_type: models.TaskType,
):
    node_managers = await create_node_managers(fail_step)

    async with create_task_group() as tg:
        for n in node_managers:
            tg.start_soon(n.run, False)

        await start_nodes(node_managers, gpu_name, gpu_vram, tx_option)

        task_id, _, _ = await create_task(
            task_type, node_contracts[0], relay, tx_option=tx_option
        )

        if task_type == models.TaskType.SD:
            with BytesIO() as dst:
                await relay.get_sd_result(task_id=task_id, index=0, dst=dst)
                dst.seek(0)
                img = Image.open(dst)
                assert img.width == 512
                assert img.height == 512
        else:
            resp = await relay.get_gpt_result(task_id=task_id)
            assert resp.model == "gpt2"
            assert len(resp.choices) == 1
            assert len(resp.choices[0].message.content) > 0

        waits = []
        for m in node_managers:
            assert m._node_state_manager is not None
            waits.append(await m._node_state_manager.pause(option=tx_option))
        for n in node_managers:
            assert (
                await n.state_cache.get_tx_state()
            ).status == models.TxStatus.Pending
        async with create_task_group() as sub_tg:
            for w in waits:
                sub_tg.start_soon(w)

        for n in node_managers:
            assert (
                await n.state_cache.get_node_state()
            ).status == models.NodeStatus.Paused

        waits = []
        for m in node_managers:
            assert m._node_state_manager is not None
            waits.append(await m._node_state_manager.resume(option=tx_option))
        for n in node_managers:
            assert (
                await n.state_cache.get_tx_state()
            ).status == models.TxStatus.Pending
        async with create_task_group() as sub_tg:
            for w in waits:
                sub_tg.start_soon(w)

        for n in node_managers:
            assert (
                await n.state_cache.get_node_state()
            ).status == models.NodeStatus.Running

        waits = []
        for m in node_managers:
            assert m._node_state_manager is not None
            waits.append(await m._node_state_manager.stop(option=tx_option))
        for n in node_managers:
            assert (
                await n.state_cache.get_tx_state()
            ).status == models.TxStatus.Pending
        async with create_task_group() as sub_tg:
            for w in waits:
                sub_tg.start_soon(w)

        for n in node_managers:
            assert (
                await n.state_cache.get_node_state()
            ).status == models.NodeStatus.Stopped

        for n in node_managers:
            await n.finish()
        tg.cancel_scope.cancel()


@pytest.mark.parametrize("task_type", [models.TaskType.SD, models.TaskType.LLM])
async def test_node_manager_auto_cancel(
    root_contracts: Contracts,
    create_node_managers: Callable[[int], Awaitable[List[NodeManager]]],
    node_contracts: List[Contracts],
    relay: Relay,
    tx_option,
    gpu_name: str,
    gpu_vram: int,
    task_type: models.TaskType,
):
    node_managers = await create_node_managers(0)
    try:
        await root_contracts.task_contract.update_timeout(1, option=tx_option)

        async with create_task_group() as tg:
            tg.start_soon(node_managers[0].run, False)

            await start_nodes(node_managers, gpu_name, gpu_vram, tx_option)

            task_id, _, _ = await create_task(
                task_type, node_contracts[0], relay, tx_option=tx_option
            )

            cancel_event = Event()

            async def _cancel(_):
                cancel_event.set()

            assert node_managers[0]._watcher is not None

            node_managers[0]._watcher.watch_event(
                "task",
                "TaskAborted",
                callback=_cancel,
                filter_args={"taskId": task_id},
            )

            assert node_managers[0]._task_system is not None

            await cancel_event.wait()

            state = await node_managers[0]._task_system.state_cache.load(
                task_id=task_id
            )
            assert state.status == models.TaskStatus.Aborted

            await node_managers[0].finish()
            tg.cancel_scope.cancel()
    finally:
        await root_contracts.task_contract.update_timeout(900, option=tx_option)


async def partial_run_task(
    config: Config,
    node_managers: List[NodeManager],
    node_contracts: List[Contracts],
    relay: Relay,
    tx_option,
    gpu_name: str,
    gpu_vram: int,
    stage: int,
    task_type: models.TaskType,
):
    # start nodes
    await start_nodes(node_managers, gpu_name, gpu_vram, tx_option)

    # create task
    task_id, round_map, block_number = await create_task(
        task_type, node_contracts[0], relay, tx_option=tx_option
    )

    result = bytes.fromhex("0102030405060708")
    if 1 <= stage:
        # submit task result commitment
        for c in node_contracts:
            nonce = secrets.token_bytes(32)
            commitment = Web3.solidity_keccak(["bytes", "bytes32"], [result, nonce])

            waiter = await c.task_contract.submit_task_result_commitment(
                task_id, round_map[c.account], commitment, nonce, option=tx_option
            )
            receipt = await waiter.wait()
            block_number = receipt["blockNumber"]
        events = await node_contracts[0].task_contract.get_events(
            "TaskResultCommitmentsReady", from_block=block_number
        )
        assert len(events) == 1
        event = events[0]
        assert event["args"]["taskId"] == task_id
        config.last_result = "0x0102030405060708"

    if 2 <= stage:
        # disclose task
        from_block = block_number
        for c in node_contracts:
            waiter = await c.task_contract.disclose_task_result(
                task_id=task_id,
                round=round_map[c.account],
                result=result,
                option=tx_option,
            )
            receipt = await waiter.wait()
            block_number = receipt["blockNumber"]
        to_block = block_number
        events = await node_contracts[0].task_contract.get_events(
            "TaskSuccess", from_block=from_block, to_block=to_block
        )
        assert len(events) == 1
        event = events[0]
        assert event["args"]["taskId"] == task_id
        assert event["args"]["result"] == result

    return task_id

@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("task_type", [models.TaskType.SD, models.TaskType.LLM])
async def test_node_manager_with_recover(
    config: Config,
    create_node_managers: Callable[[int], Awaitable[List[NodeManager]]],
    node_contracts: List[Contracts],
    relay: Relay,
    tx_option,
    gpu_name: str,
    gpu_vram: int,
    stage: int,
    task_type: models.TaskType,
):
    node_managers = await create_node_managers(0)
    task_id = await partial_run_task(
        config=config,
        node_managers=node_managers,
        node_contracts=node_contracts,
        relay=relay,
        tx_option=tx_option,
        gpu_name=gpu_name,
        gpu_vram=gpu_vram,
        stage=stage,
        task_type=task_type,
    )
    async with create_task_group() as tg:
        for n in node_managers:
            tg.start_soon(n.run, False)

        for n in node_managers:
            assert (
                await n.state_cache.get_node_state()
            ).status == models.NodeStatus.Running

        if task_type == models.TaskType.SD:
            with BytesIO() as dst:
                await relay.get_sd_result(task_id=task_id, index=0, dst=dst)
                dst.seek(0)
                img = Image.open(dst)
                assert img.width == 512
                assert img.height == 512
        else:
            resp = await relay.get_gpt_result(task_id=task_id)
            assert resp.model == "gpt2"
            assert len(resp.choices) == 1
            assert len(resp.choices[0].message.content) > 0

        waits = []
        for m in node_managers:
            assert m._node_state_manager is not None
            waits.append(await m._node_state_manager.stop(option=tx_option))
        for n in node_managers:
            assert (
                await n.state_cache.get_tx_state()
            ).status == models.TxStatus.Pending
        async with create_task_group() as sub_tg:
            for w in waits:
                sub_tg.start_soon(w)

        for n in node_managers:
            assert (
                await n.state_cache.get_node_state()
            ).status == models.NodeStatus.Stopped

        for n in node_managers:
            await n.finish()
        tg.cancel_scope.cancel()
