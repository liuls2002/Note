# slime源码详解

> 源码仓库 https://github.com/THUDM/slime
本文完成时间2026/5/7


上一篇[大模型后训练中的 RL 系统：从 SFT、RLHF、RLVR 到 Agentic RL](./agentic-rl.md)里，我们把 LLM RL / Agentic RL 系统抽象成几类组件：

```text
数据与环境
  -> rollout / agent 交互
  -> reward / verifier
  -> 训练后端
  -> 权重同步
  -> 调度与容错
```

这篇不再重复 RLHF、RLVR、PPO/GRPO 的算法背景，而是看一个真实系统 slime 如何把这些抽象组件落到工程实现上。重点是两个问题：

1. slime 是如何基于 Ray 把训练侧和 rollout 侧启动起来的？
2. 系统启动后，同步和异步训练到底是怎么协同工作的？

## 1. slime 的定位：把训练、推理和调度拆开

![slime](./imgs/slime.png)

slime 是一个面向大模型 post-training / RL scaling 的训练框架。它最核心的工程取舍是：不要把训练、推理、奖励、环境交互塞进一个单体循环，而是拆成几层相对独立的服务：

```text
driver script: train.py / train_async.py
  负责全局流程：创建资源、启动服务、提交 rollout、提交 train、保存、评估、权重同步

Ray orchestration layer
  负责 placement group、Ray actor 生命周期、远程方法调用、资源隔离、故障恢复

Rollout side: RolloutManager + SGLangEngine + rollout function
  负责启动 SGLang router/server，执行生成、reward、动态过滤、样本回收

Train side: RayTrainGroup + MegatronTrainRayActor
  负责 Megatron 初始化、logprob/value/advantage 计算、反向传播、optimizer step、权重导出

Data side: Sample / Dataset / DataSource
  负责 prompt 加载、样本分组、buffer、partial rollout 回收
```

对应到上一篇文章里的抽象系统，slime 的映射大致是：


| 抽象模块              | slime 中的主要实现                                                                  |
| ----------------- | ----------------------------------------------------------------------------- |
| 数据集 / 任务源         | `Dataset`、`RolloutDataSource`、`RolloutDataSourceWithBuffer`                   |
| 轨迹容器              | `Sample`                                                                      |
| rollout 执行器       | `RolloutManager.generate()`、`generate_rollout()`、`generate_and_rm_group()`    |
| 推理服务              | `SGLangEngine` + `sglang_router`                                              |
| reward / verifier | `async_rm()`、`batched_async_rm()`、用户自定义 rollout / generate 函数                 |
| 训练执行器             | `RayTrainGroup`、`MegatronTrainRayActor.train()`                               |
| 权重同步              | `MegatronTrainRayActor.update_weights()`、`UpdateWeightFromTensor/Distributed` |
| 调度系统              | Ray placement group、Ray remote actor、driver loop                              |


agentic RL 的特殊性在 rollout 侧：普通 RLVR 的 rollout 可能只是 `prompt -> response -> reward`，而 agentic rollout 可能是：

```text
prompt
  -> model action
  -> tool / environment step
  -> observation
  -> model action
  -> ...
  -> final answer
  -> reward
```

slime 没有把“agent 环境协议”写死在框架里。框架只要求 rollout 函数最终返回 `Sample` 或 `list[list[Sample]]`，至于中间是单轮生成、多轮工具调用、浏览器环境、代码执行环境，还是外部 agent worker，都可以通过自定义 `rollout_function_path` 或 `custom_generate_function_path` 接入。

## 2. 数据部分：Sample、Dataset 与 DataSource

slime 的数据流从原始 prompt 数据开始：`Dataset` 先把 `.jsonl` / `.parquet` 中的每一行预处理成 prompt 级别的 `Sample`；随后 `DataSource` 像 dataloader 一样按 rollout 需要不断取出样本，处理 shuffle、epoch 边界和 buffer，并把每个 prompt 复制成 `n_samples_per_prompt` 个候选，封装成 `list[list[Sample]]` 的 group 结构；rollout 阶段会在这些 `Sample` 上补全 response、reward、logprob、loss mask 等信息；最后 `RolloutManager` 再把完成的 samples 转成按 data parallel rank 切分的 `RolloutBatch`，交给 Megatron 训练侧消费。

### 2.1 Sample：rollout 与 train 的共同语言

`Sample` 定义在 `slime/utils/types.py`。它是 rollout 和训练之间的协议对象，核心字段可以按用途分成几类：


| 字段                                              | 作用                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------- |
| `prompt`                                        | 原始 prompt，可以是字符串，也可以是 OpenAI messages 形式                            |
| `tokens`                                        | 完整 token 序列，通常是 prompt tokens + response tokens                     |
| `response` / `response_length`                  | 模型生成文本及其 token 长度                                                   |
| `reward`                                        | reward model、规则验证器或环境返回的奖励                                          |
| `loss_mask`                                     | 哪些 response token 参与 loss，agentic 场景尤其重要                            |
| `rollout_log_probs`                             | SGLang 生成时返回的 token logprob，可用于 off-policy correction / mismatch 统计 |
| `status`                                        | `PENDING`、`COMPLETED`、`TRUNCATED`、`ABORTED`、`FAILED`                |
| `metadata`                                      | 任务、环境、工具调用、调试信息等扩展字段                                                |
| `session_id`                                    | 给 router consistent hashing 用，让多轮会话尽量路由到同一 worker                   |
| `multimodal_inputs` / `multimodal_train_inputs` | 多模态 rollout 和训练所需的数据                                                |


agentic RL 中最容易出错的是 `loss_mask`。如果 response 中混入了 observation、tool result、环境状态等非模型动作 token，这些 token 通常不能参与 policy loss。slime 不强制规定 agent trajectory 的格式，但通过 `loss_mask` 给了自定义 rollout 足够的表达能力。

### 2.2 Dataset：原始输入数据的预处理层

`Dataset` 定义在 `slime/utils/data.py`，它的职责是把原始数据文件中的每一行预处理成 prompt 级别的 `Sample` 对象。这里的 `Sample` 还不是一次完整 rollout 轨迹，而是“待生成”的输入样本：已经有了 prompt、label、metadata、tools、多模态输入等信息，但还没有 response、reward、rollout logprob。

它做的预处理可以简要理解为：

- 读取 `.jsonl` 或 `.parquet` 数据。
- 从配置的字段中取出 prompt、label、metadata 和 tools。
- 在需要时把原始 prompt 转成对话格式，并应用 tokenizer 的 chat template。
- 在多模态任务中解析图片、视频等输入，构造后续 rollout / train 需要的多模态字段。
- 根据最大 prompt 长度过滤过长样本。
- 保存一份 `origin_samples`，并在需要时按 epoch 做 shuffle。

### 2.3 DataSource：rollout 侧的 dataloader

默认数据源是`slime.rollout.data_source.RolloutDataSourceWithBuffer`，如果说 `Dataset` 负责“把原始文件变成 `Sample`”，那么 `DataSource` 就更接近 PyTorch 里的 `DataLoader`：它负责在训练过程中不断给 rollout 提供下一批 prompt group，并维护取数状态。

`RolloutDataSource` 的核心职责包括：

- 按 `sample_offset` 从 `Dataset.samples` 中取出下一段 prompt samples。
- 处理 epoch 边界：当前 epoch 剩余样本不够时，切到下一个 epoch，并根据配置重新 shuffle。
- 将每个 prompt sample 复制 `n_samples_per_prompt` 份，形成同一个 prompt 的多个候选回答。
- 为每个 sample 填入全局递增的 `group_index` 和 `index`。
- 返回 `list[list[Sample]]`，也就是 rollout 函数消费的 group 格式。

它返回的结构可以理解为：

```text
DataSource.get_samples(num_prompts)
  -> [
       [prompt0_sample0, prompt0_sample1, ...],
       [prompt1_sample0, prompt1_sample1, ...],
       ...
     ]
```

这里外层 list 表示 prompt group，内层 list 表示同一个 prompt 的多个采样结果。例如 `rollout_batch_size=2`，`n_samples_per_prompt=4` 时：

```text
[
  [prompt0_sample0, prompt0_sample1, prompt0_sample2, prompt0_sample3],
  [prompt1_sample0, prompt1_sample1, prompt1_sample2, prompt1_sample3],
]
```

这正好对应 GRPO / GSPO 这类 group-based advantage estimator 的数据形态：同一个 prompt 下的多个 response 会在后续 reward normalization / advantage 计算中作为一组处理。

`RolloutDataSourceWithBuffer` 在此基础上增加了 buffer。它的设计很直接：

- `add_samples(samples)`：把一批 sample groups 放入 `self.buffer`。
- `get_samples(num_samples)`：优先从 buffer 里取 group；buffer 不够时，再从底层 `Dataset` 继续取新的 prompt。

这个 buffer 主要服务于 partial rollout、动态采样和 fully async rollout。比如某些 group 在生成中途被 abort，或者动态采样中暂时没有进入当前训练 batch，就可以先放回 buffer，后续再由 `get_samples()` 取出继续使用。这样 `DataSource` 就不只是静态数据读取器，而是 rollout 系统里的状态化数据入口。

## 3. Ray 基础与 placement group

在进入 rollout 和 train 之前，先补一点 slime 使用 Ray 的基本方式。slime 的 driver 进程就是运行 `train.py` / `train_async.py` 的主进程，它负责创建 Ray actor、提交远程任务和等待结果。被 `@ray.remote` 或 `ray.remote(...)` 包装的类会在 Ray worker 进程中运行；调用 `.remote()` 不会立刻在当前线程执行函数体，而是向 Ray 提交任务并返回 `ObjectRef`。后续 `ray.get(ref)` 才会阻塞等待远程结果。

slime 里有三类 Ray 对象尤其重要：

- Ray actor：例如 `RolloutManager`、`SGLangEngine`、`MegatronTrainRayActor`。它们是独立 Ray worker 进程中的有状态对象。
- Ray object store：rollout 生成出的训练数据会用 `ray.put` 放入 object store，再由训练 actor 按 DP rank 取回。
- Placement group：把一组 CPU/GPU bundle 预先锁住，确保后续 Ray actor 按 slime 需要的拓扑落到指定 GPU 上。

还有一点容易混淆：Ray actor 进程不等同于真正的推理或训练进程。`MegatronTrainRayActor` 本身就是训练进程中的一个 Megatron rank；而 `SGLangEngine` 是一个 Ray wrapper actor，它内部还会再启动一个 SGLang HTTP server 子进程。后面讲 rollout 时会展开这层关系。

### 3.1 create_placement_groups：先确定资源版图

slime 的启动入口一般是：

```text
train.py / train_async.py
  -> create_placement_groups(args)
  -> create_rollout_manager(args, pgs["rollout"])
  -> create_training_models(args, pgs, rollout_manager)
  -> actor_model.update_weights()
```

`create_placement_groups(args)` 的职责是先把本次任务需要的 GPU 资源锁住，并切分出 actor / rollout / critic 各自应该使用的 bundle 范围。

它会先根据运行模式计算总 GPU 数：

```text
debug_train_only:
  只需要 actor GPU

debug_rollout_only:
  只需要 rollout GPU

colocate:
  actor 和 rollout 共用 actor GPU

非 colocate:
  actor GPU + rollout GPU
```

然后 `_create_placement_group(num_gpus)` 创建 `num_gpus` 个 bundle，每个 bundle 大致是 `{"GPU": 1, "CPU": 1}`。Ray 会为这些 bundle 分配实际节点和 GPU，但默认顺序不一定符合训练系统想要的稳定拓扑。于是 slime 临时启动一批 `InfoActor`，查询每个 bundle 实际落在哪个 node / GPU 上，再按 node 和 GPU id 排序，得到：

```text
reordered_bundle_indices: 逻辑 GPU 顺序 -> Ray placement group bundle index
reordered_gpu_ids:       逻辑 GPU 顺序 -> 物理 GPU id
```

最后返回：

```python
{
    "actor": (pg, actor_bundle_indices, actor_gpu_ids),
    "rollout": (pg, rollout_bundle_indices, rollout_gpu_ids),
    "critic": ...,
}
```

这个返回值会贯穿后续启动流程。rollout 侧创建 `SGLangEngine` 时用 `pgs["rollout"]`，训练侧创建 `MegatronTrainRayActor` 时用 `pgs["actor"]`。因此 placement group 是 slime 系统的资源版图：它不负责训练或生成，但决定了每个远程组件最终跑在哪张 GPU 上。

## 4. rollout 部分：推理服务、路由和生成流程

rollout 侧可以先建立一个整体印象：`RolloutManager` 是 Ray actor，负责控制 rollout；`RolloutServer` / `ServerGroup` 是普通 Python dataclass，存在于 `RolloutManager` 进程内，用来记录 router 和 engine 句柄；`SGLangEngine` 是 Ray actor，负责控制某个 SGLang server；真正执行推理的是 `SGLangEngine` 内部用 `multiprocessing.Process` 启动的 SGLang HTTP server；router 也是 `RolloutManager` 进程里用 `multiprocessing.Process` 启动的独立进程。

它们之间的关系可以概括为：

```text
driver process
  -> RolloutManager Ray actor process
       -> DataSource object
       -> rollout function object
       -> RolloutServer / ServerGroup objects
       -> router process
       -> SGLangEngine Ray actor handles
            -> SGLang HTTP server child process
```

一次默认生成的请求路径则是：

```text
Sample
  -> generate() 构造 HTTP payload
  -> HTTP POST router /generate
  -> router 选择一个 SGLang worker
  -> HTTP 转发到 SGLang server
  -> SGLang server 生成 tokens/logprobs/meta_info
  -> HTTP response 回到 generate()
  -> 更新 Sample.tokens / response / reward / rollout_log_probs / status
```

### 4.1 核心组件概览

rollout 侧最重要的类和函数有这些：


| 组件                                                       | 运行位置                            | 作用                                                         |
| -------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------- |
| `RolloutManager`                                         | Ray actor 进程                    | rollout 控制面，持有 data source、rollout 函数、router/engine 句柄     |
| `RolloutServer`                                          | `RolloutManager` 进程中的普通对象       | 描述一个模型服务，包含 router 地址和若干 server group                      |
| `ServerGroup`                                            | `RolloutManager` 进程中的普通对象       | 描述一组同构 SGLang engines，例如 regular / prefill / decode        |
| `SGLangEngine`                                           | Ray actor 进程                    | 控制一个 SGLang server，负责 init、权重更新、pause/continue、flush cache |
| SGLang HTTP server                                       | `SGLangEngine` 启动的子进程           | 真正执行推理和加载权重                                                |
| `sglang_router`                                          | `RolloutManager` 启动的子进程         | 接收 `/generate` 请求，做负载均衡和路由                                 |
| `generate_rollout`                                       | `RolloutManager` 进程中的 Python 函数 | 默认 rollout 函数入口                                            |
| `generate_and_rm_group` / `generate_and_rm` / `generate` | rollout 函数内部调用                  | 组织并发、请求推理、补 reward、更新 sample                               |


router 的作用不是生成 token，而是把多个 SGLang workers 包成一个统一入口。默认 `generate()` 只需要访问 `http://router/generate`，不需要知道后面有多少 engine、每个 engine 在哪个端口、是否是 PD disaggregation 的 prefill/decode 结构。若启用 consistent hashing，`Sample.session_id` 会通过 `X-SMG-Routing-Key` header 传给 router，让同一会话尽量传到同一个 worker，以提高 prefix cache 命中率。

### 4.2 启动流程：从 create_rollout_manager 到 SGLang server

driver 调用：

```python
rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
```

这一步首先创建一个不占 GPU 的 Ray actor：

```python
RolloutManager.options(num_cpus=1, num_gpus=0).remote(args, pg)
```

`RolloutManager.__init__()` 在它自己的 Ray actor 进程里执行，主要完成四件事：

```text
1. load_function(args.data_source_path)
   -> 创建 self.data_source

2. load_function(args.rollout_function_path / eval_function_path)
   -> 创建 self.generate_rollout / self.eval_generate_rollout

3. start_rollout_servers(args, pg)
   -> 启动 router 和 SGLang engines

4. 创建 rollout_engine_lock 和可选 health monitors
```

`start_rollout_servers(args, pg)` 再继续展开：

```text
_resolve_sglang_config(args)
  -> 得到一个或多个 ModelConfig

对每个 model:
  -> _start_router(args)
       -> multiprocessing.Process(target=run_router)
       -> 独立 router 进程开始监听 HTTP

  -> 为每个 ServerGroup 调 start_engines()
       -> 创建 SGLangEngine Ray actor
       -> 根据 placement group 绑定 GPU bundle
       -> 分配 server port / nccl port / dist_init_addr
       -> engine.init.remote(...)
       -> ray.get(init_handles) 等待 server ready
```

`SGLangEngine.init()` 在 `SGLangEngine` Ray actor 进程里运行。它会把 slime 参数整理成 SGLang 的 `ServerArgs`，然后调用 `launch_server_process(ServerArgs(...))`。这一步会启动一个新的 SGLang HTTP server 子进程，并等待 `/health_generate` 通过。server 健康后，`SGLangEngine` 再通过 HTTP 把自己的 server URL 注册到 router：

```text
SGLangEngine Ray actor process
  -> launch_server_process()
     -> SGLang HTTP server child process
  -> POST router /workers
```

所以 rollout 启动完成时，各组件的状态是：

- driver 进程只持有 `rollout_manager` 这个 Ray actor handle。
- `RolloutManager` Ray actor 进程常驻，里面保存 DataSource、rollout 函数和 server metadata。
- router 是一个独立子进程，监听统一 HTTP 入口。
- 每个 `SGLangEngine` 是独立 Ray actor，负责控制对应 server。
- 每个真正推理的 SGLang server 是 `SGLangEngine` 拉起的独立子进程。

### 4.3 工作流程：RolloutManager.generate 如何组织一批数据

训练循环里，driver 会远程调用：

```python
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
```

`RolloutManager.generate()` 在 rollout manager actor 进程中执行：

```text
health_monitoring_resume()
  -> _get_rollout_data(rollout_id)
     -> call_rollout_fn(self.generate_rollout, args, rollout_id, self.data_source)
  -> _log_rollout_data()
  -> _convert_samples_to_train_data()
  -> _split_train_data_by_dp()
  -> return list[Box(ray.put(dp_partition))]
```

默认 `self.generate_rollout` 指向 `slime.rollout.sglang_rollout.generate_rollout`。它不是 Ray actor，也不是独立进程，而是被 `RolloutManager` 在自己的进程里调用的普通 Python 函数。这个函数内部用 `run(...)` 把 `generate_rollout_async(...)` 提交到后台 asyncio event loop 线程中执行，因此默认 rollout 内部可以并发处理多个 group / sample。

默认 rollout 的批量组织方式是：

```text
generate_rollout()
  -> run(generate_rollout_async(args, rollout_id, data_source.get_samples))

generate_rollout_async()
  -> 从 DataSource 获取 prompt groups
  -> state.submit_generate_tasks(groups)
  -> 每个 group 变成一个 generate_and_rm_group task
  -> asyncio.wait 等完成的 group
  -> 动态过滤 / 收集有效 group
  -> 收满 rollout_batch_size 后 abort 剩余 pending
  -> 返回 RolloutFnTrainOutput(samples=data)
```

`RolloutManager` 拿到 `samples` 后，会 flatten group，按全局 batch 大小做必要修剪，再转换成训练需要的 dict，例如 `tokens`、`response_lengths`、`rewards`、`loss_masks`、`rollout_log_probs` 等。最后根据训练侧 data parallel size 切分成多个 partition，用 `ray.put` 放入 Ray object store。训练 actor 后续按自己的 DP rank 取对应 partition。

### 4.4 一个 Sample 如何经过 router 和推理引擎

单条 sample 的默认路径在 `generate_and_rm_group -> generate_and_rm -> generate` 中完成。

`generate_and_rm_group(args, group, sampling_params)` 负责同一个 prompt 的多个候选回答：

```text
为 group 内每个 sample 分配 session_id
  -> 为每个 sample 创建 generate_and_rm task
  -> asyncio.gather 等待同组样本完成
  -> 如果 group_rm=True，对整个 group 做 batched reward
```

`generate_and_rm(args, sample, sampling_params)` 负责单条 sample 的生成和 reward：

```text
如果 sample 已完成:
  -> 直接返回

async with GenerateState.semaphore:
  -> 如果配置 custom_generate_function_path:
       调用户自定义 generate 函数
     else:
       调默认 generate()

如果不是 group_rm 且 sample.reward 为空:
  -> 调 async_rm() 补 reward
```

默认 `generate()` 是真正发 HTTP 请求的地方：

```text
1. 用 tokenizer / processor 准备 prompt_ids 或 multimodal payload
2. 构造 payload:
     sampling_params
     input_ids 或 image_data/text
     return_logprob=True
3. 如果使用 consistent hashing:
     header["X-SMG-Routing-Key"] = sample.session_id
4. HTTP POST http://{router_ip}:{router_port}/generate
5. router 按负载均衡 / session_id 选择 SGLang worker
6. SGLang server 生成并返回 text 和 meta_info
7. generate() 更新 sample:
     tokens += new_response_tokens
     response += output["text"]
     response_length += len(new_response_tokens)
     rollout_log_probs += new_response_log_probs
     status / weight_versions / prefix cache 统计等来自 meta_info
```

这里有两段 HTTP 很关键：

- rollout 函数到 router：`generate()` 通过 `http_utils.post` 访问 router 的 `/generate`。
- router 到 SGLang server：由 `sglang_router` 内部转发到已注册的 worker URL。

而 `SGLangEngine` Ray actor 不在每次生成 token 的热路径上。生成请求不经过 Ray 调 `SGLangEngine.generate.remote(...)`，而是直接走 HTTP router。`SGLangEngine` 更像控制面 wrapper：启动 server、注册 worker、flush cache、pause/continue generation、执行权重更新等。

### 4.5 rollout 可扩展点：自定义生成与异步控制

slime 对 agentic rollout 的扩展主要有两层：

- `custom_generate_function_path`：只替换单条 sample 的生成逻辑，保留默认的 group 并发、reward、filter、buffer 逻辑。多轮 tool-use、函数调用、环境交互通常放在这里。
- `rollout_function_path`：替换整个 rollout 函数。fully async rollout 就是这个层级的改造，它让后台 worker 持续生成，`generate_rollout` 只负责从队列里取已完成 group。

因此，slime 的 rollout 框架本身并不规定 agent 环境如何表达。只要最终把 response、reward、tokens、loss_mask 等信息补回 `Sample`，后续训练链路就能复用。

## 5. train 部分：训练引擎和权重更新

训练侧的核心是两件事：一是启动 Megatron 分布式训练引擎并进行训练，二是把训练后的 actor 权重同步回 rollout 侧的 SGLang engines。先给出整体结构：

```text
driver process
  -> RayTrainGroup object
       -> MegatronTrainRayActor Ray actor handles
            -> 每个 actor 是一个 Megatron rank
            -> 内部持有 model / optimizer / scheduler / TensorBackuper / weight_updater
```

`RayTrainGroup` 本身不是远程 actor，而是 driver 进程里的一个 Python 对象，里面保存了一组 `MegatronTrainRayActor` 的 Ray actor handles。真正执行训练的是这些 `MegatronTrainRayActor` 进程。

### 5.1 核心组件概览


| 组件                            | 运行位置            | 作用                                                               |
| ----------------------------- | --------------- | ---------------------------------------------------------------- |
| `RayTrainGroup`               | driver 进程中的普通对象 | 管理一组训练 Ray actors，提供 `async_init`、`async_train`、`update_weights` |
| `MegatronTrainRayActor`       | Ray actor 进程    | 一个 Megatron rank，持有模型、优化器、数据处理和权重同步逻辑                            |
| `TrainRayActor`               | 父类              | 设置分布式环境变量，初始化 torch process group                                |
| `TensorBackuper`              | 训练 actor 进程内    | 保存 actor / ref / old_actor / teacher 等权重副本                       |
| `UpdateWeightFromTensor`      | 训练 actor 进程内    | colocate 或部分 colocate 场景下的权重更新器                                  |
| `UpdateWeightFromDistributed` | 训练 actor 进程内    | 非 colocate 场景下通过 NCCL 向 rollout engines 传权重                      |


训练侧和 rollout 侧的连接点是：

```text
actor_model.set_rollout_manager(rollout_manager)
```

这个调用会把 `RolloutManager` 的 Ray actor handle 保存到每个 `MegatronTrainRayActor` 中。后续权重更新时，训练 actor 会通过它查询哪些 SGLang engines 可以更新，以及用于避免并发广播死锁的 lock。

### 5.2 启动流程：从 create_training_models 到 Megatron ranks

driver 调用：

```python
actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
```

`create_training_models` 先创建 actor 的 `RayTrainGroup`：

```text
allocate_train_group(...)
  -> RayTrainGroup(...)
     -> _allocate_gpus_for_actor(pg, num_gpus_per_actor)
        -> ray.remote(MegatronTrainRayActor)
        -> 按 rank 在 placement group bundle 上创建 actor
        -> rank0 生成 master_addr/master_port
        -> 其他 rank 复用同一组 master_addr/master_port
```

每个 `MegatronTrainRayActor` 是一个独立 Ray worker 进程，对应 Megatron 分布式训练中的一个 rank。创建 actor 时会设置 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK`、`LOCAL_RANK` 等环境变量。

actor 创建后，`create_training_models` 调：

```text
actor_model.async_init(...)
  -> 每个 MegatronTrainRayActor.init.remote(...)
```

`MegatronTrainRayActor.init()` 在每个训练 actor 进程中执行：

```text
TrainRayActor.init()
  -> torch.distributed.init_process_group()
  -> init_gloo_group()
  -> 设置 rank/world_size/local device

megatron init(args)
  -> 初始化 Megatron 并行上下文

initialize_model_and_optimizer()
  -> 构建 model / optimizer / scheduler
  -> 从 checkpoint 恢复训练状态

TensorBackuper.create()
  -> 备份 actor 权重
  -> 可选加载 ref / old_actor / teacher

create weight_updater
  -> colocate: UpdateWeightFromTensor
  -> non-colocate: UpdateWeightFromDistributed
```

最后 `create_training_models` 会调用：

```python
actor_model.set_rollout_manager(rollout_manager)
```

这样训练侧就知道 rollout 侧在哪里。初始化完成后，driver 会立即执行一次 `actor_model.update_weights()`，把训练 actor 当前权重推给 SGLang engines，保证第一个 rollout 使用的是训练侧恢复后的正确 policy。

### 5.3 训练流程：从 rollout batch 到 optimizer step

训练 loop 中，driver 调：

```python
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

`async_train` 的含义只是“异步提交 Ray 远程任务并返回 refs”，不是算法上的异步训练。它会对每个训练 actor 调：

```python
actor.train.remote(rollout_id, rollout_data_ref, external_data=...)
```

每个 `MegatronTrainRayActor.train()` 的主流程是：

```text
如果 offload_train:
  wake_up()

_get_rollout_data(rollout_data_ref)
  -> 根据自己的 DP rank 从 Ray object store 取 partition
  -> tokens / loss_masks / rollout_log_probs 等转成 GPU tensor

if role == critic:
  train_critic()
else:
  train_actor()

如果 offload_train:
  sleep()
```

actor 训练主路径 `train_actor()` 可以概括为：

```text
get_data_iterator()
  -> 根据 global batch / micro batch / dynamic batch 构造 Megatron data iterator

可选计算 ref logprob:
  -> _switch_model("ref")
  -> forward_only(get_log_probs_and_entropy, store_prefix="ref_")

可选计算 teacher logprob:
  -> _switch_model("teacher")
  -> forward_only(..., store_prefix="teacher_")

计算 actor / old_actor logprob:
  -> _switch_model("old_actor" 或 "actor")
  -> forward_only(get_log_probs_and_entropy)

如果使用 critic:
  -> 接收 critic 返回的 values

compute_advantages_and_returns()
  -> 按 GRPO / GSPO / PPO / R++ 等方法生成 advantages / returns

train()
  -> Megatron train_one_step
  -> forward / backward / optimizer step

weights_backuper.backup("actor")
  -> 保存最新 actor 权重，供后续 update_weights 使用
```

critic 路径类似，但目标是 value model：`train_critic()` 先 forward 出 values，计算 returns，然后以 `value_loss` 做训练，并把 values 返回给 actor 训练作为外部数据。

### 5.4 权重更新总览：训练端如何找到 rollout 端

每轮训练后，driver 会调用：

```python
actor_model.update_weights()
```

`RayTrainGroup.update_weights()` 会对每个训练 actor 调 `actor.update_weights.remote()`。在 `MegatronTrainRayActor.update_weights()` 内部，第一步是通过之前保存的 `rollout_manager` handle 查询 rollout 端：

```text
rollout_manager.get_updatable_engines_and_lock()
  -> rollout_engines:       可更新的 SGLangEngine Ray actor handles
  -> rollout_engine_lock:   一个 Ray lock，避免并发 broadcast 死锁
  -> num_new_engines:       容错恢复中新建的 engines 数量
  -> engine_gpu_counts:     每个 engine 占几张 GPU
  -> engine_gpu_offsets:    每个 engine 在 placement group 中的 GPU 偏移
```

如果有新 engine，或 offload / critic 场景需要重连，训练 actor 会调用：

```text
weight_updater.connect_rollout_engines(...)
```

这一步建立训练端和 rollout 端之间的传输关系。之后才进入真正的 `weight_updater.update_weights()`。

权重更新前后还有两个控制动作：

```text
pause_generation / flush_cache
  -> 防止生成过程中换权重
  -> 清理旧 KV/cache 状态

continue_generation
  -> 权重更新完成后恢复生成
```

这些控制动作是 Ray remote 调到 `SGLangEngine`，再由 `SGLangEngine` 通过 HTTP 调自己的 SGLang server endpoint 完成的。

### 5.5 模型格式转换：Megatron 权重如何变成 SGLang 可加载权重

训练侧模型是 Megatron 格式，rollout 侧 SGLang 通常按 HF 风格的权重名和张量形状加载。因此传输前需要做两类转换：

1. 并行切分还原：Megatron 参数可能被 TP / PP / EP 切分。更新器需要在合适的 rank 上把 TP shard all-gather 成完整张量；MoE expert 参数还要处理 expert parallel 的 gather。
2. 命名和形状转换：`named_params_and_buffers()` 先枚举 Megatron 全局参数名，`convert_to_hf()` 或 `HfWeightIteratorBase` 再把 Megatron 参数转换成 HF / SGLang 期望的名字和布局。

非 expert 参数和 expert 参数会分开处理。非 expert 参数通常是 TP gather 后转 HF；expert 参数还涉及 EP gather，再转成 HF。为了避免一次传输过大，slime 会按 `update_weight_buffer_size` 分 bucket 推送。

如果模型有 int4 / fp4 等压缩量化配置，权重更新前后还会触发 SGLang server 的 `post_process_weights`：更新前可能先恢复可加载状态，更新后再做量化后处理。

> 从 SGLang server 的角度看，slime 传过去的不是“一个 checkpoint 文件”，而是一批在线权重更新请求。请求里会先给出 metadata，例如权重名、dtype、shape、load format、weight version；server 根据这些 metadata 准备接收缓冲区，并把收到的 tensor 交给自己的 model runner / weight loader。对于张量并行的 rollout engine，server 端每个 rank 会按自己的模型并行切分规则加载需要的部分；因此 slime 侧需要传完整权重名和形状，而不只是传一块裸 tensor。真正的 tensor 数据可以来自 NCCL broadcast，也可以来自 CUDA IPC 句柄，但最终都会进入 SGLang 的在线 `update_weights` 流程，替换当前推理模型中的对应参数。

### 5.6 分布式传输链路：UpdateWeightFromDistributed

非 colocate 场景下，训练 GPU 和 rollout GPU 是分开的，slime 使用 `UpdateWeightFromDistributed` 通过 NCCL 把权重从 Megatron 训练进程传到 SGLang engines。

连接阶段：

```text
connect_rollout_engines()
  -> 选择 PP source rank:
       DP rank = 0 且 TP rank = 0
  -> 每个 PP rank 建一个 group_name:
       slime-pp_{pp_rank}
  -> connect_rollout_engines_from_distributed()
       -> 创建 NCCL group
       -> rank 0 是训练侧 source rank
       -> 其余 ranks 是各个 SGLang engine GPU
       -> SGLangEngine.init_weights_update_group.remote(...)
          -> HTTP /init_weights_update_group 到 SGLang server
```

这里不是“全模型都聚合到全局 train rank 0”。更准确地说，Megatron 的每个 pipeline stage 有自己的 source rank：这个 rank 满足 `DP=0`、`TP=0`，并负责该 PP stage 上的参数。对于 TP 切分的参数，source rank 会先通过 TP group all-gather 还原成 SGLang 需要的完整 HF-style tensor；对于 MoE expert 参数，还会额外处理 expert parallel 的 gather。这样每个 PP source rank 分桶推送自己负责的那部分权重，而不是单点聚合整个模型。

更新阶段：

```text
UpdateWeightFromDistributed.update_weights()
  -> weight_version += 1
  -> rank0 pause_generation + flush_cache

  非 expert 参数:
    -> named_params_and_buffers()
    -> all_gather_param() 还原 TP shard
    -> convert_to_hf()
    -> bucket 满后 _update_bucket_weights_from_distributed()

  expert 参数:
    -> all_gather_param() 还原 expert TP shard
    -> expert parallel all_gather
    -> convert_to_hf()
    -> bucket 推送

  -> rank0 continue_generation
```

每个 bucket 的传输链路是最关键的：

```text
训练 PP source rank
  -> Ray remote 调每个 SGLangEngine.update_weights_from_distributed(...)
       传 names / dtypes / shapes / group_name / weight_version 这些 metadata
  -> SGLangEngine 通过 HTTP 调本地 SGLang server:
       /update_weights_from_distributed
  -> 训练 rank 通过 NCCL dist.broadcast(param.data, 0, group=group)
       真正的 tensor 数据从训练 GPU 广播到 SGLang engine GPU
  -> SGLang server 根据 metadata 接收并加载 tensor
```

所以这里是“metadata 走 Ray + HTTP，tensor 数据走 NCCL”。rollout 端参与 NCCL group 的是 SGLang engine 内部的各个 GPU rank；它们会根据 SGLang server 收到的权重名、shape 和自身的 TP/PP 配置，把收到的权重加载到本 rank 对应的参数位置。可以粗略理解为：训练侧把 Megatron shard 还原并转换成 SGLang/HF 命名的权重块，通过 NCCL 广播给 rollout 侧相关 ranks，SGLang 再在 server 内部按推理并行策略切分和落位。

这也是为什么需要 `rollout_engine_lock`：多个训练 rank / PP group 同时广播时，如果 SGLang engines 进入 NCCL collective 的顺序不一致，容易死锁；lock 用来串行化关键广播段。

### 5.7 colocate 传输链路：UpdateWeightFromTensor

colocate 场景下，训练和 rollout 可能共享同一批 GPU。此时默认更新器是 `UpdateWeightFromTensor`。它仍然会做 HF 格式转换，但本地 colocated engines 的 tensor 传输方式不同：

```text
UpdateWeightFromTensor.update_weights()
  -> weight_version += 1
  -> pause_generation + flush_cache
  -> weights_getter() 取 TensorBackuper 中的 actor 权重
  -> HfWeightIteratorBase 产出 HF-format weight chunks
  -> _send_to_colocated_engine()
       -> FlattenedTensorBucket 打包张量
       -> MultiprocessingSerializer 序列化 CUDA IPC 信息
       -> dist.gather_object(..., backend="gloo") 汇集到 engine 对应 source rank
       -> Ray remote 调 SGLangEngine.update_weights_from_tensor(...)
       -> SGLangEngine 通过 HTTP 调 SGLang server:
            /update_weights_from_tensor
  -> continue_generation
```

这条链路可以理解为“本机 / 共享 GPU 场景尽量用 CUDA IPC 和 Ray IPC 传张量句柄，避免远程 NCCL 广播”。训练侧会按 bucket 产出 HF-style weight chunks。对于 colocated 的 engine，每个 engine 覆盖一组训练 ranks，这些 ranks 把自己构造好的 tensor bucket 序列化成 CUDA IPC 描述符，再通过 Gloo `gather_object` 汇集到该 engine 对应的 source rank，由这个 source rank 用 Ray remote 调 `SGLangEngine.update_weights_from_tensor(...)`。Ray/HTTP 传过去的主要是 metadata 和 CUDA IPC 句柄，tensor 数据本体仍留在 GPU 内存中，SGLang server 通过 IPC 打开这些 GPU tensor，再按自身 rank 需要加载对应参数。

因此，`UpdateWeightFromTensor` 中“少了一次远程 tensor 传输”，但并不是没有通信：控制信息、metadata、序列化后的 IPC handle 仍然会经过 Gloo、Ray 和 HTTP；只是大 tensor 数据不再像 distributed 路径那样跨训练 GPU 和 rollout GPU 做 NCCL broadcast。如果 placement group 中同时存在 colocated engines 和非 colocated engines，`UpdateWeightFromTensor` 也会把后半部分 engines 分出来，对它们复用 distributed NCCL 路径。

总结一下两种权重更新方式：


| 场景         | 更新器                                             | metadata 路径                               | tensor 路径                               |
| ---------- | ----------------------------------------------- | ----------------------------------------- | --------------------------------------- |
| 非 colocate | `UpdateWeightFromDistributed`                   | Ray remote -> SGLangEngine -> HTTP server | NCCL broadcast                          |
| colocate   | `UpdateWeightFromTensor`                        | Ray remote -> SGLangEngine -> HTTP server | CUDA IPC / Gloo gather_object / Ray IPC |
| 混合         | `UpdateWeightFromTensor` + distributed fallback | 两者都有                                      | colocated 用 IPC，remote 用 NCCL           |


## 6. 同步/异步训练流程

前面分别看了 rollout 和 train，现在把两者放回完整系统。

### 6.1 同步系统：train.py

`train.py` 是同步训练入口。这里的“同步”指 driver 层的 rollout 和 train 串行：

```text
rollout 0
  -> train 0
  -> update weights
  -> rollout 1
  -> train 1
  -> update weights
  -> ...
```

初始化阶段：

```text
configure_logger()
  -> create_placement_groups(args)
  -> init_tracking(args)
  -> create_rollout_manager(args, pgs["rollout"])
  -> create_training_models(args, pgs, rollout_manager)
  -> optional rollout_manager.onload_weights()
  -> actor_model.update_weights()
  -> optional rollout_manager.onload_kv()
```

训练 loop：

```text
for rollout_id:
  optional eval before train

  rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

  optional rollout_manager.offload()

  if use_critic:
    value_refs = critic_model.async_train(...)
    actor_model.async_train(..., external_data=value_refs)
  else:
    actor_model.async_train(...)

  optional save
  optional train clear/offload

  optional rollout_manager.onload_weights()
  actor_model.update_weights()
  optional rollout_manager.onload_kv()

  optional eval
```

同步系统的优点是简单、on-policy 关系清晰：rollout `k` 使用训练前或上轮同步后的权重，train `k` 训练这批数据，训练后再更新 rollout engine。缺点也明显：如果训练和 rollout 使用不同 GPU 池，那么 rollout 时训练 GPU 等待，train 时 rollout GPU 等待。

参考源码：train.py 的核心 loop

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    if args.use_critic:
        value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
        if actor_trains_this_step:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
        else:
            ray.get(value_refs)
    else:
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
```

### 6.2 框架级异步：train_async.py

`train_async.py` 解决的是 driver 层 rollout/train overlap。它要求：

```python
assert not args.colocate
```

因为 rollout 和 train 要并行跑，不能共享同一批 GPU。

核心时间线是：

```text
先启动 rollout 0

loop rollout_id = 0:
  等 rollout 0 完成
  立刻启动 rollout 1
  train 0

loop rollout_id = 1:
  等 rollout 1 完成
  立刻启动 rollout 2
  train 1

...
```

抽象成流水线：

```text
rollout:  R0 | R1 | R2 | R3 |
train:       T0 | T1 | T2 | T3 |
```

相比同步系统，`train_async.py` 让 SGLang rollout GPU 和 Megatron train GPU 同时工作。代价是 rollout 使用的权重可能滞后一段时间，因此需要 `update_weights_interval` 控制什么时候同步新权重。

`train_async.py` 在更新权重前会做一个重要同步：

```python
rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
rollout_data_next_future = None
actor_model.update_weights()
```

也就是说，它会先等正在进行的 rollout 完成，再更新 SGLang 权重，避免 generation 中途被换权重。

参考源码：train_async.py 如何 overlap rollout 和 train

```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)

    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

    if (rollout_id + 1) % args.update_weights_interval == 0:
        rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
        rollout_data_next_future = None
        actor_model.update_weights()
```

### 6.3 全异步系统：rollout 控制本身变成异步

`train_async.py` 的异步主要发生在 driver 层：下一轮 rollout 和当前 train overlap。但每次 `RolloutManager.generate(rollout_id)` 仍然是“为这一轮收满 batch，然后返回”。

`examples/fully_async/fully_async_rollout.py` 展示了另一种思路：让 rollout worker 跨越 `generate_rollout` 调用生命周期持续运行。

它的结构是：

```text
global AsyncRolloutWorker
  -> 独立 worker_thread
  -> asyncio continuous_worker_loop()
  -> 不断从 data_buffer.get_samples(1) 取 group
  -> 不断提交 generate_and_rm_group()
  -> 完成后放入 output_queue

generate_rollout_fully_async()
  -> 只从 output_queue 收集已完成 group
  -> 收满 rollout_batch_size 后返回给训练
```

这和默认 rollout 的区别是：

```text
默认 rollout:
  generate_rollout_async 开始
    -> 提交任务
    -> 收满 batch
    -> abort pending
    -> 返回

fully async rollout:
  global worker 一直活着
    -> 持续生成
    -> 持续把结果放队列
  generate_rollout_fully_async 只是取货
```

所以“全异步”的本质不是 slime 训练侧重写了一套系统，而是 rollout 函数的控制策略变了。slime 的框架接口仍然是：

```python
def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples
```

只要把 `--rollout-function-path` 指向这个函数，`RolloutManager` 仍然会通过 `call_rollout_fn()` 调它；返回的样本仍然会走 `_convert_samples_to_train_data()` 和 `_split_train_data_by_dp()`；训练侧也不需要知道 rollout 内部是同步收集、异步预取，还是长期后台 worker。

参考源码：fully_async 中的后台 rollout worker

```python
class AsyncRolloutWorker:
    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self.worker_thread_func,
                daemon=True,
            )
            self.worker_thread.start()

    async def continuous_worker_loop(self):
        active_tasks = set()
        max_concurrent_tasks = self.args.rollout_batch_size

        while self.running:
            while len(active_tasks) < max_concurrent_tasks and self.running:
                samples = self.data_buffer.get_samples(1)
                for group in samples:
                    task = asyncio.create_task(
                        generate_and_rm_group(
                            self.args,
                            group,
                            sampling_params=self.state.sampling_params.copy(),
                            evaluation=False,
                        )
                    )
                    task.add_done_callback(...)
                    active_tasks.add(task)
                    break

            await asyncio.sleep(1)
```

全异步系统带来的收益是 rollout 更连续，长尾样本不容易让整个系统停住；代价是 policy lag 和 buffer 策略更复杂。此时 `Sample.weight_versions`、`rollout_log_probs`、`keep_old_actor`、`update_weights_interval`、partial rollout 的 `loss_mask` 都会变得更重要，因为训练数据可能来自不同权重版本或不同时间窗口。

### 6.4 三种模式对比

三种模式可以这样比较：


| 模式                  | rollout 控制     | train/rollout 是否 overlap | 权重新鲜度 | 复杂度 |
| ------------------- | -------------- | ------------------------ | ----- | --- |
| `train.py`          | 每轮现生成现训练       | 否                        | 最新    | 低   |
| `train_async.py`    | 下一轮 rollout 预取 | 是                        | 有可控滞后 | 中   |
| fully async rollout | 后台 worker 持续生成 | 是                        | 滞后更明显 | 高   |


更直观地看：

```text
同步:
  R0 -> T0 -> W0 -> R1 -> T1 -> W1

框架级异步:
  R0 -> R1 -> R2 -> R3
        T0 -> T1 -> T2
             W at interval boundary

全异步:
  rollout worker:  continuous RRRRRRRRRRRR
  train loop:          take batch -> T0 -> take batch -> T1
  weight update:       update at configured boundary
```

其中 `W` 表示把 Megatron actor 权重同步到 SGLang engines。

## 7. slime 的核心设计总结

slime 的工程设计可以概括成四句话：

1. 用 Ray placement group 先锁定 GPU 拓扑，再把训练 actor 和 rollout engine 精确放到对应 bundle。
2. 用 `RolloutManager` 作为 rollout 控制面，启动并管理 SGLang router/server，同时持有 DataSource 和 rollout 函数。
3. 用 `RayTrainGroup` 管理 Megatron ranks，让每个训练 actor 从 Ray object store 取自己的 DP partition 完成训练。
4. 用可替换 rollout 函数把 agentic 复杂性留在 rollout 侧，使训练侧只依赖统一的 `Sample -> RolloutBatch` 协议。

如果从 Agentic RL 系统角度看，slime 最值得学习的不是某一个 loss 公式，而是它的边界划分：

```text
框架负责:
  资源、服务、并发、权重同步、数据切分、训练执行

用户负责:
  任务数据、agent 交互逻辑、reward/verifier、特殊 sample mask、特殊过滤策略
```

这种边界让 slime 可以同时支持普通 RLVR、tool-use agent、partial rollout、动态采样、异步 rollout，以及更复杂的多模型 SGLang serving。系统复杂度主要被收束在两个协议上：

```text
rollout 函数协议:
  (args, rollout_id, data_source, evaluation) -> RolloutFnTrainOutput

样本协议:
  Sample(prompt, tokens, response, reward, loss_mask, metadata, ...)
```

理解了这两个协议，再沿着 `create_placement_groups -> create_rollout_manager -> create_training_models -> generate/train/update_weights` 这条主线看源码，slime 的复杂系统就不再是一团异步调用，而是一套清晰的 Ray actor 编排。