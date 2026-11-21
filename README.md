# rubik-davi

No labels, no search tables: a small neural net learns a **cost-to-go heuristic** and a
**move policy** for the Rubik's cube, and beam search turns them into a solver.
2×2 and 3×3, with a live training dashboard and a 3D interactive cube you can scramble
and solve by hand.

从零学还原魔方：DAVI 式价值迭代 + 自监督策略 + beam 搜索。**训练 loss 里没有任何最短解标签**，
`oracle.py`（2×2 全空间精确距离）只当考卷，并在模块边界上与训练器隔离。

![2×2 训练看板](docs/img/dashboard-2x2.png)
![3×3 3D 看板](docs/img/dashboard-3x3.png)

> **先说定位**：这不是生产级求解器。真要快，用两阶段（Kociemba）——`min2phase` /
> `rob-twophase` 毫秒级给出约 20 步解。这个项目回答的是另一个问题：**不用任何标签，
> 网络能学到什么程度、代价是什么、边界在哪里。** 边界是测出来的，不是猜的。

---

## 1. 可用权重（直接能跑）

| 文件 | 魔方 | 体积 | 参数 | 训练迭代 | 实测（见 §3） |
|---|---|---|---|---|---|
| [`weights/rubik3x3_417k.pt`](weights/rubik3x3_417k.pt) | 3×3 | 2.52 MB | 66 万 | 417k（300k 从零 + 117k 续训） | 8/12/16 步打乱 **100%**；20 步 **85%**（beam 1024） |
| [`weights/rubik2x2_20k.pt`](weights/rubik2x2_20k.pt) | 2×2 | 2.19 MB | 57 万 | 20k | 随机状态 **99.7%** 解出，**82.6%** 恰好最短 |

SHA256 与训练来源见 [`weights/manifest.json`](weights/manifest.json)：

```
6115f70dbd35…  rubik3x3_417k.pt
1809fa18823b…  rubik2x2_20k.pt
```

权重就是普通 PyTorch `state_dict`，不依赖本项目也能加载：

```python
ck = torch.load("weights/rubik3x3_417k.pt", map_location="cpu", weights_only=False)
ck["meta"]          # {'it': 417000, 'hidden': 512, 'device': 'mps'}
ck["state"].keys()  # trunk.* / vhead.* / phead.*
```

---

## 2. 使用方法

### 2.1 一条命令解题（不需要训练）

```bash
pip install -r requirements.txt

python3 solve.py --cube 3 --scramble 20 --width 1024      # 随机打乱 20 步并求解
python3 solve.py --cube 3 --state <54位贴纸数字> --width 1024   # 指定局面（面序 U R F D L B）
python3 solve.py --cube 2 --scramble 14 --width 128
```

输出 JSON：`scramble / solved / length / sec / moves`，`moves` 可直接执行。
实测：3×3 的 20 步打乱 → 26 步解，**0.58 秒**（CPU 单核）。

### 2.2 3D 交互看板（推荐）

```bash
./start3.sh        # → http://127.0.0.1:8851
```

- 拖动旋转、滚轮缩放；「随机打乱」8/12/16/20 步任选，逐帧播放打乱过程
- 下方 12 个按钮自己拧（`U U' R R' F F' D D' L L' B B'`），「撤销一步」可回退
- 右侧实时显示：网络建议的动作、12 个候选动作各自的概率、走完之后的估值 V
- 「网络自动还原」：beam 搜索给整条解并逐步播放；宽度 128 / 512 / **1024**（默认）
- 权重按文件时间戳热加载，训练还在跑也能拿最新权重上手

### 2.3 训练 + 2D 看板

```bash
./start.sh                     # 2×2：训练 20k（M4 约 7 分钟）+ 看板 → :8849
ITERS=300000 ./start3.sh       # 3×3：从头训练 300k（M4 约 90 分钟）
TRAIN=0 ./start3.sh            # 只看板，不动训练
PORT=8850 RUN=runs/mine ./start3.sh
```

续训：`python3 train3.py --resume runs3/x/model.pt --lr 1e-4 --iter-offset 300000 ...`

---

## 3. 实测数据

所有数字都来自 §1 的两个权重文件；3×3 的每条解都会被**逐帧重放验证**（`is_solved` 精确判定）。

### 3.1 2×2（考卷：全空间 3 674 160 状态 BFS，God's number 14）

n=300 个**均匀随机**状态（平均最优 10.63 步）：

| 求解器 | 解出率 | 平均步数 | 恰好最短 |
|---|---|---|---|
| 纯策略贪心（宽度 1） | 14.7% | 11.59 | 52% |
| beam-policy w128 | 96.7% | 11.24 | 71.4% |
| **beam-policy w192** | **99.7%** | 10.98 | **82.6%** |
| beam-value w64 | 84.7% | — | **96.5%**（已解出的样本中） |

价值头对真值距离 **MAE 1.76 步、相关性 0.13**——弱距离估计，只有配搜索才有效。
同一个网络，宽度 1 → 192，解出率 14.7% → 99.7%：**网络没变，变的是搜索预算。**

### 3.2 3×3（无真值可枚举，只报解出率与步数）

权重 `rubik3x3_417k.pt`，每档 20 个随机局面（`seed=31415`），比例误差约 ±10 个百分点：

| 打乱步数 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|
| beam 128 | 100% | 80% | 45% | 40% |
| beam 512 | 100% | 100% | 100% | 80% |
| **beam 1024** | **100%** | **100%** | **100%** | **85%** |
| 平均解长（w1024） | 7.9 | 11.5 | 24.2 | 25.9 |

策略头自监督 top-1 = **55.5%**（12 选 1，随机基线 8.3%）。

**主要瓶颈是搜索宽度，不是网络容量。** 同一批 20 步局面：宽度 128→512→1024 是 40%→80%→85%；
加到 2048 仍全解，平均 23.6 步（1.0 s/局面）。诊断依据：把「正确动作」在策略输出里的排名
拉出来看，**top-12 覆盖率恒为 100%**、平均排名约 3，而且 beam 失败与成功局面的排名几乎一样
（3.9 vs 3.0）——**信息在，只是被窄 beam 浪费了。**

### 3.3 训练量 vs 搜索宽度（消融）

| 变化 | 20 步打乱解出率 |
|---|---|
| 300k 权重，beam 128 | 0–30%（两次测量，n=6 / n=20） |
| 300k 权重，beam 1024 | 62%（n=8） |
| +117k 续训，beam 128 | 40%（n=20） |
| **+117k 续训，beam 1024** | **85%（n=20）** |

多训 117k 单独贡献约 +10~15 pp；把 beam 加宽单独贡献约 +40 pp；两者叠加到 85%。
（口径不同已标 n，不做跨口径外推。）

---

## 4. 算法策略

### 4.1 状态与动作

- **3×3**：8 角块 + 12 棱块，状态 = 位置置换 + 朝向（角 3 值、棱 2 值）。输入 = 每个位置
  「哪个块 + 什么朝向」的 one-hot，共 **256 维**。
- **2×2**：8 角块，**88 维**。
- **动作**：12 个 90° 转（`U U' R R' F F' D D' L L' B B'`）；2×2 用 6 个（D/L/B 在 2×2 上等价于整体旋转）。
- 转动表**由几何推导**：位置用 ±1 坐标、朝向用「绕角块体对角线的 120° 旋转」定义，
  导入时断言每个动作在每个位置上是纯循环移位。

> 编码对照：贴纸 one-hot 与块级编码都能过拟合，但**泛化差异巨大**——贴纸编码的 MLP 在 12k 步
> 监督训练后仍是 R²≈0；换成块级「谁在哪」编码才学得动。

### 4.2 价值头：DAVI 式自举（无标签）

```
J(s) = min_a [ 1 + J(s') ] ,   J(目标态) = 0
```

- 数据：从**还原态**出发随机游走（`k` 由课程 4→20 增长），每个样本独立采样；
- 目标：用**目标网络**估每个后继的 `V`，取 `min` 后 +1，Huber 损失；
- 锚定：每批强制注入 16~32 个还原态样本、目标 0，否则网络对目标态没有定义；
- **为什么是 `min`**：这是 cost-to-go。写成 `max_a[1+V(s')]`（照抄 DAVI 的「最大化奖励」却
  没把奖励放在目标处）会让 V 单调漂到上界并塌成常数——第一版就这么挂的，见 §6。

### 4.3 策略头：自监督「撤销最后一步」

从还原态随机游走 k 步，**最后一步已知**，于是标签免费：`policy(s) → 逆动作`。
这是 AlphaCube 那一路的自监督，稠密、无需搜索，几百步开始收敛（2×2 500 步 58%；
3×3 最终 55.5%）。

天花板也来自这里：同一状态可由很多游走出来，「上一步」**不是状态的函数**，
所以深层准确率饱和在 ~55%，单靠策略直走解不出深局面——必须配搜索。

### 4.4 beam 搜索（真正干活的求解器）

每步保留 W 条候选路径，两种打分：

- `policy`：`f = −Σ log p(a)`（主用）
- `value`：`f = g + V(s)`（DeepCubeA/BWAS 那一路）

按状态索引去重，某个候选的下一跳等于还原态就收工并**重放验证**。
宽度是唯一要调的超参，收益见 §3.2——这就是「网络当启发式、搜索负责收敛」的实际含义。

### 4.5 为什么不是裸 DQN

1. **单目标 + 稀疏奖励**：3×3 有 4.3×10¹⁹ 个状态只认一个目标态，±1 终局奖励等于让网络当预言家；
2. **信用分配跨度太长**：一条解 20 步，中途没有中间奖励；
3. **`max` 备份会发散**：见 §4.2 / §6。

改造后的形态是 **价值回归（自举）+ 策略自监督 + 搜索**，不是「策略网络端到端」。
另外，策略头**不决定搜索的可行性**——探针显示正确的动作几乎总在候选里，瓶颈在预算。

---

## 5. 正确性校验（数字为什么可信）

| 校验项 | 2×2 | 3×3 |
|---|---|---|
| 全空间 BFS 距离表 | 3 674 160 状态 / God's number 14 / 与公开 QTM 分布**逐项吻合** | 不可能（4.3×10¹⁹） |
| 块级状态 vs 独立实现 `pycuber`（40 条随机序列） | — | **40/40 完全一致** |
| 54 贴纸图 vs `pycuber`（60 条随机序列） | — | **60/60 一致** |
| 群阶恒等式 | — | `order(R U)=105`、`order(R U')=63` ✓ |
| 不变量（扭转%3 / 翻转%2 / 奇偶） | — | 300 条序列 0 违反 |
| 求解结果重放验证 | 每条解都验 | 每条解都验 |

```bash
pip install pycuber
python3 check3.py        # 贴纸级 60/60 + 群阶
python3 check3e.py       # 块级 40/40（绕开贴纸图直接比状态）
python3 derive_view.py   # 反推参考实现的配色与视角轴
```

---

## 6. 踩过并修掉的真 bug（值得抄走）

1. **Bellman 写成 `max`** 却给非目标步 +1 → V 单调漂到上界后塌成常数。cost-to-go 是 `min`。
2. **一批样本共用同一串打乱** → 2048 个样本其实是同一个状态。特征是评估均值落在整数上（10.0/5.0/2.0）。
   （同一个坑写测试脚本时又踩一次：循环里每次 `rng(777)` 重建，20 个样本 = 1 个局面。）
3. **层符号全传 +1** → U/F 单转看着完美，D/L/B 转错层。
4. **R、L 方向整体反了** —— 单面比对看不出来（90° 后该面本来纯色），只有群阶能抓。
5. **四个侧面贴纸视图水平轴全反** —— 穷举 8 组 (right, up) 轴、逐格比色才定位。
6. **beam 回溯用 `paths[0]`** 而不是 `paths[i % W]` → 终端报「已还原」画面却没还原；
   同类 bug 还出现过 `paths` 被更新两遍。**救命的办法只有一条：每条解都重放验证。**

---

## 7. 已知限制

- 3×3 的 20 步打乱 **85%**（beam 1024），不是 100%；再宽（2048）能到 100%，但慢一倍。
- 价值头是弱启发式（2×2 MAE 1.76 步），只能当排序信号，别当真实距离。
- 策略头受「撤销最后一步」的歧义限制，单走（无搜索）解不出深局面。
- beam 是**离线思考**：3×3 w1024 约 0.6 s/局面（CPU 单核），2×2 w32 约 0.9 ms。
- 只有 MPS / CPU 训练代码，没有多卡；复现 DeepCubeA 量级需要远超单机的算力。
- 看板是标准库 HTTP 服务、无鉴权，仅限本机。
- 3×3 续训在 108k 处被外部杀过一次（日志无异常栈，怀疑与当时在 MPS 上并行跑搜索有关），
  已从检查点续跑；`runs3/*/offset.json` 把「本 run 迭代号」换算成累计迭代号。

---

## 8. 目录

```
solve.py                               # 用权重直接解题（最小示例）
cube2.py  oracle.py  model.py  scorecard.py  train.py  serve.py  web/     # 2×2
cube3.py  model3.py  scorecard3.py  train3.py  serve3.py  web3/           # 3×3
check3.py check3e.py derive_view.py    # 与 pycuber 的交叉校验（开发用）
probe3.py                              # 诊断探针：策略 top-k 排名 / beam 宽度收益
final_eval.py  final_eval3.py          # 终评报告
weights/                               # §1 的两个权重 + manifest
docs/img  docs/results                 # 截图与评估 JSON
start.sh  start3.sh                    # 一键启动
```

## 9. 参考

- McAleer et al., *Solving the Rubik's Cube Without Human Knowledge* (2018) — DAVI
- Agostinelli et al., *Solving the Rubik's cube with deep reinforcement learning and search* (Nature MI 2019) — DeepCubeA / BWAS
- Takano, *Self-Supervision is All You Need for Solving Rubik's Cube* (2023) — AlphaCube
- Chervov et al., *A Machine Learning Approach That Beats Large Rubik's Cubes* (NeurIPS 2025) — CayleyPy

## License

MIT，见 `LICENSE`。
