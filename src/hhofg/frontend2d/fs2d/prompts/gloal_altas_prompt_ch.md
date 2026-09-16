你是一名家用服务机器人操控与可操作性建模方面的专家。

[任务]
为功能场景图生成一个紧凑的“厨房全局可操作性图谱（先验）（Kitchen Global Operability Atlas (prior)）”。
该图谱列出可操控实体（O）以及它们典型的功能承载体（C）与交互单元（U）。
输出必须是严格 JSON 对象（无额外文字、无 Markdown）。

[定义]

- O（object）：在图谱中建模为一个节点的可操控实体。
  为减少遗漏，请按类别思考（每个类别后给出若干示例；示例并不穷尽）：
  • 大型家电：refrigerator，oven，microwave，dishwasher
  • 小型家电：kettle，coffee machine，rice cooker
  • 家具与储物：cabinet
  • 固定装置与管道：sink，faucet
  • 台面容器/工具：bottle，jar
  • 基础设施：ceiling light，switch panel，power outlet，window，electric outlet
  注意：仅当实体具有可执行交互界面时才纳入（见下方约束）。
- C（functional carrier）：物体内部用于组织交互的中层结构，通常承载一个或多个 U
  （例如 door，drawer，control panel）。
- U（interactive unit）：可被直接按压/旋转/拉/推/拨动的原子级执行器
  （例如 handle，knob，button，cap，lid，switch，stopper，lever）。

[本地边 vs. 远程边 —— 重要约束]
功能关系分为两类：

- 本地关系（E_loc）：由实体内部物理“part-of（部件-整体）”结构形成的内部控制链，
  通常为 O <- C <- U 或 O <- U（例如 oven <- control panel <- knob；cabinet <- drawer/door <- handle/knob；bottle <- cap；kettle/pot <- handle）。
- 远程关系（E_rem）：物理上彼此独立的实体之间的功能关联（场景中相互独立的实体），
  例如 sink <- faucet，ceiling light <- switch。

在该图谱中：

- 本地关系必须且只能通过每个物体条目内部的 O/C/U 层级来表达
  （functional_carriers 与 direct_interactive_units）。不要输出单独的本地边列表。
- 远程关系必须且只能通过一个单独的候选列表来表达（见 "remote_relation_candidates"）。
  不要把远程关系混入本地层级中。

=========================
通用约束（高优先级）
==========

1. 仅包含具有可执行交互界面的 O 条目

- 仅当一个 O 至少具有一个可执行交互界面时才纳入：
  （functional_carriers 非空）或（direct_interactive_units 非空）。
- 排除仅能“抓取并移动”且没有清晰执行器的实体。

2. 最小化（避免过度生成）

- 对每个 O：优先 0–2 个 carriers。
- 对每个 carrier：优先 1–2 个 interactive units。
- 对 direct_interactive_units：优先 1–3 个 units。
- 如果不确定某项是否常见，就省略它；不要猜测长尾细节。

3. 规范粒度（避免长尾部件）

- 优先使用高频、通用、可泛化的名称。
- 避免罕见/过度具体的部件（例如 “freezer door”、“water reservoir lid”、“ice dispenser”、“filter latch” 等）。
- 不要发明过于细粒度的承载体名称：如果存在更标准/更简单的承载体，就使用更简单的那个。

4. 命名（稳定 + 紧凑）

- object / carrier：使用小写英文+空格（不使用连字符）。
- interactive units：优先单个、通用 token（例如 handle、knob、cap、button、switch、lever、stopper），避免冗长或高度具体的短语。
- 上述示例仅用于说明，不是固定词表；避免细粒度变体，例如
  “temperature-knob”、“start-button”、“mode-selector”。

5. U 归属规则（硬性规则）

- 每个 U 必须分配给它直接作用/直接控制的最小结构。
- 只有当同时满足以下两个条件时，U 才可以放在某 carrier 的 interactive_units 下：
  (a) 该 U 在物理上属于该 carrier（安装在该 carrier 上或是其组成部分），并且
  (b) 执行该 U 会直接控制该 carrier（开/关、锁/解锁、开始/停止、调节等），
  即该 U 的主要效果目标是该 carrier。
- 否则，如果该 U 用于操控整个物体（抓握/搬运/倾倒/移动），或它不特定于某个单一 carrier，
  那么它必须放入该物体的 direct_interactive_units。
- U 必须是原子级执行器；不要用结构/装配术语作为 U（例如 door、drawer、faucet、control panel）。

=========================
远程关系候选（仅 E_rem；列出成对对象，不要关系标签）
=============================

添加一个顶层字段 "remote_relation_candidates"，仅列出远程功能关系候选（E_rem）。

硬性规则：

- 只包含在物理上彼此独立的实体之间、可信的远程功能候选关系：
  跨实体依赖，例如控制/触发/供电/使能。
- 关系示例（仅用于说明）：
  - ceiling light controlled_by switch panel
  - appliance powered_by power outlet/electric outlet

- 不要在这里包含本地（part-of）关系。本地关系必须只能通过 O/C/U 层级来表达。
- 不要包含纯空间/结构关联，例如支撑、放置、邻接、包含、布局或共址。
  禁止示例（不要包含）：
  - cabinet <-> countertop
  - refrigerator/oven/microwave <-> cabinet
  - dishwasher <-> sink
  - switch panel <-> power outlet

- 保持紧凑。
- 只使用出现在 "objects" 列表中的对象名称，且字符串必须完全一致。

=========================
输出 JSON schema（严格）
==================

{
"scene_type": "kitchen",
"atlas_version": "v7",
"objects": [
{
"object": "<object name>",
"roles": ["O"],
"functional_carriers": [
{"carrier": "<carrier name>", "interactive_units": ["<unit1>", "<unit2>"]}
],
"direct_interactive_units": ["<unit1>", "<unit2>"]
}
],
"remote_relation_candidates": [
{"from_object": "<object>", "to_object": "<object>"}
]
}

remote_relation_candidates 的方向语义（必须遵守）：

- 使用与本地关系相同的约定：目标在左侧。
- 在 JSON 中这意味着：to_object <- from_object。
- from_object 必须是控制器/源端（例如 switch panel、power outlet/electric outlet、faucet/valve）。
- to_object 必须是被控制/消耗端目标（例如 ceiling light、refrigerator/microwave、sink）。

示例（按 to <- from 书写）：

- ceiling light <- switch panel
- refrigerator <- power outlet
- sink <- faucet

现在只输出严格 JSON。不要输出任何额外文字。不要使用 Markdown。
