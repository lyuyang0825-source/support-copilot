# support-copilot

**可溯源、会认怂、会交接的客服副驾驶。回答前先检索知识库，回答时逐条引用，拿不准就转人工，转人工时把已收集信息打包好。**

[![ci](https://github.com/lyuyang0825-source/support-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/lyuyang0825-source/support-copilot/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> 这是四个行业独立仓库之一 · [总览与另外三个产品](https://github.com/lyuyang0825-source/ai-industry-labs)

## 为什么做这个方向

成本已经不是问题，信任才是：AI 客服年费几万元就能顶几十个真人坐席，但 2025 年上海市消保委受理的 AI 客服相关投诉达 1087 件，集中在答非所问、人工接入难、流程死循环；黑猫投诉里“转人工”相关投诉超过 3 万条。

“降本”经常变成“降责”：有在线教育公司上线 AI 客服后，日均对话从 2000 涨到 8000，但满意度下降 15%，转人工率从 20% 飙到 45%，客诉量反而增加 30%。

知识库本身是糊涂账：产品手册、退换货政策、促销规则散落在不同文档；Gartner 调研显示约 60% 的企业客服机器人实际能解决的问题不到一半，生成式 AI 对复杂售后场景的理解准确率不足 60%。

监管已经给出方向：2026 年 9 月 1 日实施的《顾客联络服务 人工与智能客户服务协同要求》明确，智能客服处理标准化问题，复杂/涉安全问题交人工；切换要“便捷入口 + 信息同步、不重复询问”；涉及价格、退款、赔偿、合同变更，AI 可以解释规则、收集材料、生成工单，但最终确认应由人工完成。

这个仓库就是按这条标准做的开源参考实现。

## 它做什么

| 能力 | 实现 |
| --- | --- |
| 知识治理 | Markdown + 元信息（id / 适用口径 / 生效区间 / owner / 风险级别），启动时校验、检测同题冲突 |
| 检索 | 纯标准库 BM25，中文二元组 + 电商同义词扩展 |
| 回答 | 只引用命中的知识原文，回复必带“依据：kb-xxx” |
| 置信度闸门 | 高分直接答；中分“答 + 需人工确认”；低分拒答并转人工 |
| 越权保护 | 退款金额、赔偿、价格、合同、账号安全等诉求强制人工确认；承诺词扫描 |
| 模型润色（可选） | OpenAI 兼容适配层，默认关闭；模型只改措辞，数字/承诺/跑题三道校验不过就回退规则答案 |
| 数字溯源 | 回复里的“3 个工作日”“7 天”等必须能在引用原文找到，否则降级人工 |
| 交接 | 工单包含意图、置信度、已核对知识、已收集字段、缺失字段、交接摘要、`do_not_reask` |
| 质检 | 对已有会话打分：引用覆盖率、越权承诺、该转未转、重复询问、下一步 |

### 决策矩阵

| 条件 | 动作 | 说明 |
| --- | --- | --- |
| 置信度 ≥ 0.70 且非高风险意图 | `answer` | 直接答，带引用 |
| 置信度 0.52–0.70 | `answer_with_human` | 给规则解释，同时建工单 |
| 置信度 < 0.52 或 other 意图且分数 < 8 | `refuse_and_escalate` | 不猜，直接转人工 |
| 退款/赔偿/价格/合同/账号/客诉/法务 | `answer_with_human` | 机器人只解释规则与收集材料，最终人工确认 |
| 问候语 / 过短输入 | `clarify` | 引导补充订单号与问题描述 |

## 开源模型怎么接（可选）

默认是纯规则链路，不下载权重、不需要 GPU。要接开源模型时，走任何 OpenAI Chat Completions 兼容服务即可：

```bash
export LLM_BASE_URL="https://api.siliconflow.cn/v1"   # 或本地 http://localhost:11434/v1
export LLM_API_KEY="sk-..."                           # 本地 Ollama 可留空
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"

python3 -m support_copilot.cli -q "退款多久到账" --llm
python3 scripts/run_evals.py --llm                    # 同一套 200 条用例做 A/B
```

没有 GPU、也没有 API key 时，可以先跑协议桩验证链路：

```bash
python3 scripts/mock_llm_server.py &                  # 只回显规则答案，不是模型
export LLM_BASE_URL="http://127.0.0.1:8123/v1"
python3 -m support_copilot.cli -q "退款多久到账" --llm
```

模型不决定动作、不选知识、不生成引用、不写工单，只改写“已命中的知识答案”。输出要过三道校验：承诺词、数字溯源、与原文的词元重合度；任何一道不过就回退原文，并在 `notes` 里写明原因。用户问题在发给模型前会先做 PII 脱敏。

这个仓库**不发布训练权重**，只发布适配层、评测集和模型卡：[docs/model-card.md](docs/model-card.md)。没有脱敏数据时，不训“演示用”权重。

## 知识库格式

```markdown
---
id: kb-refund-001
question: 实物商品支持几天无理由退货
answer: 自签收之日起 7 天内，商品保持完好、配件齐全可申请无理由退货。定制类、生鲜、已拆封的个人护理与贴身用品除外；退款在仓库验收后 3–5 个工作日原路退回。
scope: 实物商品（不含定制、生鲜、已拆封个护）
effective_from: 2025-01-01
effective_to: 2026-12-31
owner: 售后组
risk: high
tags: 退货,退款,售后
---
```

规则：

- 没有 `id` 的知识不允许进入回答；
- `answer` 必须是可独立引用的最小事实，不写“详见页面”；
- 高风险知识必须写明例外/边界，否则 `--lint` 会提示；
- 同题、生效区间重叠但答案不一致 → 判为冲突，必须先治理再上线。

## 快速开始

```bash
git clone https://github.com/lyuyang0825-source/support-copilot.git
cd support-copilot

python3 -m support_copilot.cli --lint                 # 知识库体检
python3 -m support_copilot.cli --demo                 # 6 条演示对话
python3 -m support_copilot.cli -q "退款多久到账"      # 单条回答
python3 scripts/run_evals.py --tickets                # 200 条决策评测 + 会话质检
python3 scripts/run_evals.py --suite core             # 只看最初的 10 条边界用例
python3 -m unittest discover -s tests                 # 30 tests
```

## demo：6 条对话

| 用户 | 决策 | 置信度 | 引用 |
| --- | --- | --- | --- |
| 退款多久到账 | `answer` | 0.88 | kb-refund-001、kb-coupon-001 |
| 我要退款 | `answer_with_human` | 0.88 | kb-refund-001、kb-coupon-001 |
| 价保几天？帮我补差价 | `answer_with_human` | 0.94 | kb-price-001 |
| 账号被冻结了怎么解封 | `answer_with_human` | 0.83 | kb-account-001 |
| 发票抬头能改吗 | `answer` | 0.85 | kb-invoice-001 |
| 你们支持火星配送吗 | `refuse_and_escalate` | 0.00 | — |

火星配送这条是故意的：虽然“配送”能命中物流规则，但目的地不在服务范围，直接拒答转人工，不拿物流时效硬套。

## 评测：200 条决策 + 会话质检

`scripts/run_evals.py` 默认跑 200 条开发集，校验“动作 + 期望知识是否命中”，并按 12 类汇总：

```
通过 199/200  (99.5%)

分类：
  价保-信息       20/20  (100%)
  价保-诉求       10/10  (100%)
  优惠券         15/15  (100%)
  保修          20/20  (100%)
  发票          20/20  (100%)
  客服时间        15/15  (100%)
  投诉法务         5/5   (100%)
  物流          19/20  (95%)
  知识库外        15/15  (100%)
  账号安全        20/20  (100%)
  退款-信息       25/25  (100%)
  退款-诉求       15/15  (100%)
```

唯一未过的是“到货时间怎么算”：检索分 5.56，落在 0.52–0.70 的“答 + 人工确认”区间。它没有答错，只是被保守降级；这类边界比“硬答一个错答案”更可接受，因此保留为已知降级，而不是改期望值粉饰通过率。

配置模型后，`python3 scripts/run_evals.py --llm` 会跑同一套用例，并额外输出“润色生效 / 回退条数”，用于判断模型有没有把答案改坏。用协议桩跑过一次：180/200 条走润色、20 条拒答或无需润色，0 条校验失败——这只能证明管道接得通，不能证明真实模型效果。

会话质检对三份会话打分：

| 会话 | 分数 | 问题 |
| --- | --- | --- |
| `ticket_01_good.json` | 100 | 无 |
| `ticket_02_bad.json` | 10 | 零引用、越权承诺（一定/保证）、该转未转、重复询问订单号、结尾无下一步 |
| `ticket_03_safety.json` | 30 | 用户已描述漏电触电，机器人仍在推销“质量绝对没问题”并只补优惠券，未升级安全/投诉流程 |

`ticket_03` 是新增的**安全边界样例**：质检不是只看语气，而是抓“该升没升”的风险。

## 设计约束（有意为之）

- **引用优先于流畅**：没有引用就不直接答；
- **模型只改措辞**：动作、知识、引用、工单全部由规则层决定，模型不越权；
- **数字必须溯源**：回复中的天数/金额要能在知识原文找到，防止模型改数；
- **只做标准化问题**：复杂售后、安全、资金争议交人；
- **交接不重复问**：`do_not_reask` 明确记录已提供信息；
- **纯标准库**：可离线部署，检索可替换为向量方案但不改变引用约束；
- **知识先治理再上线**：冲突、过期、高风险无边界，`--lint` 直接拦。

## 当前限制

- BM25 对口语化长句仍会漏召回；同义词表需要按业务持续补。
- 200 条是合成开发集，只能防回归，不能证明真实线上分布的效果。
- 模型适配层默认关闭；仓库没有训练权重，也没有跑过真实 7B 的 A/B，只提供接入与校验框架。
- 未接入真实工单系统、CRM、订单中心；工单是本地 JSON。
- 质检基于规则与关键词，不能替代人工抽检全部会话。
- 多轮上下文只保留结构化槽位，不做完整对话理解。

## 路线图

- [x] 200 条决策评测 + OpenAI 兼容模型适配层
- [ ] 接一个开源 7B，跑 base / tuned / 规则三组 A/B
- [ ] 向量检索 + BM25 混合，保留引用 ID
- [ ] 对接工单系统（飞书/企业微信/自研 CRM）
- [ ] 坐席实时辅助：人工会话中提示条款依据与下一句
- [ ] 知识缺口挖掘：从拒答与转人工反推缺失知识
- [ ] 质检抽样与趋势看板

## 目录

```
support-copilot/
├── support_copilot/
│   ├── kb.py       # 知识解析、有效期、冲突、lint
│   ├── bm25.py     # 中文二元组 + 同义词
│   ├── policy.py   # 意图、阈值、承诺词、PII、数字溯源
│   ├── llm.py      # OpenAI 兼容适配层 + 三道输出校验
│   ├── agent.py    # 回答编排 + 意图对齐 + 工单 + 交接摘要
│   ├── qa.py       # 会话质检
│   └── cli.py
├── prompts/
│   └── support_reply.md   # 润色器系统提示词
├── data/
│   ├── kb/         # 8 条知识
│   ├── tickets/    # 3 份质检样例（好/坏/安全边界）
│   └── evals/      # 10 条核心 + 200 条开发集
├── scripts/
│   ├── run_evals.py
│   └── mock_llm_server.py   # 无 GPU/Key 时的协议桩
├── tests/          # 30 个单测
├── demo_out/       # 演示输出
├── docs/
│   ├── industry.md
│   ├── architecture.md
│   └── model-card.md
└── .github/workflows/ci.yml
```

更多：[行业调研](docs/industry.md) · [架构与设计取舍](docs/architecture.md) · [模型接入卡](docs/model-card.md)
