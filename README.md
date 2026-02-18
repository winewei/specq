# specq — Spec Queue Orchestrator

[![CI](https://github.com/winewei/specq/actions/workflows/ci.yml/badge.svg)](https://github.com/winewei/specq/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 从规范化任务构建 DAG 队列，通过 Context Compiler 精炼上下文，调度 Claude Code Agent 逐个执行，由多模型投票验收。

**写好 spec，一键全自动。**

## 工作原理

```
① Scanner     扫描 changes/ 目录 → 发现 change
② Parser      解析 proposal.md + tasks.md → WorkItem
③ DAG         depends_on → 拓扑排序 → blocked / ready
④ Scheduler   从 ready 中选一个（解锁度 → 优先级 → 风险）
⑤ Compiler    LLM 精炼 proposal 为 task brief
⑥ Executor    Claude Code SDK → 实现 + lint + test → commit
⑦ Voters      httpx → N 个模型并行独立投票
⑧ Aggregate   risk_policy 汇总 → approved / rejected / needs_review
⑨ Complete    通过 → 解锁下游 → re-scan
              拒绝 → findings → Compiler 重编译 → 重试
```

## 快速开始

### 安装

```bash
pip install specq
# 或
uv tool install specq
```

### 初始化项目

```bash
cd your-project
specq init
```

这会创建：

```
.specq/
├── config.yaml           ← 入库：团队共享配置
└── local.config.yaml     ← gitignore：个人 API keys
changes/
└── 000-example/          ← 示例 change spec
    ├── proposal.md
    └── tasks.md
```

### 编写 Change Spec

```bash
mkdir changes/001-add-auth
```

**`changes/001-add-auth/proposal.md`**：

```markdown
---
depends_on: []
risk: medium
---
# Add JWT Authentication

为 API 添加 JWT 认证，支持 access token 和 refresh token。
```

**`changes/001-add-auth/tasks.md`**：

```markdown
## task-1: JWT Token Service
实现 JWT token 签发与验证。

## task-2: Auth Middleware
实现认证中间件，拦截需要认证的路由。

## task-3: Login & Refresh API
实现登录和刷新 token 的 API 端点。
```

### 预览执行计划

```bash
specq plan
```

```
  specq — Spec-driven Orchestrator
  Changes: 1 · DAG: valid

  #   ID                        Status     Deps            Risk     Verify
  --- ---                       ---        ---             ---      ---
  1   001-add-auth              ready      —               medium   majority

  Executor: claude-code (claude-sonnet-4-5)
  Compiler: claude-haiku-4-5
  Voters: gpt-4o, gemini-2.5-pro, claude-sonnet-4-5
```

### 执行

```bash
specq run              # 执行所有 ready 的 change
specq run 001-add-auth # 执行指定 change
specq run --all        # 全量扫描后执行
```

## CLI 命令

| 命令 | 说明 |
|---|---|
| `specq init` | 初始化项目 |
| `specq plan` | 预览执行计划（dry-run） |
| `specq run [id]` | 执行 change |
| `specq status [id]` | 状态概览 / 详情 |
| `specq deps` | DAG 依赖图 |
| `specq logs <id>` | 执行日志 |
| `specq votes <id>` | 投票验收报告 |
| `specq accept <id>` | 接受 needs_review |
| `specq reject <id>` | 拒绝 → failed |
| `specq retry <id>` | 重试 failed |
| `specq skip <id>` | 跳过，解锁下游 |
| `specq config` | 查看合并后配置 |
| `specq scan` | 手动触发扫描 |

## 三层配置

优先级从高到低：

1. **环境变量**（`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`）
2. **`.specq/local.config.yaml`**（gitignore，个人偏好 + API keys）
3. **`.specq/config.yaml`**（入库，团队共享）

proposal.md 的 YAML frontmatter 可覆盖单个 change 的配置。

### 配置示例

```yaml
# .specq/config.yaml
changes_dir: changes
base_branch: main

compiler:
  provider: anthropic
  model: claude-haiku-4-5

executor:
  type: claude_code
  model: claude-sonnet-4-5
  max_turns: 50

verification:
  voters:
    - provider: openai
      model: gpt-4o
    - provider: google
      model: gemini-2.5-pro
    - provider: anthropic
      model: claude-sonnet-4-5
  checks:
    - spec_compliance
    - regression_risk
    - architecture

risk_policy:
  low: skip
  medium:
    strategy: majority
  high:
    strategy: unanimous

budgets:
  max_retries: 3
  max_duration_sec: 600
  max_turns: 50
```

## 状态机

```
pending → blocked → ready → compiling → running → verifying
                                                      │
                                        ┌─────────────┤
                                   approved       rejected
                                      │          retry < max?
                                 ┌────┴────┐        │
                                auto   high risk    └→ ready (重试)
                                 │        │
                            accepted  needs_review ──→ accepted (人工确认)
```

## 风险策略

| 风险等级 | 默认验收策略 | 通过条件 |
|---|---|---|
| low | skip | 不验收，自动通过 |
| medium | majority | 过半 voter pass |
| high | unanimous | 全票 pass + 人工确认 |

## 零侵入设计

- 项目内只有 `.specq/config.yaml` 一个入库文件
- `rm -rf .specq/` → 项目零残留
- 无全局目录，所有状态都在项目 `.specq/` 内

## 开发

```bash
git clone https://github.com/winewei/specq.git
cd specq
pip install -e ".[dev]"
pytest tests/ -v
```

## License

[MIT](LICENSE)
