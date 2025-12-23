# 🕷️ S.P.I.D.E.R. SDK

**Speculative Planning with Iterative Deep Exploration & Refinement**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Dual%20AGPL%2FCommercial-blue.svg)](#license)

An AI-powered SDK for solving SWE-Bench coding tasks with state-of-the-art agentic capabilities.

---

## ✨ Features

- 🧠 **Agentic Tool Loop** - ReAct pattern: Think → Act → Observe
- 👥 **Multi-Agent System** - Specialized agents (Planner, Coder, Tester, Reviewer)
- 🔧 **8 Built-in Tools** - read_file, search_code, write_file, run_tests...
- 📊 **4 Solving Modes** - Simple → Agentic → Multi-Agent → Full
- 💰 **Cost-Efficient** - ~$0.001 per task via OpenRouter
- 🐳 **Docker Isolation** - Safe containerized execution

---

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/kbpr25/spider-sdk.git
cd spider-sdk
pip install -e .

# Configure (create .env file)
echo "OPENROUTER_API_KEY=your-key" > .env

# Use CLI
spider demo                              # Test installation
spider solve "Fix the null pointer bug"  # Solve a problem
```

### Python API

```python
from spider.core.agent import UltimateSolver, SolverMode

solver = UltimateSolver(mode=SolverMode.FULL)
success, patch, meta = solver.solve(task)
```

---

## 🎯 Solving Modes

| Mode | Strategy | Expected SWE-Bench |
|------|----------|-------------------|
| `simple` | Single LLM call | ~30% |
| `agentic` | ReAct loop | ~55% |
| `multi` | 4-agent team | ~70% |
| `full` | All combined | ~80% |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 Agentic Layer    │ ReAct Agent + 8 Tools               │
├─────────────────────────────────────────────────────────────┤
│  👥 Multi-Agent      │ Planner → Coder → Tester → Reviewer │
├─────────────────────────────────────────────────────────────┤
│  🔧 Iron Interface   │ Git • Docker • LLM Gateway          │
├─────────────────────────────────────────────────────────────┤
│  🧮 Core Algorithms  │ MCTS • LOUDS • CRDT • Vector Clocks │
└─────────────────────────────────────────────────────────────┘
```

---

## � License

S.P.I.D.E.R. SDK is **dual-licensed**:

### Open Source License (AGPL-3.0)
Free for open source projects under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

**Note**: AGPL requires that if you use this software in a network service, you must release your modifications under AGPL.

### Commercial License
For proprietary/commercial use without AGPL obligations:
- Enterprise integrations
- SaaS products
- Closed-source applications

**Contact**: [bpreddy2525@gmail.com] for commercial licensing inquiries.

---

## 🤝 Why Dual License?

- **Developers & Researchers**: Use freely under AGPL
- **Enterprises**: Commercial license for proprietary use
- **AI Companies**: Custom licensing for integration

---

**Made with 🕷️ by the S.P.I.D.E.R. Team**
