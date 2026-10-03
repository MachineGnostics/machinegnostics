# Machine Gnostics - Agents Index

**Location:** `.github/agents/`

This directory contains detailed specifications for each of the 8 specialized AI agents in the Machine Gnostics project.

---

## 🤖 All Agents

### Domain Experts

1. **[magnet-expert.md](./magnet-expert.md)** 🧠
   - Expert in Machine Gnostic Neural Network (MAGNET) framework
   - Implements gnostic layers, activations, and loss functions
   - Focus: `src/machinegnostics/magnet/`
   - Skills: fact-grounded-coding, profiling, type-annotations

2. **[magcal-expert.md](./magcal-expert.md)** 📊
   - Expert in calibration algorithms and electromagnetic calculations
   - Handles data conversion and computational methods
   - Focus: `src/machinegnostics/magcal/`
   - Skills: fact-grounded-coding, type-inference

3. **[metrics-agent.md](./metrics-agent.md)** 📈
   - Expert in diagnostic metrics creation and validation
   - Ensures metrics standardization and correctness
   - Focus: `src/machinegnostics/metrics/`
   - Skills: fact-grounded-coding, type-annotations

4. **[ml-models-specialist.md](./ml-models-specialist.md)** 🤖
   - Expert in ML model development and training pipelines
   - Handles model evaluation and hyperparameter tuning
   - Focus: `src/machinegnostics/ml_models/`
   - Skills: fact-grounded-coding, profiling

5. **[ml-integration-specialist.md](./ml-integration-specialist.md)** ☁️
   - Expert in MLflow, cloud integrations, and deployment
   - Manages experiment tracking and model serving
   - Focus: `src/machinegnostics/integrations/`
   - Skills: type-annotations

### Support Specialists

6. **[manager-agent.md](./manager-agent.md)** 🎯
   - Project coordination and code quality assurance
   - Ensures standards compliance and architecture consistency
   - Focus: All modules in `src/machinegnostics/`
   - Skills: refactoring, type-annotations

7. **[unit-tester-agent.md](./unit-tester-agent.md)** 🧪
   - Expert in test strategy and quality assurance
   - Writes comprehensive tests and optimizes test execution
   - Focus: `tests/`
   - Skills: profiling, type-annotations

8. **[documentation-agent.md](./documentation-agent.md)** 📚
   - Expert in technical documentation and user guides
   - Creates API docs, tutorials, and conceptual documentation
   - Focus: `docs/`
   - Skills: pylance-docs, type-annotations

---

## Quick Selection Guide

**Choose your agent by task:**

| Task | Agent |
|------|-------|
| Add MAGNET layer or activation | Magnet Expert 🧠 |
| Improve calibration algorithm | Magcal Expert 📊 |
| Create diagnostic metric | Metrics Agent 📈 |
| Build ML model | ML Models Specialist 🤖 |
| Set up MLflow/cloud | ML Integration Specialist ☁️ |
| Code review/architecture | Manager Agent 🎯 |
| Write tests | Unit Tester Agent 🧪 |
| Create documentation | Documentation Agent 📚 |

---

## Agent Responsibilities Summary

| Agent | Primary Responsibility | Key Skills |
|-------|----------------------|-----------|
| Magnet Expert | MAGNET framework development | PyTorch, gnostic theory |
| Magcal Expert | Calibration algorithms | NumPy, numerical methods |
| Metrics Agent | Diagnostic metrics | Statistical metrics |
| ML Models | ML models & training | Scikit-learn, PyTorch |
| Integration | MLflow & cloud | AWS, Azure, GCP |
| Manager | Quality & coordination | Code review, architecture |
| Unit Tester | Testing & validation | pytest, test design |
| Documentation | User docs & tutorials | Markdown, MkDocs |

---

## How to Use These Files

1. **For New Task**: Find relevant agent file above
2. **Review Responsibilities**: Check what agent can do
3. **Read When to Use**: Understand decision criteria
4. **Check Primary Files**: Know what code agent works with
5. **Note Recommended Skills**: Use skill files for enhanced capabilities

---

## Reading Individual Agent Files

Each agent file contains:
- **Overview**: One-sentence description
- **Primary Responsibilities**: 8-12 key tasks
- **When to Use**: Decision guidance with examples
- **Primary Files/Directories**: Code modules agent focuses on
- **Key Technologies**: Tools and frameworks
- **Recommended Skills**: Which Copilot skills enhance this agent
- **Example Prompts**: Ready-to-use commands
- **Expertise Stack**: Skill breakdown

---

## Related Documentation

- **[../skills/](../skills/)** - Skill descriptions and usage
- **[../AGENT-ROLES.md](../AGENT-ROLES.md)** - Original comprehensive agent guide
- **[../AGENTS-SKILLS-ENHANCEMENT.md](../AGENTS-SKILLS-ENHANCEMENT.md)** - Skills integration roadmap
- **[../README.md](../README.md)** - Documentation index
