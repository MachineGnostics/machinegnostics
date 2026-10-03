# Machine Gnostics Project - Copilot Agent System

## 📚 Documentation Index

This directory contains all configuration and guidelines for the Machine Gnostics AI Agent System. Use this index to find the right document for your needs.

---

## 🚀 Getting Started

**New to Machine Gnostics agents?** Start here:

1. **[HOW-TO-USE-AGENTS.md](./HOW-TO-USE-AGENTS.md)** - How to actually use agents & skills ⭐ START HERE
   - Quick start (30 seconds)
   - Workflows for common tasks
   - How to talk to agents
   - Common use cases
   - Pro tips and learning path

2. **[agents/README.md](./agents/README.md)** - Select an agent
   - All 8 agents listed
   - Quick selection guide
   - Click to individual agent specifications

3. **[skills/README.md](./skills/README.md)** - Find a skill
   - All 6 skills listed
   - Decision tree by use case
   - Click to individual skill documentation

4. **[copilot-instructions.md](./copilot-instructions.md)** - Project overview
   - General philosophy
   - Python style guidelines
   - Agent guidelines
   - Available skills section

---

## 📖 Comprehensive Documentation

### How to Use Agents & Skills
- **[HOW-TO-USE-AGENTS.md](./HOW-TO-USE-AGENTS.md)** - Complete guide to using agents and skills ⭐ NEW
  - Quick start workflows (30 seconds to productive)
  - Detailed agent expertise breakdown
  - How to communicate with agents effectively
  - Common use cases and recommended workflows
  - Pro tips and best practices
  - Learning path for new users

### Core Configuration
- **[copilot-instructions.md](./copilot-instructions.md)** - Main project instructions and standards
  - General philosophy
  - Python style guidelines
  - Code organization standards
  - Agent guidelines and workflows

### Detailed Agent Documentation
- **[agents/](./agents/)** - Individual agent specifications folder
  - **[agents/README.md](./agents/README.md)** - Agent index and quick selection guide
  - **[magnet-expert.md](./agents/magnet-expert.md)** - MAGNET framework development
  - **[magcal-expert.md](./agents/magcal-expert.md)** - Calibration & electromagnetic
  - **[metrics-agent.md](./agents/metrics-agent.md)** - Diagnostic metrics
  - **[ml-models-specialist.md](./agents/ml-models-specialist.md)** - ML model development
  - **[ml-integration-specialist.md](./agents/ml-integration-specialist.md)** - MLflow & cloud
  - **[manager-agent.md](./agents/manager-agent.md)** - Coordination & quality
  - **[unit-tester-agent.md](./agents/unit-tester-agent.md)** - Testing & quality
  - **[documentation-agent.md](./agents/documentation-agent.md)** - Documentation & guides

### Code Standards & Quality
- **[STANDARDS.md](./STANDARDS.md)** - Code and documentation standards
  - File header requirements
  - Docstring standards (class, method, module)
  - Test file standards
  - Documentation structure
  - **Import standards and guidelines** ⭐ NEW

### Import Architecture & Guidelines
- **[IMPORT-ARCHITECTURE.md](./IMPORT-ARCHITECTURE.md)** - Import system design overview ⭐ NEW
  - Why the import architecture matters
  - Module organization by subsystem
  - Design decisions and constraints
  - Testing import structure
  - Migration guidance

- **[IMPORT-GUIDE.md](./IMPORT-GUIDE.md)** - Detailed implementation guide ⭐ NEW
  - Recommended import patterns
  - __init__.py guidelines for each level
  - Import usage examples
  - Public API design rules
  - Developer best practices

### Concept & Framework Documentation
- **[MAGNET-CORRECTION.md](./MAGNET-CORRECTION.md)** - MAGNET framework explanation
  - What MAGNET actually is
  - Mathematical foundations
  - Component architecture
  - Development phases

### Agent Enhancement & Skills
- **[skills/](./skills/)** - Individual skill specifications folder
  - **[skills/README.md](./skills/README.md)** - Skills index and decision tree
  - **[python-fact-grounded-coding.md](./skills/python-fact-grounded-coding.md)** - Math & theory validation
  - **[python-add-type-annotations.md](./skills/python-add-type-annotations.md)** - Type safety
  - **[python-type-inference.md](./skills/python-type-inference.md)** - Complex type understanding
  - **[pylance-docs.md](./skills/pylance-docs.md)** - Documentation auditing
  - **[pylance-python-profiling.md](./skills/pylance-python-profiling.md)** - Performance optimization
  - **[pylance-refactoring.md](./skills/pylance-refactoring.md)** - Code standards & cleanup

---

## 🎯 The 8 Specialized Agents

### Domain Experts

| # | Agent | Focus Area | When to Use |
|---|-------|-----------|------------|
| 1️⃣ | **Magcal Expert** | `src/machinegnostics/magcal/` | Calibration, data conversion, EM calculations |
| 2️⃣ | **Magnet Expert** | `src/machinegnostics/magnet/` | PyTorch neural networks, magnetic analysis |
| 3️⃣ | **Metrics Agent** | `src/machinegnostics/metrics/` | Diagnostic metrics, validation |
| 4️⃣ | **ML Models Specialist** | `src/machinegnostics/ml_models/` | Classical ML, ensemble methods, training |
| 5️⃣ | **ML Integration Specialist** | `src/machinegnostics/integrations/` | MLflow, cloud integration, deployment |

### Support Specialists

| # | Agent | Focus Area | When to Use |
|---|-------|-----------|------------|
| 6️⃣ | **Unit Tester** | `tests/` | Writing comprehensive unit and integration tests |
| 7️⃣ | **Documentation** | `docs/` | Creating API docs, guides, tutorials |
| 8️⃣ | **Manager** | All modules | Coordination, quality assurance, architecture |

---

## 🔄 Quick Workflow

### Choose Your Path

**Start Simple: Single Module Work**
```
You → Specialist Agent → Code Complete
```

**Standard: Full Feature Development**
```
You → Manager Agent (coordinate)
   ↓
   Specialist Agent (implement)
   ↓
Manager Agent (review)
   ↓
Unit Tester (test)
   ↓
Documentation Agent (document)
   ✅ Complete
```

**Complex: Multi-Module Changes**
```
You → Manager Agent (plan & assign)
   ↓
Multiple Specialists (parallel development)
   ↓
Manager Agent (integrate & validate)
   ↓
Unit Tester (comprehensive testing)
   ↓
Documentation (update all docs)
   ✅ Complete
```

---

## 📋 Standard Request Templates

### For Specialist Agents
```
[Agent Name], I need to [task description]

Context: [Background information]
Files: [Which files/modules involved]
Requirements: [Specific requirements]
Expected output: [What success looks like]
```

### For Manager Agent
```
Manager Agent: I need to [describe goal]

This involves: [Which modules/systems]
Complexity: [Simple/Medium/Complex]
Dependencies: [Any blockers or prerequisites]

Which agent should lead? What's the best approach?
```

### For Test Agent
```
Unit Tester Agent: Write tests for [module/feature]

Focus: [Specific test scenarios]
Coverage target: [Percentage or specific areas]
Edge cases: [Specific edge cases to test]
```

### For Documentation Agent
```
Documentation Agent: Create documentation for [feature/module]

Include: [Specific sections needed]
Audience: [Who this is for - users/developers]
Format: [API reference/guide/tutorial/concept]
```

---

## ✅ Success Checklist

A completed task should have:

- ✅ Code follows all standards from `copilot-instructions.md`
- ✅ Files have proper headers (module docstring, author, date)
- ✅ All public APIs have comprehensive docstrings
- ✅ Type hints present and consistent
- ✅ Unit tests with meaningful coverage
- ✅ Documentation updated/created
- ✅ No unused imports or dead code
- ✅ Related code is kept local
- ✅ Git diff is clean

---

## 🎓 Learning Path

### Level 1: Beginner
1. Read [AGENTS-SETUP.md](./AGENTS-SETUP.md)
2. Review [QUICK-AGENT-GUIDE.md](./QUICK-AGENT-GUIDE.md)
3. Try a simple task with a specialist agent

### Level 2: Intermediate
1. Read [AGENT-ROLES.md](./AGENT-ROLES.md) completely
2. Review [STANDARDS.md](./STANDARDS.md) for code quality
3. Try a medium task with multiple agents
4. Use Manager Agent to coordinate

### Level 3: Advanced
1. Master all documentation
2. Lead complex multi-module projects
3. Mentor others on agent usage
4. Suggest improvements to standards

---

## 📞 How to Find What You Need

**I want to...**

| Goal | Start Here |
|------|-----------|
| Learn about agents | AGENTS-SETUP.md |
| Get quick reference | QUICK-AGENT-GUIDE.md |
| Understand one agent | AGENT-ROLES.md → Find agent section |
| Check code standards | STANDARDS.md or copilot-instructions.md |
| Start a magcal task | QUICK-AGENT-GUIDE.md → Pick Magcal Expert |
| Deploy to production | AGENT-ROLES.md → ML Integration Specialist |
| Write tests | AGENT-ROLES.md → Unit Tester Agent |
| Create documentation | AGENT-ROLES.md → Documentation Agent |
| Review code quality | AGENT-ROLES.md → Manager Agent |
| Complex coordination | AGENT-ROLES.md → Manager Agent |

---

## 📁 Documentation Structure

```
.github/
├── README.md ←────────────── YOU ARE HERE
│
├── AGENTS-SETUP.md ─────── Getting started guide
├── QUICK-AGENT-GUIDE.md ─ Quick reference
├── AGENT-ROLES.md ──────── Detailed agent specs
├── STANDARDS.md ────────── Code/doc standards
├── copilot-instructions.md ─ Main project instructions
│
└── workflows/ ──────────── GitHub Actions workflows
```

---

## 🔗 Related Documentation

For project-specific information, also see:

- **README.md** (root) - Project overview and installation
- **docs/** - User documentation and tutorials
- **src/machinegnostics/** - Source code with inline documentation
- **tests/** - Test suite demonstrating usage
- **CITATION.cff** - Project citation information
- **pyproject.toml** - Project configuration

---

## 🚀 Starting Your First Task

1. **Identify what you need** to build or fix
2. **Check which agent** should handle it (see Quick Reference above)
3. **Read relevant documentation** (this README points to the right file)
4. **Request from the agent** using the template above
5. **Chain to next agent** if needed (Manager can coordinate)
6. **Verify against standards** before considering complete

---

## 💡 Pro Tips

- 🎯 **Be specific**: Provide clear requirements and context
- 🔗 **Chain agents properly**: One agent's output becomes next agent's input
- 📚 **Reference standards**: Link to STANDARDS.md or copilot-instructions.md
- ✅ **Use checklists**: Standards.md has many helpful checklists
- 🤝 **Coordinate early**: Manager Agent helps with complex multi-agent work
- 📝 **Document decisions**: Keep decisions in code/docs as you work

---

## ❓ FAQ

**Q: How do I know which agent to use?**
A: Check QUICK-AGENT-GUIDE.md's decision tree, or ask Manager Agent if unsure.

**Q: Can I use multiple agents for one task?**
A: Yes! Manager Agent will coordinate. Just ask "I need work on [modules], how should we coordinate?"

**Q: What if an agent's output doesn't match standards?**
A: Forward to Manager Agent for quality review. They enforce standards.

**Q: How do I know my work is complete?**
A: Check the Success Checklist above and run validation with Manager Agent.

**Q: Can agents work in parallel?**
A: Yes! Manager Agent can assign multiple specialists to work simultaneously.

---

## 📞 Support

For questions about:

- **Agent roles**: See AGENT-ROLES.md
- **Code standards**: See STANDARDS.md
- **Getting started**: See AGENTS-SETUP.md
- **Quick reference**: See QUICK-AGENT-GUIDE.md
- **Project philosophy**: See copilot-instructions.md

**When in doubt**: Ask the Manager Agent! They're here to help coordinate.

---

## 🎉 Ready to Get Started?

1. Pick your task
2. Choose your agent (or ask Manager Agent)
3. Make your request
4. Build amazing diagnostic software! 🚀

Welcome to Machine Gnostics! 🧲⚡🔬
