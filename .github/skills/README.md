# Machine Gnostics - Skills Index

**Location:** `.github/skills/`

This directory contains detailed specifications for the 6 specialized Copilot skills available to Machine Gnostics agents.

---

## 🎯 All Skills

### Mathematical & Theory Validation

1. **[python-fact-grounded-coding.md](./python-fact-grounded-coding.md)** 📐
   - Grounds code analysis in verified facts and specifications
   - Validates mathematical implementations
   - Primary agents: Magnet, Magcal, Metrics, ML Models
   - **When to use:** Implementing formulas, validating algorithms
   - **Example:** Verify Fi layer computes f = sech(2θ) correctly

### Type Safety & Documentation

2. **[python-add-type-annotations.md](./python-add-type-annotations.md)** 🏷️
   - Adds/validates inline type annotations
   - Improves IDE support and documentation
   - Primary agents: All agents (public API focus)
   - **When to use:** Publishing API, improving IDE support
   - **Example:** Type all public classes in magnet/ module

3. **[python-type-inference.md](./python-type-inference.md)** 🔍
   - Determines correct types for complex symbols
   - Validates type consistency in data pipelines
   - Primary agents: Magcal, Magnet, Metrics, ML Models
   - **When to use:** Understanding complex types, debugging mismatches
   - **Example:** Infer exact PyTorch tensor types in Dense layer

4. **[pylance-docs.md](./pylance-docs.md)** 📖
   - Audits docstrings for completeness and consistency
   - Validates API documentation
   - Primary agents: Documentation, Manager
   - **When to use:** Creating API docs, quality gates
   - **Example:** Audit all public classes for complete docstrings

### Performance & Quality

5. **[pylance-python-profiling.md](./pylance-python-profiling.md)** ⚡
   - Profiles CPU, memory, and execution flow
   - Identifies performance bottlenecks
   - Primary agents: Magnet Expert, Unit Tester, ML Models
   - **When to use:** Optimization, performance issues
   - **Example:** Profile MAGNET training loop, find bottleneck

### Code Organization & Standards

6. **[pylance-refactoring.md](./pylance-refactoring.md)** 🔧
   - Automates code refactoring and standards enforcement
   - Cleans imports and organizes code
   - Primary agents: Manager Agent (coordination)
   - **When to use:** Import enforcement, code cleanup
   - **Example:** Verify depth-2 import constraint compliance

---

## Skills by Use Case

### 🧠 Implementing Gnostic Theory (MAGNET)
```
python-fact-grounded-coding
├─ Validates f = sech(2θ), h = tanh(2θ)
├─ Verifies f² + h² = 1.0
└─ Grounds in mathematical spec

python-add-type-annotations
├─ Types PyTorch layers
├─ Documents tensor types
└─ Improves IDE support

pylance-python-profiling
├─ Profiles training loops
├─ Finds bottlenecks
└─ Optimizes performance
```

### 📊 Data Calibration (MAGCAL)
```
python-fact-grounded-coding
├─ Validates calibration formulas
├─ Checks numerical precision
└─ Verifies correctness

python-type-inference
├─ Infers numpy array types
├─ Validates shape consistency
└─ Checks dtype preservation
```

### 📈 Metrics Development
```
python-fact-grounded-coding
├─ Grounds formulas in statistics
├─ Validates calculations
└─ Checks edge cases

python-add-type-annotations
├─ Types metric functions
├─ Documents input/output types
└─ Improves usability
```

### 🎯 Code Quality & Standards
```
pylance-refactoring
├─ Enforces import depth (max 2)
├─ Cleans unused imports
└─ Verifies __all__ definitions

python-add-type-annotations
├─ Validates type consistency
├─ Checks public API coverage
└─ Improves code quality
```

### 🧪 Testing & Performance
```
pylance-python-profiling
├─ Identifies slow tests
├─ Optimizes test execution
└─ Profiles test fixtures

python-add-type-annotations
├─ Types test fixtures
├─ Improves test code quality
└─ Ensures type safety
```

### 📚 Documentation
```
pylance-docs
├─ Audits docstring coverage
├─ Checks parameter documentation
└─ Validates Examples sections

python-add-type-annotations
├─ Documents complex types
├─ Creates type reference
└─ Improves usability
```

---

## Skills by Agent

### 🧠 Magnet Expert Agent
- `python-fact-grounded-coding` - Validate gnostic math
- `pylance-python-profiling` - Optimize training loops
- `python-add-type-annotations` - Type PyTorch API

### 📊 Magcal Expert Agent
- `python-fact-grounded-coding` - Verify calibrations
- `python-type-inference` - Validate data types
- `python-add-type-annotations` - Type functions

### 📈 Metrics Agent
- `python-fact-grounded-coding` - Ground in theory
- `python-add-type-annotations` - Type metrics
- `python-type-inference` - Validate types

### 🤖 ML Models Specialist
- `python-fact-grounded-coding` - Validate algorithms
- `pylance-python-profiling` - Optimize training
- `python-add-type-annotations` - Type models

### ☁️ ML Integration Specialist
- `python-add-type-annotations` - Type cloud SDKs
- `python-fact-grounded-coding` - Validate logic

### 🎯 Manager Agent
- `pylance-refactoring` - Enforce standards
- `python-add-type-annotations` - Validate consistency
- `pylance-docs` - Audit documentation

### 🧪 Unit Tester Agent
- `pylance-python-profiling` - Optimize tests
- `python-add-type-annotations` - Type fixtures
- `python-fact-grounded-coding` - Validate test logic

### 📚 Documentation Agent
- `pylance-docs` - Audit docstrings
- `python-add-type-annotations` - Document types
- `python-fact-grounded-coding` - Validate examples

---

## Quick Decision Tree

**What do you need to do?**

```
├─ Validate mathematical formula?
│  └─→ python-fact-grounded-coding
│
├─ Make code type-safe?
│  └─→ python-add-type-annotations
│
├─ Understand complex type?
│  └─→ python-type-inference
│
├─ Audit documentation?
│  └─→ pylance-docs
│
├─ Optimize performance?
│  └─→ pylance-python-profiling
│
└─ Clean code/enforce standards?
   └─→ pylance-refactoring
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- ✅ python-fact-grounded-coding
- ✅ pylance-refactoring
- ✅ python-add-type-annotations

### Phase 2: Enhancement (Weeks 2-3)
- ✅ All Phase 1 skills (expanded use)
- ✅ python-type-inference
- ✅ pylance-docs
- ✅ pylance-python-profiling (per agent)

### Phase 3: Optimization (Weeks 4+)
- ✅ All skills (regular use)
- ✅ Periodic code quality maintenance
- ✅ Performance optimization focus

---

## How to Use These Files

1. **Find relevant skill**: Use decision tree or use case guide
2. **Read skill file**: Understand when to use and how
3. **Review examples**: Check example prompts for your task
4. **Execute with agent**: Tell relevant agent to use skill
5. **Review results**: Validate output and iterate

---

## Reading Individual Skill Files

Each skill file contains:
- **Overview**: What the skill does
- **Purpose**: Why it matters
- **Use This Skill When**: Decision criteria with checklists
- **Primary Agents**: Which agents use this skill
- **Example Prompts**: Ready-to-use commands by agent
- **Best Practices**: How to get best results
- **Success Criteria**: How to know it worked
- **Related Skills**: Complementary skills

---

## Recommended Workflow

1. **Identify problem** - What needs to be done?
2. **Choose skill** - Which skill matches?
3. **Find agent** - Who should use it?
4. **Craft prompt** - Use example as template
5. **Execute** - Tell agent + skill to work
6. **Review** - Validate results
7. **Iterate** - Refine approach if needed

---

## Related Documentation

- **[../agents/](../agents/)** - Agent specifications
- **[../AGENTS-SKILLS-ENHANCEMENT.md](../AGENTS-SKILLS-ENHANCEMENT.md)** - Skills integration guide
- **[../SKILLS-QUICK-START.md](../SKILLS-QUICK-START.md)** - 5-minute quick start
- **[../README.md](../README.md)** - Documentation index
