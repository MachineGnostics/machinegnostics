# How to Use Machine Gnostics Agents & Skills

**Quick Answer:** Use agents to delegate tasks and get expert help from specialized AI agents. Use skills to enhance agent capabilities with specific techniques.

---

## 🎯 Quick Start (30 seconds)

### Step 1: Identify Your Task
```
What do I need to do?
  → Add MAGNET layer/activation?       → Magnet Expert 🧠
  → Improve calibration?               → Magcal Expert 📊
  → Create metrics?                    → Metrics Agent 📈
  → Build ML model?                    → ML Models Specialist 🤖
  → Setup MLflow/cloud?                → ML Integration Specialist ☁️
  → Code review?                       → Manager Agent 🎯
  → Write tests?                       → Unit Tester Agent 🧪
  → Document code?                     → Documentation Agent 📚
```

### Step 2: Find Agent Documentation
```
Location: .github/agents/[agent-name].md
Example: .github/agents/magnet-expert.md
```

### Step 3: Copy Example Prompt
```
Find "Example Prompts" section in agent file
Modify prompt for your specific task
Send to the agent
```

### Step 4: Get Result
Agent completes task and returns work

---

## 📋 Full Workflow

### Workflow 1: Simple Task (Single Agent)

```
You → Manager Agent: "I need a Dense layer for MAGNET"
        ↓
Manager: "Use Magnet Expert"
        ↓
You → Magnet Expert: "Implement Dense layer with proper docstrings"
        ↓
Magnet Expert: 
  • Creates dense.py in magnet/layers/
  • Adds proper module docstring and file header
  • Implements Dense class with full docstring
  • Includes example usage
  • Returns code ready for review
        ↓
Result: ✅ Dense layer implementation complete
```

### Workflow 2: Complex Task (Multiple Agents)

```
You → Manager Agent: "Build end-to-end diagnostic pipeline"
        ↓
Manager coordinates:
  Step 1: Magcal Expert → Implement data preprocessing
  Step 2: Magnet Expert → Build neural network model
  Step 3: Metrics Agent → Create diagnostic metrics
  Step 4: ML Models → Implement training pipeline
  Step 5: Unit Tester → Write comprehensive tests
  Step 6: Documentation → Create user guides
        ↓
Manager: Integrates all components and validates
        ↓
Result: ✅ Complete pipeline with tests & documentation
```

### Workflow 3: Quality Gate (Before Finalizing)

```
You → Manager Agent: "Review these files for standards compliance"
        ↓
Manager checks:
  ✓ File headers with docstrings
  ✓ Each public class in dedicated file
  ✓ Docstrings on all public methods
  ✓ Proper imports (depth-2 constraint)
  ✓ __all__ definitions
  ✓ Type hints present
        ↓
Manager: "Ready to merge" or "Fix these issues..."
        ↓
Result: ✅ Code approved or issues identified
```

---

## 🤖 How to Talk to an Agent

### Format 1: Direct Task Request
```
"Implement Fi activation function in magnet/activations/fi.py with:
  • Standard docstring format
  • Based on sech(2θ) mathematics
  • Range validation (0, 1)
  • Example usage in docstring
  • Type hints on all parameters"
```

### Format 2: Using Example Prompts
Each agent has ready-to-use prompts. Find them in agent files:

**From magnet-expert.md:**
```
"Implement a new MAGNET layer with full docstrings and type hints"
"Create activation functions for gnostic neural networks"
"Develop Sequential model supporting gnostic layers"
```

### Format 3: Problem + Guidance
```
"I need to add batch normalization to MAGNET. 
Should it go in magnet/layers/batch_norm.py?
What should the public API look like?
Please implement with tests."
```

---

## 🧠 Understanding Agent Expertise

Each agent has deep knowledge in specific areas:

### 1️⃣ Magnet Expert 🧠
**Knows:** MAGNET framework, PyTorch, gnostic theory, activations, losses  
**Can:** Implement any MAGNET component with mathematical rigor  
**Use when:** Working with magnet/ module

**Example prompt:**
```
"Implement Fi activation function with:
- Math: f = sech(2θ)
- Proper docstring with mathematical explanation
- Conservation identity validation (f² + h² = 1.0)
- Type hints and examples"
```

### 2️⃣ Magcal Expert 📊
**Knows:** Calibration, electromagnetic calculations, NumPy, numerical methods  
**Can:** Implement calibration algorithms with precision  
**Use when:** Working with magcal/ module

**Example prompt:**
```
"Improve calibration algorithm for magnetic sensors:
- Current issue: accuracy drops at high temperatures
- Need: Temperature compensation factor
- Reference: See calibration_config docstring
- Must maintain numerical precision"
```

### 3️⃣ Metrics Agent 📈
**Knows:** Diagnostic metrics, statistics, validation  
**Can:** Create correct, validated metrics  
**Use when:** Adding new metrics to system

**Example prompt:**
```
"Create diagnostic accuracy metric that:
- Compares fidelity values (Fi activations)
- Handles edge cases (zero division)
- Returns normalized score [0, 1]
- Includes validation checks"
```

### 4️⃣ ML Models Specialist 🤖
**Knows:** ML models, training pipelines, scikit-learn, PyTorch  
**Can:** Build and optimize ML models  
**Use when:** Working with ml_models/ module

**Example prompt:**
```
"Build anomaly detection model using:
- MAGNET neural network backbone
- Diagnostic metrics for classification
- Hyperparameter tuning strategy
- Cross-validation evaluation"
```

### 5️⃣ ML Integration Specialist ☁️
**Knows:** MLflow, cloud platforms, deployment, experiment tracking  
**Can:** Setup cloud infrastructure and pipelines  
**Use when:** Setting up integrations/ module

**Example prompt:**
```
"Setup MLflow integration for:
- Experiment tracking with MAGNET models
- Model versioning and registry
- Artifact storage in S3
- Model serving with Docker"
```

### 6️⃣ Manager Agent 🎯
**Knows:** Project standards, architecture, code quality  
**Can:** Review, coordinate, and enforce standards  
**Use when:** Need code review or multi-agent coordination

**Example prompt:**
```
"Review these files for standards compliance:
- Check file organization (one class per file)
- Verify docstring completeness
- Validate import depth (max 2)
- Check __all__ definitions
- Ensure type hints present"
```

### 7️⃣ Unit Tester Agent 🧪
**Knows:** Testing strategies, pytest, edge cases, coverage  
**Can:** Write comprehensive tests  
**Use when:** Need test coverage for code

**Example prompt:**
```
"Write tests for Dense layer:
- Test basic forward pass
- Test shape validation
- Test edge cases (zero input, infinity)
- Test mathematical correctness
- Aim for 95%+ coverage"
```

### 8️⃣ Documentation Agent 📚
**Knows:** API documentation, tutorials, technical writing  
**Can:** Create comprehensive documentation  
**Use when:** Need docs for new features

**Example prompt:**
```
"Create documentation for Fi activation:
- Overview section (what is Fi?)
- Mathematical foundation
- API reference with examples
- Usage in neural networks
- Performance notes"
```

---

## 💎 Using Skills to Enhance Agents

Skills supercharge agent capabilities. Each agent has recommended skills:

### Available Skills

| Skill | Purpose | Best For |
|-------|---------|----------|
| python-fact-grounded-coding | Verify math implementations | MAGNET, Magcal, Metrics agents |
| python-add-type-annotations | Add type safety | All agents |
| python-type-inference | Understand complex types | Manager, ML Models agents |
| pylance-docs | Audit documentation | Documentation, Manager agents |
| pylance-python-profiling | Performance optimization | ML Models, Magnet agents |
| pylance-refactoring | Code quality & standards | Manager, Unit Tester agents |

### How to Use a Skill with an Agent

**Example: Use fact-grounded-coding with Magnet Expert**

```
Agent: Magnet Expert
Skill: python-fact-grounded-coding

Prompt:
"Implement Fi activation function with FACT-GROUNDED validation:
- Validate mathematical correctness: f = sech(2θ)
- Verify conservation identity: f² + h² = 1.0
- Test with known inputs (θ=0, θ=±∞)
- Use Pylance to ground implementation in mathematics"
```

**Example: Use profiling with ML Models Specialist**

```
Agent: ML Models Specialist
Skill: pylance-python-profiling

Prompt:
"Optimize training loop performance:
- Profile current training implementation
- Identify bottlenecks (CPU vs GPU)
- Optimize hot paths
- Target 20% performance improvement"
```

---

## 📍 Where to Find Documentation

```
Getting Started:
  .github/README.md                    ← Main index
  
Agent Information:
  .github/agents/README.md             ← Agent index & selection guide
  .github/agents/[agent-name].md       ← Specific agent details
  
Skill Information:
  .github/skills/README.md             ← Skill index & decision tree
  .github/skills/[skill-name].md       ← Specific skill details
  
Project Standards:
  .github/copilot-instructions.md      ← Project philosophy & rules
  .github/STANDARDS.md                 ← Code standards & guidelines
  .github/MAGNET-CORRECTION.md         ← MAGNET framework details
  .github/IMPORT-ARCHITECTURE.md       ← Import system design
  .github/IMPORT-GUIDE.md              ← Import implementation
```

---

## 🎯 Common Use Cases & Recommended Agents

### Scenario 1: "Add new MAGNET layer"

**Agents to use (in order):**
1. **Manager** - Coordinate and plan
2. **Magnet Expert** - Implement layer
3. **Unit Tester** - Write tests
4. **Documentation** - Write docs
5. **Manager** - Final review

**Skills to use:**
- python-fact-grounded-coding (validate math)
- python-add-type-annotations (type safety)
- pylance-docs (doc quality)

---

### Scenario 2: "Improve calibration accuracy"

**Agents to use:**
1. **Manager** - Analyze problem
2. **Magcal Expert** - Fix algorithm
3. **Unit Tester** - Validate improvements
4. **Manager** - Code review

**Skills:**
- python-fact-grounded-coding (numerical validation)
- python-type-inference (complex types)

---

### Scenario 3: "Deploy model to MLflow"

**Agents to use:**
1. **Manager** - Plan deployment
2. **ML Models** - Prepare model
3. **ML Integration** - Setup MLflow
4. **Unit Tester** - Test deployment
5. **Documentation** - Create deployment guide

**Skills:**
- pylance-refactoring (clean code)
- python-add-type-annotations (API safety)

---

### Scenario 4: "Full code review before release"

**Agent:** Manager Agent

**Prompt:**
```
"Full standards compliance review:
- File organization (one class per file)
- Docstrings complete (module, class, method)
- Imports follow depth-2 constraint
- Type hints on all public APIs
- __all__ properly defined
- No circular dependencies
- Test coverage adequate
- Documentation complete

Report issues by severity and file."
```

---

## 🚀 Pro Tips

### Tip 1: Start with Manager
If unsure which agent to use, start with **Manager Agent**. They'll route you to the right specialist.

```
You → Manager: "I need to add anomaly detection to MAGNET"
Manager: "Use ML Models Specialist + Magnet Expert coordination"
```

### Tip 2: Use Example Prompts as Templates
Don't write prompts from scratch. Find similar examples in agent files and modify them.

**From magnet-expert.md:**
```
"Implement a new MAGNET layer with full docstrings and type hints"
→ Modify to: "Implement Conv2D layer for MAGNET..."
```

### Tip 3: Chain Agents for Complex Work
Complex features need multiple agents:

```
Simple:     Magcal Expert → Done
Medium:     Magnet Expert → Unit Tester → Done
Complex:    Manager → Expert → Tester → Documentation → Manager review
```

### Tip 4: Use Skills to Verify Quality
After an agent completes work, use skills to verify:

```
Agent: Created Fi activation
Skill: python-fact-grounded-coding → Verify math
Skill: python-add-type-annotations → Add types
Skill: pylance-docs → Check docstrings
Result: High-quality, well-documented code
```

### Tip 5: Reference Standards
All agents know project standards. Reference them:

```
"Follow .github/STANDARDS.md file organization standard"
"Use import patterns from .github/IMPORT-GUIDE.md"
"Ensure docstrings match .github/STANDARDS.md templates"
```

---

## 📞 Agent Availability

All 8 agents are available now:

✅ **Magnet Expert** - MAGNET framework  
✅ **Magcal Expert** - Calibration  
✅ **Metrics Agent** - Diagnostic metrics  
✅ **ML Models Specialist** - ML models  
✅ **ML Integration Specialist** - Cloud/MLflow  
✅ **Manager Agent** - Coordination  
✅ **Unit Tester Agent** - Testing  
✅ **Documentation Agent** - Documentation  

All 6 skills are available:

✅ python-fact-grounded-coding  
✅ python-add-type-annotations  
✅ python-type-inference  
✅ pylance-docs  
✅ pylance-python-profiling  
✅ pylance-refactoring  

---

## 🎓 Learning Path

### Day 1: Get Familiar
1. Read `.github/README.md` (5 min)
2. Skim `.github/agents/README.md` (5 min)
3. Read one agent file completely (e.g., magnet-expert.md) (10 min)

### Day 2: Try It
1. Find a small task (add a metric, write a test)
2. Pick appropriate agent from index
3. Copy example prompt and modify
4. Send to agent and iterate

### Day 3: Use Skills
1. Read `.github/skills/README.md` (5 min)
2. Use fact-grounded-coding with Magnet Expert (10 min)
3. Try refactoring skill with Manager (10 min)

### Week 1: Complex Projects
1. Start complex task with Manager
2. Let Manager coordinate agents
3. Review final work with Manager + Skills

---

## ✅ Verification Checklist

Before submitting work to production:

- [ ] Used correct agent for task
- [ ] Followed example prompts from agent file
- [ ] Code passes Manager Agent review
- [ ] Docstrings match STANDARDS.md template
- [ ] Files organized (one class per file)
- [ ] Imports follow depth-2 constraint
- [ ] Type hints on public APIs
- [ ] Tests written and passing
- [ ] Documentation created
- [ ] No breaking changes to existing code

---

## 📞 Need Help?

**Question:** Which agent should I use?  
**Answer:** Check "Quick Selection Guide" above or ask Manager Agent

**Question:** How do I know if task is done?  
**Answer:** Send to Manager Agent for final review

**Question:** Can I use multiple skills?  
**Answer:** Yes! Use as many as needed. Each agent has recommended skills.

**Question:** What if agent makes mistakes?  
**Answer:** Iterate! Provide feedback and agent will refine work.

**Question:** Can agents coordinate automatically?  
**Answer:** Yes! Manager Agent coordinates multi-agent tasks. Just tell Manager your goal.

---

## 🎯 Success Criteria

Your agents are working well when:

✅ Tasks complete with high quality  
✅ Code passes all standards checks  
✅ Documentation is complete  
✅ Tests have good coverage  
✅ No back-and-forth needed (first try success)  
✅ Code is maintainable and clear  
✅ Team understands what was built  

---

**Happy Building!** 🚀

Start with a small task, pick the right agent, and watch it work. The agents are experts in their domains—trust their expertise and iterate as needed.
