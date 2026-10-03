# pylance-python-profiling Skill

## Overview
Profiles Python code to identify CPU time hotspots, trace execution flows, and analyze memory usage using advanced profiling tools.

## Purpose

This skill enables:
- CPU profiling to find performance bottlenecks
- Call tree analysis to understand execution flow
- Memory profiling to identify leaks
- Trace analysis for optimization opportunities

## Use This Skill When

✅ **Training loops are slow**
  - MAGNET model training performance
  - Multi-scale optimization issues
  - Batch processing bottlenecks

✅ **Test suite performance**
  - Test execution time exceeds targets
  - Fixtures are slow to create
  - Data loading takes too long

✅ **Data pipelines need optimization**
  - Data conversion in magcal
  - Metrics computation on large batches
  - Model inference performance

✅ **Memory issues**
  - Large batch processing OOM
  - Memory leaks in training loops
  - Neural network memory usage

## Primary Agents

- 🧠 **Magnet Expert** - Optimizes MAGNET training loops
- 🧪 **Unit Tester** - Optimizes slow test suite
- 🤖 **ML Models Specialist** - Improves training performance

## Example Prompts

### For Magnet Expert (Training Optimization)
```
"Use pylance-python-profiling to profile the MAGNET training loop.

Analyze:
1. Which operations consume most CPU time?
2. Where are the bottlenecks in forward/backward pass?
3. Can matrix operations be optimized?
4. Is batch size appropriate?
5. Suggest optimizations for 20%+ speedup"
```

### For Unit Tester (Test Performance)
```
"Profile test_magnet/test_layers.py to optimize test execution.

Identify:
1. The 5 slowest test functions
2. Whether fixtures are slow or tests themselves
3. Data loading bottlenecks
4. Memory issues in test setup/teardown
5. Recommendations for 20%+ speedup"
```

### For ML Models Specialist (Training Optimization)
```
"Profile model training loop to identify performance issues.

Check:
1. Data loading vs training time split
2. GPU utilization during training
3. Memory usage pattern
4. Gradient computation time
5. Optimization to reach target speed"
```

## Profiling Output

Typical output includes:
```
Total Time: 125.4 seconds

Top Functions by CPU Time:
  1. forward_pass: 45.2s (36%)
  2. backward_pass: 38.1s (30%)
  3. data_loading: 22.3s (18%)
  4. metrics_computation: 12.1s (10%)
  5. other: 7.7s (6%)

Memory Usage:
  Peak: 8.2 GB
  Leaked: 0 MB
```

## Optimization Strategies

### For Training Loops
```python
# Profile identifies forward_pass as bottleneck
# Solutions to try:
1. Use half-precision (torch.float16)
2. Increase batch size
3. Use gradient accumulation
4. Enable cudnn benchmarking
5. Use mixed-precision training
```

### For Data Loading
```python
# Profile shows data_loading is slow
# Solutions:
1. Use DataLoader with num_workers > 0
2. Prefetch data in background
3. Cache preprocessed data
4. Use faster I/O (SSD vs HDD)
5. Consider data augmentation overhead
```

### For Memory Issues
```python
# Profile shows memory leak
# Debugging steps:
1. Check tensor references aren't accumulating
2. Clear model.eval() cache
3. Use torch.no_grad() in inference
4. Monitor GPU memory with nvidia-smi
5. Use memory_profiler for detailed analysis
```

## Success Criteria

✅ Bottlenecks clearly identified  
✅ CPU/memory hotspots found  
✅ Optimization recommendations provided  
✅ Performance improvement measured  
✅ No quality regression after optimization  

## Recommended Workflow

1. **Set baseline** - Measure current performance
2. **Profile** - Use skill to identify bottlenecks
3. **Prioritize** - Focus on largest time consumers
4. **Optimize** - Implement targeted improvements
5. **Measure** - Verify performance gains
6. **Repeat** - Profile again to find next bottleneck

## Related Skills

- `python-fact-grounded-coding` - For algorithm validation
- `python-add-type-annotations` - For cleaner code
- `pylance-refactoring` - For code reorganization
