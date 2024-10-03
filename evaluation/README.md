## Evaluation Cases / Synthetic Examples
These Examples are used to evaluate the solution.

*Note: on Linux machines, you might need to use `python3` instead of `python`*

### Run all test
In the project root:
```commandline
python -m unittest evaluation.test_resilience_metrics
```

### Run one specific class of tests or one specific test
In the project root:

```commandline
python -m unittest evaluation.test_resilience_metrics.TestTriangular.test_t_long
```

```commandline
python -m unittest evaluation.test_resilience_metrics.TestParabolic.test_p_root
```
