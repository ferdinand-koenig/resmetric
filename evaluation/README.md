## Evaluation Cases / Synthetic Examples
These Examples are used to evaluate the solution.

### Run all test
In Project root:
```commandline
python -m unittest evaluation.test_resilience_metrics
```

### Run one specific class of tests or one specific test
In project root

```commandline
python -m unittest evaluation.test_resilience_metrics.TestTriangular.test_t_long
```

```commandline
python -m unittest evaluation.test_resilience_metrics.TestParabolic.test_p_root
```