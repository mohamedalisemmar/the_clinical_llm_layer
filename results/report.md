# Evaluation report: LLM only vs LLM + intermediary layer

- AI outputs: provider `mock`, model `mock-clinical-simulator`, generated 2026-09-30T20:03:48+00:00
- Participants: 0, case responses: 0

> WARNING: AI outputs come from the mock simulator. These numbers are not findings.

## 6. Reliability indicator vs actual correctness

```
output_correct     correct  incorrect  share_correct
layer_reliability                                   
LOW                      4          6            0.4
MODERATE                10          0            1.0
HIGH                    10          0            1.0
```

- Incorrect outputs flagged LOW: 6 of 6 (100%).
- Incorrect outputs rated HIGH (missed): 0 of 6.
- Correct outputs rated HIGH: 10 of 24; rated LOW (false alarm): 4 of 24.
- False alarms (LOW but correct): A01, A04, A05, M04.
- Consistency is not correctness: incorrect outputs whose repeated answers were nevertheless consistent: none.

## 7. Which signals carry information

```
correct                                    flagged on incorrect outputs  flagged on correct outputs
signal                                                                                             
Agreement across prompting strategies                               1.0                        0.29
Completeness of requested fields                                    0.0                        0.00
Consistency across repeated answers                                 1.0                        0.33
Contradictions                                                      1.0                        0.00
Inconsistencies in the case data                                    0.0                        0.12
Missing information                                                 0.0                        0.21
Model's stated confidence (supplementary)                           0.0                        0.21
```

- Share of cases in which each signal warned or failed. A useful signal fires far more often on incorrect outputs than on correct ones.

## 9. Content of the clinician rationale (Function C)

```
                     factors  missing (critical)  alternatives  has 'what would change'  empty sections
category                                                                                               
ambiguous               3.83                 0.0           3.0                      1.0             0.0
contradictory_input     3.67                 0.0           3.0                      1.0             0.0
missing_information     3.20                 4.0           3.0                      1.0             0.0
seeded_error            2.00                 0.0           3.0                      1.0             0.0
straightforward         4.00                 0.0           3.0                      1.0             0.0
```


## 10. Cost and latency (is the layer lightweight?)

```
                                          LLM only  LLM + layer  Ratio (layer / only)
LLM calls per case                          1.0000      10.0417                  10.0
Generation time per case (s, sequential)    0.0000       0.0000                   NaN
Input tokens per case                     194.7917    4164.4583                  21.4
Output tokens per case                    110.7083    1085.4583                   9.8
Estimated cost per case (USD)               0.0007       0.0096                  12.8
```

- Averaged over 24 generated cases. Prices: 1.0 / 5.0 USD per million input / output tokens (set in .env).
- Mean seconds by step: baseline 0.0, structure 0.0, structured_b 0.0, critical_c 0.0, contradiction 0.0, judge 0.0, equivalence 0.0.
- Outputs are precomputed, so participants never wait for this; the repeated Prompt B calls are independent and could run in parallel in a deployed version.

## Figures

- `results/figures/reliability_vs_correctness.png`