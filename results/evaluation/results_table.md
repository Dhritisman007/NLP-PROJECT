### Correction results on the held-out test split (synthetic errors + clean twins)

| pair   | strategy    | scope                   |    n |   flagged |   restored_exactly |   restored_given_flagged |   flags_cleared |   chrF_before |   chrF_after |   chrF_gain |   made_worse |
|:-------|:------------|:------------------------|-----:|----------:|-------------------:|-------------------------:|----------------:|--------------:|-------------:|------------:|-------------:|
| en-hi  | raw         | Omission                |  263 |     0.673 |              0     |                    0     |           0     |         0.778 |        0.778 |      0      |        0     |
| en-hi  | raw         | Mistranslation          |  264 |     0.932 |              0     |                    0     |           0     |         0.963 |        0.963 |      0      |        0     |
| en-hi  | raw         | Entity / number         |  221 |     0.914 |              0     |                    0     |           0     |         0.946 |        0.946 |      0      |        0     |
| en-hi  | raw         | Hallucination           |  264 |     0.932 |              0     |                    0     |           0     |         0.927 |        0.927 |      0      |        0     |
| en-hi  | raw         | ALL errors              | 1012 |     0.861 |              0     |                    0     |           0     |         0.902 |        0.902 |      0      |        0     |
| en-hi  | raw         | Clean (over-correction) | 1012 |     0.143 |                    |                          |                 |               |              |             |        0     |
| en-hi  | constrained | Omission                |  263 |     0.673 |              0     |                    0     |           0.006 |         0.778 |        0.778 |     -0.0001 |        0.004 |
| en-hi  | constrained | Mistranslation          |  264 |     0.932 |              0     |                    0     |           0     |         0.963 |        0.962 |     -0.0004 |        0.008 |
| en-hi  | constrained | Entity / number         |  221 |     0.914 |              0.647 |                    0.708 |           0.673 |         0.946 |        0.978 |      0.0316 |        0.027 |
| en-hi  | constrained | Hallucination           |  264 |     0.932 |              0     |                    0     |           0     |         0.927 |        0.927 |      0      |        0     |
| en-hi  | constrained | ALL errors              | 1012 |     0.861 |              0.141 |                    0.164 |           0.157 |         0.902 |        0.909 |      0.0068 |        0.009 |
| en-hi  | constrained | Clean (over-correction) | 1012 |     0.143 |                    |                          |                 |               |              |             |        0.005 |
| en-de  | raw         | Omission                |  266 |     0.767 |              0     |                    0     |           0     |         0.76  |        0.76  |      0      |        0     |
| en-de  | raw         | Mistranslation          |  267 |     0.948 |              0     |                    0     |           0     |         0.967 |        0.967 |      0      |        0     |
| en-de  | raw         | Entity / number         |  212 |     0.948 |              0     |                    0     |           0     |         0.958 |        0.958 |      0      |        0     |
| en-de  | raw         | Hallucination           |  267 |     0.97  |              0     |                    0     |           0     |         0.922 |        0.922 |      0      |        0     |
| en-de  | raw         | ALL errors              | 1012 |     0.906 |              0     |                    0     |           0     |         0.899 |        0.899 |      0      |        0     |
| en-de  | raw         | Clean (over-correction) | 1012 |     0.079 |                    |                          |                 |               |              |             |        0     |
| en-de  | constrained | Omission                |  266 |     0.767 |              0     |                    0     |           0     |         0.76  |        0.76  |     -0.0003 |        0.004 |
| en-de  | constrained | Mistranslation          |  267 |     0.948 |              0     |                    0     |           0     |         0.967 |        0.967 |     -0.0003 |        0.004 |
| en-de  | constrained | Entity / number         |  212 |     0.948 |              0.675 |                    0.711 |           0.716 |         0.958 |        0.985 |      0.0264 |        0.028 |
| en-de  | constrained | Hallucination           |  267 |     0.97  |              0     |                    0     |           0     |         0.922 |        0.922 |      0      |        0     |
| en-de  | constrained | ALL errors              | 1012 |     0.906 |              0.141 |                    0.156 |           0.157 |         0.899 |        0.905 |      0.0054 |        0.008 |
| en-de  | constrained | Clean (over-correction) | 1012 |     0.079 |                    |                          |                 |               |              |             |        0.008 |
