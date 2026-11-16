### Baselines (a)-(d) of the project brief

| pair   | method                                                 | flags_errors   | false_alarms_on_clean   | errors_restored_exactly   | chrF_gain_on_errors   | clean_sentences_changed   |
|:-------|:-------------------------------------------------------|:---------------|:------------------------|:--------------------------|:----------------------|:--------------------------|
| en-hi  | (a) raw MT                                             |                |                         | 0.0                       | 0.0                   | 0.0                       |
| en-hi  | (b) fixed threshold on LaBSE similarity (flag only)    | 0.842          | 0.573                   | cannot correct            | cannot correct        | cannot correct            |
| en-hi  | (c) LLM post-editing, no spans                         | not run        | not run                 | not run                   | not run               | not run                   |
| en-hi  | (d) full pipeline = detector + rules + span-guided LLM | not run        | not run                 | not run                   | not run               | not run                   |
| en-hi  | (d') detector + rule-based fixer (no LLM)              | 0.861          | 0.143                   | 0.141                     | 0.0068                | 0.005                     |
| en-de  | (a) raw MT                                             |                |                         | 0.0                       | 0.0                   | 0.0                       |
| en-de  | (b) fixed threshold on LaBSE similarity (flag only)    | 0.894          | 0.637                   | cannot correct            | cannot correct        | cannot correct            |
| en-de  | (c) LLM post-editing, no spans                         | not run        | not run                 | not run                   | not run               | not run                   |
| en-de  | (d) full pipeline = detector + rules + span-guided LLM | not run        | not run                 | not run                   | not run               | not run                   |
| en-de  | (d') detector + rule-based fixer (no LLM)              | 0.906          | 0.079                   | 0.141                     | 0.0054                | 0.008                     |
