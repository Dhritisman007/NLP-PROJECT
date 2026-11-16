### Detector, held-out test split (evaluated once)

| pair   | model   | class           |   precision |   recall |    F1 |   MCC |   support |
|:-------|:--------|:----------------|------------:|---------:|------:|------:|----------:|
| en-hi  | logreg  | OK              |       0.86  |    0.857 | 0.858 | 0.717 |      1012 |
| en-hi  | logreg  | OMISSION        |       0.755 |    0.608 | 0.674 | 0.636 |       263 |
| en-hi  | logreg  | MISTRANSLATION  |       0.873 |    0.909 | 0.891 | 0.874 |       264 |
| en-hi  | logreg  | ENTITY_NUMBER   |       0.843 |    0.9   | 0.871 | 0.855 |       221 |
| en-hi  | logreg  | HALLUCINATION   |       0.829 |    0.92  | 0.873 | 0.854 |       264 |
| en-hi  | logreg  | any error vs OK |       0.857 |    0.861 | 0.859 | 0.717 |      1012 |
| en-de  | gbm     | OK              |       0.907 |    0.921 | 0.914 | 0.827 |      1012 |
| en-de  | gbm     | OMISSION        |       0.807 |    0.737 | 0.77  | 0.738 |       266 |
| en-de  | gbm     | MISTRANSLATION  |       0.961 |    0.921 | 0.941 | 0.932 |       267 |
| en-de  | gbm     | ENTITY_NUMBER   |       0.868 |    0.929 | 0.897 | 0.886 |       212 |
| en-de  | gbm     | HALLUCINATION   |       0.956 |    0.97  | 0.963 | 0.957 |       267 |
| en-de  | gbm     | any error vs OK |       0.92  |    0.906 | 0.913 | 0.827 |      1012 |
