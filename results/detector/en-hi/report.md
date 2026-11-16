# Detector report - en-hi

Features: 41 in groups ['rules', 'lex', 'labse', 'align', 'roundtrip', 'surface']. Selection by 5-fold sentence-grouped CV over train+val (1994 examples); the test split was not used for any choice.

## logreg (cross-validated, params {'C': 0.03}; macro-F1 95% CI [0.837, 0.872])

| class | precision | recall | F1 | MCC | support |
|---|---|---|---|---|---|
| OK | 0.879 | 0.896 | 0.887 | 0.772 | 997 |
| OMISSION | 0.826 | 0.590 | 0.689 | 0.661 | 266 |
| MISTRANSLATION | 0.890 | 0.936 | 0.912 | 0.899 | 267 |
| ENTITY_NUMBER | 0.861 | 0.914 | 0.887 | 0.874 | 197 |
| HALLUCINATION | 0.856 | 0.955 | 0.903 | 0.888 | 267 |
| *any error vs OK* | 0.894 | 0.877 | 0.885 | 0.772 | 997 |

macro-F1 (5 classes) 0.856, accuracy 0.870, multiclass MCC 0.811

## gbm (cross-validated, params {'max_depth': 3, 'max_iter': 100}; macro-F1 95% CI [0.832, 0.870])

| class | precision | recall | F1 | MCC | support |
|---|---|---|---|---|---|
| OK | 0.864 | 0.902 | 0.883 | 0.761 | 997 |
| OMISSION | 0.864 | 0.575 | 0.691 | 0.671 | 266 |
| MISTRANSLATION | 0.856 | 0.959 | 0.905 | 0.891 | 267 |
| ENTITY_NUMBER | 0.907 | 0.838 | 0.871 | 0.858 | 197 |
| HALLUCINATION | 0.858 | 0.951 | 0.902 | 0.888 | 267 |
| *any error vs OK* | 0.897 | 0.859 | 0.877 | 0.761 | 997 |

macro-F1 (5 classes) 0.850, accuracy 0.866, multiclass MCC 0.804

Best by CV macro-F1: **logreg**

Baseline - threshold on `labse_cos` only (any-error F1, tuned on train+val): 0.707

## Ablation (grouped CV, best model)

| run | #features | macro-F1 | any-F1 | F1 OMISSION | F1 MISTRANSLATION | F1 ENTITY_NUMBER | F1 HALLUCINATION |
|---|---|---|---|---|---|---|---|
| all | 41 | 0.856 | 0.885 | 0.689 | 0.912 | 0.887 | 0.903 |
| drop:rules | 26 | 0.701 | 0.833 | 0.693 | 0.690 | 0.406 | 0.886 |
| drop:lex | 35 | 0.842 | 0.868 | 0.711 | 0.918 | 0.813 | 0.903 |
| drop:labse | 40 | 0.854 | 0.880 | 0.702 | 0.910 | 0.883 | 0.888 |
| drop:align | 32 | 0.853 | 0.875 | 0.717 | 0.917 | 0.886 | 0.868 |
| drop:roundtrip | 35 | 0.838 | 0.876 | 0.679 | 0.884 | 0.873 | 0.881 |
| drop:surface | 37 | 0.854 | 0.881 | 0.699 | 0.913 | 0.886 | 0.891 |
| only:rules | 15 | 0.493 | 0.688 | 0.081 | 0.884 | 0.792 | 0.307 |
| only:lex | 6 | 0.150 | 0.669 | 0.000 | 0.000 | 0.400 | 0.253 |
| only:labse | 1 | 0.138 | 0.667 | 0.365 | 0.143 | 0.185 | 0.000 |
| only:align | 9 | 0.377 | 0.664 | 0.556 | 0.278 | 0.202 | 0.832 |
| only:roundtrip | 6 | 0.489 | 0.744 | 0.616 | 0.022 | 0.405 | 0.610 |

## Span-level F1 (cross-validated predictions, train+val)

Gated by predicted type:

| type | side | precision | recall | F1 | gold chars |
|---|---|---|---|---|---|
| OMISSION | src (silver) | 0.506 | 0.398 | 0.445 | 6349 |
| MISTRANSLATION | tgt | 0.813 | 0.973 | 0.886 | 904 |
| MISTRANSLATION | src (silver) | 0.812 | 0.736 | 0.772 | 159 |
| ENTITY_NUMBER | tgt | 0.796 | 0.974 | 0.876 | 645 |
| ENTITY_NUMBER | src (silver) | 0.360 | 0.573 | 0.442 | 206 |
| HALLUCINATION | tgt | 0.518 | 0.809 | 0.632 | 11579 |

Ungated (raw signal evidence):

| type | side | precision | recall | F1 | gold chars |
|---|---|---|---|---|---|
| OMISSION | src (silver) | 0.187 | 0.448 | 0.263 | 6349 |
| MISTRANSLATION | tgt | 0.783 | 1.000 | 0.879 | 904 |
| MISTRANSLATION | src (silver) | 0.614 | 0.830 | 0.706 | 159 |
| ENTITY_NUMBER | tgt | 0.678 | 1.000 | 0.808 | 645 |
| ENTITY_NUMBER | src (silver) | 0.173 | 0.777 | 0.283 | 206 |
| HALLUCINATION | tgt | 0.207 | 0.840 | 0.333 | 11579 |

(src side = silver gold from alignment; 11 of 355 deletion examples could not be mapped)

## FINAL held-out test (logreg, evaluated once)

| class | precision | recall | F1 | MCC | support |
|---|---|---|---|---|---|
| OK | 0.860 | 0.857 | 0.858 | 0.717 | 1012 |
| OMISSION | 0.755 | 0.608 | 0.674 | 0.636 | 263 |
| MISTRANSLATION | 0.873 | 0.909 | 0.891 | 0.874 | 264 |
| ENTITY_NUMBER | 0.843 | 0.900 | 0.871 | 0.855 | 221 |
| HALLUCINATION | 0.829 | 0.920 | 0.873 | 0.854 | 264 |
| *any error vs OK* | 0.857 | 0.861 | 0.859 | 0.717 | 1012 |

macro-F1 (5 classes) 0.833, accuracy 0.844, multiclass MCC 0.774

macro-F1 95% CI 0.815-0.851; any-error F1 95% CI 0.843-0.874
Baseline (LaBSE threshold only) any-error F1 on test: 0.697

Span-level F1 on test (gated):

| type | side | precision | recall | F1 | gold chars |
|---|---|---|---|---|---|
| OMISSION | src (silver) | 0.462 | 0.415 | 0.437 | 6456 |
| MISTRANSLATION | tgt | 0.800 | 0.950 | 0.869 | 922 |
| MISTRANSLATION | src (silver) | 0.606 | 0.725 | 0.660 | 91 |
| ENTITY_NUMBER | tgt | 0.826 | 0.934 | 0.877 | 755 |
| ENTITY_NUMBER | src (silver) | 0.338 | 0.528 | 0.412 | 326 |
| HALLUCINATION | tgt | 0.506 | 0.770 | 0.611 | 11494 |