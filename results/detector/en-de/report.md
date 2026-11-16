# Detector report - en-de

Features: 41 in groups ['rules', 'lex', 'labse', 'align', 'roundtrip', 'surface']. Selection by 5-fold sentence-grouped CV over train+val (1994 examples); the test split was not used for any choice.

## logreg (cross-validated, params {'C': 0.3}; macro-F1 95% CI [0.872, 0.903])

| class | precision | recall | F1 | MCC | support |
|---|---|---|---|---|---|
| OK | 0.884 | 0.922 | 0.902 | 0.801 | 997 |
| OMISSION | 0.845 | 0.710 | 0.772 | 0.743 | 269 |
| MISTRANSLATION | 0.949 | 0.896 | 0.921 | 0.910 | 268 |
| ENTITY_NUMBER | 0.871 | 0.916 | 0.893 | 0.882 | 191 |
| HALLUCINATION | 0.945 | 0.963 | 0.954 | 0.947 | 269 |
| *any error vs OK* | 0.918 | 0.879 | 0.898 | 0.801 | 997 |

macro-F1 (5 classes) 0.888, accuracy 0.895, multiclass MCC 0.846

## gbm (cross-validated, params {'max_depth': 3, 'max_iter': 100}; macro-F1 95% CI [0.872, 0.903])

| class | precision | recall | F1 | MCC | support |
|---|---|---|---|---|---|
| OK | 0.891 | 0.915 | 0.903 | 0.804 | 997 |
| OMISSION | 0.821 | 0.732 | 0.774 | 0.743 | 269 |
| MISTRANSLATION | 0.960 | 0.907 | 0.933 | 0.923 | 268 |
| ENTITY_NUMBER | 0.859 | 0.921 | 0.889 | 0.877 | 191 |
| HALLUCINATION | 0.938 | 0.952 | 0.945 | 0.936 | 269 |
| *any error vs OK* | 0.912 | 0.889 | 0.900 | 0.804 | 997 |

macro-F1 (5 classes) 0.889, accuracy 0.895, multiclass MCC 0.846

Best by CV macro-F1: **gbm**

Baseline - threshold on `labse_cos` only (any-error F1, tuned on train+val): 0.711

## Ablation (grouped CV, best model)

| run | #features | macro-F1 | any-F1 | F1 OMISSION | F1 MISTRANSLATION | F1 ENTITY_NUMBER | F1 HALLUCINATION |
|---|---|---|---|---|---|---|---|
| all | 41 | 0.889 | 0.900 | 0.774 | 0.933 | 0.889 | 0.945 |
| drop:rules | 26 | 0.787 | 0.859 | 0.771 | 0.741 | 0.616 | 0.937 |
| drop:lex | 35 | 0.873 | 0.892 | 0.777 | 0.930 | 0.825 | 0.944 |
| drop:labse | 40 | 0.883 | 0.895 | 0.768 | 0.933 | 0.882 | 0.933 |
| drop:align | 32 | 0.871 | 0.889 | 0.764 | 0.928 | 0.879 | 0.889 |
| drop:roundtrip | 35 | 0.864 | 0.883 | 0.737 | 0.897 | 0.861 | 0.943 |
| drop:surface | 37 | 0.892 | 0.905 | 0.787 | 0.940 | 0.885 | 0.942 |
| only:rules | 15 | 0.478 | 0.678 | 0.303 | 0.886 | 0.739 | 0.102 |
| only:lex | 6 | 0.190 | 0.672 | 0.000 | 0.214 | 0.460 | 0.180 |
| only:labse | 1 | 0.188 | 0.685 | 0.282 | 0.244 | 0.179 | 0.007 |
| only:align | 9 | 0.471 | 0.660 | 0.649 | 0.320 | 0.211 | 0.904 |
| only:roundtrip | 6 | 0.628 | 0.759 | 0.690 | 0.684 | 0.445 | 0.523 |

## Span-level F1 (cross-validated predictions, train+val)

Gated by predicted type:

| type | side | precision | recall | F1 | gold chars |
|---|---|---|---|---|---|
| OMISSION | src (silver) | 0.731 | 0.506 | 0.598 | 7371 |
| MISTRANSLATION | tgt | 0.945 | 0.865 | 0.903 | 1174 |
| MISTRANSLATION | src (silver) | 0.713 | 0.784 | 0.747 | 111 |
| ENTITY_NUMBER | tgt | 0.838 | 0.929 | 0.881 | 672 |
| ENTITY_NUMBER | src (silver) | 0.179 | 0.487 | 0.261 | 152 |
| HALLUCINATION | tgt | 0.846 | 0.826 | 0.836 | 14709 |

Ungated (raw signal evidence):

| type | side | precision | recall | F1 | gold chars |
|---|---|---|---|---|---|
| OMISSION | src (silver) | 0.478 | 0.538 | 0.506 | 7371 |
| MISTRANSLATION | tgt | 0.817 | 0.916 | 0.864 | 1174 |
| MISTRANSLATION | src (silver) | 0.511 | 0.838 | 0.635 | 111 |
| ENTITY_NUMBER | tgt | 0.653 | 0.987 | 0.786 | 672 |
| ENTITY_NUMBER | src (silver) | 0.014 | 0.651 | 0.027 | 152 |
| HALLUCINATION | tgt | 0.585 | 0.849 | 0.692 | 14709 |

(src side = silver gold from alignment; 4 of 329 deletion examples could not be mapped)

## FINAL held-out test (gbm, evaluated once)

| class | precision | recall | F1 | MCC | support |
|---|---|---|---|---|---|
| OK | 0.907 | 0.921 | 0.914 | 0.827 | 1012 |
| OMISSION | 0.807 | 0.737 | 0.770 | 0.738 | 266 |
| MISTRANSLATION | 0.961 | 0.921 | 0.941 | 0.932 | 267 |
| ENTITY_NUMBER | 0.868 | 0.929 | 0.897 | 0.886 | 212 |
| HALLUCINATION | 0.956 | 0.970 | 0.963 | 0.957 | 267 |
| *any error vs OK* | 0.920 | 0.906 | 0.913 | 0.827 | 1012 |

macro-F1 (5 classes) 0.897, accuracy 0.904, multiclass MCC 0.860

macro-F1 95% CI 0.882-0.913; any-error F1 95% CI 0.901-0.925
Baseline (LaBSE threshold only) any-error F1 on test: 0.706

Span-level F1 on test (gated):

| type | side | precision | recall | F1 | gold chars |
|---|---|---|---|---|---|
| OMISSION | src (silver) | 0.707 | 0.482 | 0.573 | 7891 |
| MISTRANSLATION | tgt | 0.944 | 0.872 | 0.907 | 1141 |
| MISTRANSLATION | src (silver) | 0.769 | 0.542 | 0.636 | 153 |
| ENTITY_NUMBER | tgt | 0.777 | 0.932 | 0.848 | 761 |
| ENTITY_NUMBER | src (silver) | 0.180 | 0.665 | 0.283 | 203 |
| HALLUCINATION | tgt | 0.851 | 0.856 | 0.853 | 14582 |