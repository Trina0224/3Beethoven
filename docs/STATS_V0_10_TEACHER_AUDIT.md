# v0.10 teacher corpus audit

All 112 accepted records (96 training, 16 validation) were independently read. Exact rational validation checks every equation stage against the gold answer, not only the last line. Five mathematically valid but pedagogically poor chains were repaired: v10_train_type_i_030, v10_train_type_i_032, v10_train_type_ii_051, v10_train_type_ii_052, v10_train_type_ii_087. Their replacement stages were independently read and verified. Original solutions and cache tags remain in each repaired record and in the Kaggle pre-audit snapshot.

The final corpus contains 75 reference-conditioned records (62 training, 13 validation), including 37 verbatim-reference format repairs (29 training, 8 validation). These are not independent teacher solutions. Previously audited v0.9 rules are reused. Adjacent whitespace-equivalent duplicate stages are removed only when constructing training targets; raw teacher responses remain unchanged.

Approved records SHA-256: `aced5b6372725288a331975deb4a6f780ca416a7e00ed1043814418fef816f79`.

Teacher generation and repairs: 221 calls, 221 responses, reported cost $0.02018775, no responses missing cost. No v0.10 student training or final test has run at this checkpoint. The frozen protocol remains in STATS_V0_10_PROTOCOL.md. Installation of missing bitsandbytes==0.50.2 was rejected by automatic approval review because action-time installation confirmation is required; Secrets worked during generation. Preparation can be completed without installing software.
