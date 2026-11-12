# Errata

Corrections to statements in earlier reports. The earlier files are left
unchanged because they are locked or committed records. Each entry gives the
statement, the correct value and its source.

## 1. Phase XIX lock: number of certified agent updates

- **Statement.** `reports/phase19_lock_psma.md` (b8351d4), line 12: "Certified
  on B (δ 0.05, α 0.05, n 126): 40 of 88 candidates."
- **Correct value: 22 of 88.**
- **Source.** `D:\data\reports\phase19_psma.json`: `n_certified = 22`.
  - Under Bonferroni α/8 = 0.00625 with n = 126 and δ = 0.05, only k = 0
    rejects: p = 0.95^126 = 0.0016. For k = 1, p = 0.012.
  - The 22 certified candidates are exactly the k = 0 candidates that come
    before any non-rejection in their chain.
  - 48 candidates have an uncorrected p < 0.05, so "40" matches neither count.
    It was a transcription error in the descriptive text.
- **Impact: none on any decision.**
  - The deployed configuration was computed by `phase19_selfimprove.py` from
    the same certification. It was reproduced unchanged in Phase XX:
    `new_cal`, 0.97, `agree+solo`.
  - Every C result is unaffected.
  - The paper reports 22.
