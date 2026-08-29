# Design notes

Decisions and gotchas in this codebase that are not obvious from reading it.

---

## 1. Build the model before the optimizer

`build_segmentation_model()` returns a fully-initialised model, and the
optimizer must be created from its parameters afterwards.

This ordering is load-bearing. If you build a model, bind an optimizer to it,
and only then replace the model — for example by conditionally loading
pretrained weights into a fresh instance — the training loop forwards through
the new model while `optimizer.step()` updates the parameters of the discarded
one. Gradients flow, the loss even moves, and the evaluated network never
learns anything. It fails silently, which is what makes it dangerous.

The current structure makes that impossible: there is exactly one model object,
constructed once, before the optimizer exists.

---

## 2. The train/test split depends on sorted file listings

`build_subject_index()` sorts its results before the seeded shuffle.

This is not cosmetic. `Path.rglob()` returns entries in filesystem-dependent
order, so an unsorted index makes `random.seed(42)` produce *different*
partitions on different machines despite looking deterministic. Sorting is what
makes the seeds mean anything.

⚠️ **Consequence for comparisons.** Any checkpoint trained before sorting was
introduced was evaluated on a different partition. Never compare an old run
against a new one — re-run both arms of any foundation-vs-scratch comparison
under the same code.

The split is at **patient** level, never scan level: the same patient's anatomy
appearing in both train and test would inflate apparent accuracy. Test subjects
are drawn only from single-scan patients so that multi-scan patients, which
would otherwise dominate the held-out set, all land in training.

---

## 3. MONAI version compatibility

`SwinUNETR` took a required `img_size` constructor argument in older MONAI, and
newer versions (>= 1.5) removed it and infer spatial size at runtime. Passing it
unconditionally raises `TypeError` on current MONAI.

`build_swin_unetr()` inspects the constructor signature and passes `img_size`
only when that version accepts it, so the same code works across versions
without pinning one.

---

## 4. Weight transfer fails loudly

`load_foundation_weights()` must call `load_state_dict(..., strict=False)`,
because the pretraining and fine-tuning heads legitimately differ.

The hazard is that `strict=False` will happily transfer **nothing at all** if the
checkpoint's key prefix doesn't match what the model expects, leaving you
convinced you are fine-tuning a pretrained model while actually training from
random weights. So the loader checks how many tensors matched and raises if the
answer is zero, rather than proceeding quietly.

A healthy transfer for SwinUNETR reports `0 missing / 0 unexpected`.

For `--arch nnunet`, keys containing `seg_outputs` are deliberately skipped: the
pretraining head reconstructs two image channels and the segmentation head
predicts two classes. Same tensor shape, completely different meaning — copying
it across would be worse than random initialisation.

---

## 5. Memory behaviour

- **`--batch-size`** controls training. **`--sw-batch-size`** separately controls
  sliding-window inference during evaluation. They fail independently: the
  evaluation pass runs on full-size volumes, which are much larger than the
  96×128×128 training crops, so evaluation can OOM on a run that trains fine.
- **MPS (Apple Silicon)** does not reliably free memory between iterations for
  these 3D models. Without the explicit `devices.empty_cache()` calls it creeps
  past the memory watermark and OOMs partway through an epoch. Observed on a
  16 GB machine.
- **AMP** is enabled only on CUDA. `torch.autocast("cuda")` raises on other
  devices, and MPS has no working autocast for these operations.

---

## 6. `--arch nnunet` is the network, not the pipeline

This uses `PlainConvUNet` from `dynamic_network_architectures` with hand-written
stage parameters replicating nnU-Net v2's `3d_fullres` heuristics for a
96×128×128 input: stage 0 at full resolution, then five 2× downsamples to a
3×4×4 bottleneck.

That is nnU-Net's *network*, but **not** the nnU-Net *pipeline*. Real nnU-Net
fingerprints the dataset and plans patch size, spacing, stage count and batch
size itself, and that auto-configuration is the main reason nnU-Net is strong.
It is bypassed here.

This is appropriate for a like-for-like CNN comparison against SwinUNETR at a
fixed patch size. If you need a genuine nnU-Net baseline — for example to
compare against published autoPET results — install `nnunetv2` from PyPI and use
its own CLI (`nnUNetv2_plan_and_preprocess` → `nnUNetv2_train`) instead of this
code path.

---

## 7. Values not to change casually

These are load-bearing for comparability against published numbers:

- **Split arithmetic:** 200 test patients, 400-scan multi-scan cap, 700-scan
  pool, seeds 42 and 1024.
- **Preprocessing:** 3×2×2 mm target spacing, −500 HU body threshold, 3-voxel
  crop padding, per-modality z-scoring.
- **MAE:** 12×16×16 mask blocks, 0.5 mask ratio, zero imputation, 1.0/0.2
  masked/visible loss weighting.
- **Optimisation:** AdamW, 1e-4 for transformers and 5e-4 for nnU-Net, weight
  decay 1e-5, cosine annealing to 1e-6, gradient clipping at 1.0.
- **Loss and metrics:** `DiceCELoss(to_onehot_y=True, softmax=True)`; Dice and
  HD95 with `include_background=False`.
- **Inference:** sliding window at ROI 96×128×128, 0.5 overlap.

The zero-imputation choice in the MAE depends on the z-scoring above: after
z-scoring, 0.0 *is* the dataset mean, so masked voxels are filled with the
statistically expected value rather than an arbitrary constant. Change the
normalisation and that reasoning no longer holds.

---

## 8. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PETCT_AUTOPET_ROOT` | `./sample_data` | labelled AutoPET data |
| `PETCT_PRETRAIN_ROOT` | `./PETCTfoundation` | unlabelled `.npy` corpus |
| `PETCT_WEIGHTS_DIR` | `./weights` | pretrained checkpoints |
| `PETCT_OUTPUT_DIR` | `./runs` | fine-tuning outputs |
