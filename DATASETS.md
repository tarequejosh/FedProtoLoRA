# Datasets

All five datasets are public. They are downloaded from Kaggle by `scripts/download_data.py` into
`data/raw/<client>/`. The images are not redistributed in this repository, and each dataset remains subject to the
terms given on its source page.

| Client | Modality | Dataset | Kaggle slug | Classes | Grouping unit |
|---|---|---|---|---|---|
| `derm` | Dermoscopy | HAM10000 (Tschandl et al., 2018) | `kmader/skin-cancer-mnist-ham10000` | 7 | lesion (`lesion_id`) |
| `oct` | Retinal OCT | Kermany OCT (Kermany et al., 2018) | `paultimothymooney/kermany2018` | 4 | patient (file-name prefix) |
| `histo` | Breast histopathology | BreaKHis (Spanhol et al., 2016) | `ambarish/breakhis` | 2 | patient |
| `cxr` | Paediatric chest X-ray | Kermany CXR (Kermany et al., 2018) | `paultimothymooney/chest-xray-pneumonia` | 2 | patient (file-name prefix) |
| `brain` | Brain MRI | Brain Tumor MRI (Nickparvar, 2021) | `masoudnickparvar/brain-tumor-mri-dataset` | 4 | near-duplicate group |

## After curation (seed 42)

| Client | Images | Groups | Train / val / test |
|---|---:|---:|---|
| Dermoscopy | 9,884 | 7,399 | 5,999 / 1,483 / 1,484 |
| Retinal OCT | 72,472 | 4,611 | 6,001 / 1,501 / 10,872 |
| Histopathology | 7,909 | 70 | 5,794 / 812 / 1,303 |
| Chest X-ray | 5,693 | 3,027 | 3,983 / 855 / 855 |
| Brain MRI | 7,013 | 4,755 | 4,907 / 1,053 / 1,053 |

Training and validation sets are capped at 6,000 and 1,500 images per client (label-stratified subsample, identical
for every method); this affects OCT and dermoscopy. Test sets are never subsampled.

## Patient grouping

Every image belongs to a group that never crosses splits.

- **HAM10000:** `lesion_id` from the metadata file (one lesion has several images).
- **OCT:** patient ID from the file name, e.g. `CNV-1016042-1` → `1016042`.
- **Chest X-ray:** patient ID from the file name, e.g. `person1946_bacteria_4874` → `person1946`. Normal images use
  `IM-xxxx` identifiers (`IM-0115-0001` → `IM-0115`, `NORMAL2-IM-1427-0001` → `NORMAL2-IM-1427`).
- **BreaKHis:** year and patient number (`SOB_B_A-14-22549AB-100-001` → `14-22549`). The tumour-subtype code and slide
  suffix are dropped because one patient (14-13412) appears under two subtypes and slides 22549AB/CD/G belong to one patient.
- **Brain MRI:** no patient identifiers exist. Each image starts as its own group, and near-duplicates are merged
  into one group (see below), so that they always share a split.

## De-duplication

1. **Exact duplicates (MD5).** The first copy is kept. If copies carry different labels, all copies are removed
   (466 OCT images).
2. **Perceptual near-duplicates** (64-bit pHash, Hamming distance ≤ 4).
   - Dermoscopy, OCT, histopathology, chest X-ray: an image whose near-duplicate belongs to a *different* patient is
     removed. Near-duplicates of the same patient are already kept together by the grouping.
   - Brain MRI: near-duplicates are merged into one group (union-find).

Every removed or merged image is listed in `data/dedup_reports/<client>_dedup_report.csv` with the reason, and
`orig_group_id` in the manifests keeps the group before merging.

## Splits

Splits are 70/15/15 and group-disjoint, made with a 20-fold `StratifiedGroupKFold` whose folds are assigned 14/3/3
to train/val/test. BreaKHis has only 70 patients, so its splits are stratified at patient level (about ten test
patients per seed). Manifests for seeds 42, 101, 2024, 7 and 1234 are in `data/splits/<client>_<seed>.csv`.

`tests/test_no_leakage.py` checks every manifest: no group, MD5 hash or near-duplicate pHash (Hamming ≤ 4) may occur
in more than one split.

## Setting S2

Each modality except histopathology is split into three institutions by whole patient groups, with Dirichlet(α)
label proportions (α ∈ {0.1, 0.5}). An institution's training, validation and test data follow the same skewed
distribution. Histopathology is excluded because its 70 patients cannot form three institutions with usable
validation and test sets. The partition is computed at run time from the split manifests and the seed
(`src/fedmedlora/data/partition.py`).
