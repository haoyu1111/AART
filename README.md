# AART-Net

This repository contains the AART components used for LGE-CMR LV, myocardium,
and scar segmentation. It is intentionally limited to the AART method:
comparison baselines and paper plotting scripts are not included.

## What is included

- `aart.AARTConv2d`: eight-branch anatomy-aligned radial-tangential convolution.
- `aart.AARTUNet2D`: AART-ResUNet-style four-class segmentation network.
- `aart.losses`: cross-entropy plus foreground Dice loss.
- `aart.data`: NPZ dataset adapter for 2D LGE-CMR slices.
- `aart.metrics`: patient-level Dice, IoU, HD95, and NSD utilities.
- `scripts/train_aart.py`: minimal training entry point.
- `scripts/infer_aart.py`: inference and patient-level evaluation.

## Final AART branch bank

The default operator uses the selected eight primitive responses:

1. `inner`
2. `blood_side_mismatch`
3. `radial_contrast`
4. `tangent_mean`
5. `tangent_range`
6. `tangent_persistence`
7. `blob_mean`
8. `blob_range`

The implementation computes only the requested branch families and uses a
batched `grid_sample` call for each radial, tangential, and blob sampling group.
Training-time diagnostic branch-energy bookkeeping from the research workspace is
not enabled in this clean version.

## Data format

The training and inference scripts expect a directory with split folders such as:

```text
data_root/
  train/
    CenterA/
      Case001_z000.npz
      ...
  val/
  test/
```

Each `.npz` file should contain:

- `image`: 2D LGE-CMR slice.
- `label`: integer mask with `0` background, `1` LV cavity, `2` myocardium, and `3` scar.

Images are min-max normalized to `[0, 1]` in the dataset adapter.

## Quick smoke test

```bash
python -m pip install -e .
python -m pytest tests
```

## Train

```bash
python scripts/train_aart.py \
  --data_root /path/to/npz_dataset \
  --output_dir outputs/aart \
  --train_split train \
  --val_split val \
  --train_centers CenterA \
  --eval_centers CenterA \
  --epochs 35 \
  --base_channels 32 \
  --augment \
  --amp
```

## Evaluate

```bash
python scripts/infer_aart.py \
  --data_root /path/to/npz_dataset \
  --split test \
  --centers CenterB,CenterC,CenterE,CenterF,CenterG \
  --checkpoint outputs/aart/best_val_fg_mean.pth \
  --output_dir outputs/aart_eval
```

## Open-source release checklist

- Confirm dataset licenses for EMIDEC and MyoPS++ before redistributing any data.
- Upload trained checkpoints only if all dataset and institutional permissions allow it.
- Keep baseline implementations in separate repositories or cite their original sources.
