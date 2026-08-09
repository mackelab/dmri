# Example data

`dmri predict` reads the four-file FSL layout described in
[Prediction](prediction.md#input). Public diffusion datasets rarely ship exactly
that layout, so this guide downloads two small open datasets, converts them, and
runs a prediction on each.

Both are chosen to be cheap: a full HCP-style subject is around 285 MB with 105
volumes, and the two here are 10 MB / 67 volumes and 21 MB / 64 volumes. Network
work scales with the number of measurements, so fewer volumes means a faster run.

| | `cfin_2shell` | `isbi_phantom` |
|---|---|---|
| source | CFIN multi-b, Aarhus | ISBI 2013 HARDI challenge |
| content | real brain | software phantom |
| matrix | 96 x 96 x 19 | 50 x 50 x 50 |
| volumes | 67 | 64 |
| shells | b = 0, 1000, 2000 | b = 0, 1500, 2500 |
| voxels in mask | 55,314 | 125,000 |
| download | 174 MB | 22 MB |

Prefer `cfin_2shell` for a realistic run: its shells match the b = 1000 / 2000
that the pretrained checkpoints were trained around. The phantom is useful for a
different reason — it has known ground-truth crossing fibres — but see the
caveat at the end before trusting its numbers.

## CFIN: real brain, two shells

The dataset is published as a single 496-volume acquisition: 33 gradient
directions repeated across 15 shells from b = 200 to b = 3000. Download it, then
keep only the two shells the checkpoints expect.

```bash
mkdir -p ~/dmri_data_small/cfin_raw && cd ~/dmri_data_small/cfin_raw

BASE=https://digital.lib.washington.edu/researchworks/bitstream/handle//1773/38488
N=__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4

wget -c $BASE/$N.nii
wget -c $BASE/$N.bval
wget -c $BASE/$N.bvec
```

The double slash in `handle//1773/38488` is not a typo — it is what the host
serves, and what dipy's own fetcher uses. A single slash returns 404.

Convert to the FSL layout, dropping the shells you do not need:

```bash
python scripts/make_fsl_dataset.py \
  --data $N.nii --bval $N.bval --bvec $N.bvec \
  --shells 1000 2000 \
  --out ~/dmri_data_small/cfin_2shell
```

```text
dropping 429 of 496 volumes
wrote /home/you/dmri_data_small/cfin_2shell
  data   (96, 96, 19, 67)  10 MB
  shells {0: 1, 1000: 33, 2000: 33}
  mask   55,314 of 175,104 voxels in brain
```

Then predict:

```bash
dmri predict ~/dmri_data_small/cfin_2shell \
  --output-subdir out_fast --quality fast --non-interactive
```

The run prints a viewer link at the end; open `out_fast/view_results.html` to
page through the maps. See [Output](prediction.md#output) for what is written.

## ISBI 2013 phantom

```bash
mkdir -p ~/dmri_data_small/isbi_raw && cd ~/dmri_data_small/isbi_raw

BASE=https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/38465

wget -c $BASE/phantom64.nii.gz
wget -c $BASE/phantom64.bval
wget -c $BASE/phantom64.bvec
```

```bash
python scripts/make_fsl_dataset.py \
  --data phantom64.nii.gz --bval phantom64.bval --bvec phantom64.bvec \
  --out ~/dmri_data_small/isbi_phantom
```

The phantom fills its whole volume with signal — its b = 0 is saturated in every
voxel — so the generated mask covers all 125,000 voxels. That is correct here,
not a masking failure, but it does mean there is no background to skip.

**This dataset needs `round_bvals` disabled.** The loader rounds b-values to the
nearest 1000 by default, which is what lets real acquisitions labelled `b = 5`
be recognised as b0. Applied to 1500 and 2500 it is destructive: 1500 rounds up
and 2500 rounds *down* (numpy uses banker's rounding, so `round(2.5) == 2`), and
both shells collapse onto a single nominal b = 2000.

```bash
dmri predict ~/dmri_data_small/isbi_phantom \
  --output-subdir out_fast --quality fast --non-interactive \
  --set evaluation.input.round_bvals=false
```

## What the converter does

`scripts/make_fsl_dataset.py` handles the differences between a published
dataset and the layout `dmri predict` reads:

- renames `.bval`/`.bvec` to `bvals`/`bvecs`, transposing the b-vectors to the
  3-by-N orientation FSL uses;
- snaps near-zero b-values to exactly `0`, because the normalisation step tests
  `bvals == 0`, and zeroes the direction on those volumes;
- keeps only `--shells` (b0 is always kept), reporting how many volumes it drops;
- generates `nodif_brain_mask.nii.gz` from the mean b0 with dipy's
  `median_otsu`, falling back to an Otsu threshold if dipy is unavailable.

`--tolerance` (default 60) sets how far a b-value may sit from a nominal shell
and still count as part of it.

## Compatibility caveats

Both datasets sit inside the b-value range the checkpoints were trained on
(0 to 4000, with typical shells at 1000, 2000 and 3000), but neither is a
substitute for checking that a checkpoint suits your acquisition — see the
warning in [Prediction](prediction.md#input).

Two specifics worth knowing:

- **Each has a single b0 volume**, against five in a typical HCP-style subject.
  The b0 normalisation is correspondingly noisier.
- **The phantom's 1500 / 2500 shells** are in range but are not among the
  typical values the training distribution favours, so results there are less
  representative than the CFIN run.
