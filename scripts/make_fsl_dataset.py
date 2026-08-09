"""Convert a downloaded dMRI dataset into the FSL layout `dmri predict` expects.

Writes `data.nii.gz`, `bvals`, `bvecs` and `nodif_brain_mask.nii.gz`. Public
datasets rarely ship this layout: b-value files are named `.bval`/`.bvec`, there
is often no brain mask, and multi-shell acquisitions carry far more volumes than
a prediction run needs. `--shells` keeps a subset, which is the main cost lever
-- the forward pass is linear in the number of measurements.

See docs/guides/example-data.md for worked examples.
"""

import argparse
import pathlib

import nibabel as nb
import numpy as np


def brain_mask(b0):
    """A binary mask from the b0 volume.

    Uses dipy's median_otsu where available (what FSL's `bet` approximates for
    this purpose); falls back to an Otsu threshold on a smoothed b0 so the
    script does not hard-depend on it.
    """
    try:
        from dipy.segment.mask import median_otsu

        _, mask = median_otsu(b0, median_radius=2, numpass=1)
        return mask.astype(np.uint8)
    except Exception:
        from scipy.ndimage import binary_fill_holes, gaussian_filter
        from skimage.filters import threshold_otsu

        smooth = gaussian_filter(b0.astype(np.float32), 1.0)
        return binary_fill_holes(smooth > threshold_otsu(smooth)).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--bval", required=True)
    p.add_argument("--bvec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--shells",
        type=float,
        nargs="*",
        default=None,
        help="Keep only these b-values (b0 always kept), e.g. --shells 1000 2000",
    )
    p.add_argument("--tolerance", type=float, default=60.0)
    args = p.parse_args()

    img = nb.load(args.data)
    bvals = np.loadtxt(args.bval).ravel()
    bvecs = np.loadtxt(args.bvec)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    if bvecs.shape[1] != bvals.size:
        raise SystemExit(f"bvecs {bvecs.shape} do not match {bvals.size} bvals")

    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.shape[-1] != bvals.size:
        raise SystemExit(f"data has {data.shape[-1]} volumes, bvals has {bvals.size}")

    # b0 is whatever sits below the shell tolerance, so acquisitions labelled
    # b=5 rather than b=0 are still treated as unweighted.
    is_b0 = bvals <= args.tolerance
    keep = is_b0.copy()
    if args.shells:
        for shell in args.shells:
            keep |= np.abs(bvals - shell) <= args.tolerance
        if not keep.all():
            print(f"dropping {int((~keep).sum())} of {bvals.size} volumes")
    else:
        keep[:] = True

    data, bvals, bvecs = data[..., keep], bvals[keep], bvecs[:, keep]
    # The loader tests `bvals == 0` exactly; snap near-zero labels to 0.
    bvals[bvals <= args.tolerance] = 0.0
    # And zero the direction on unweighted volumes, as FSL does.
    bvecs[:, bvals == 0] = 0.0

    b0 = data[..., bvals == 0].mean(axis=-1)
    mask = brain_mask(b0)

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    affine, header = img.affine, img.header
    nb.Nifti1Image(data, affine, header).to_filename(out / "data.nii.gz")
    nb.Nifti1Image(mask, affine, header).to_filename(out / "nodif_brain_mask.nii.gz")
    np.savetxt(out / "bvals", bvals[None, :], fmt="%g")
    np.savetxt(out / "bvecs", bvecs, fmt="%.6f")

    shells, counts = np.unique(np.round(bvals / 100) * 100, return_counts=True)
    print(f"wrote {out}")
    print(f"  data   {data.shape}  {(out / 'data.nii.gz').stat().st_size / 1e6:.0f} MB")
    summary = dict(zip(shells.astype(int).tolist(), counts.tolist(), strict=True))
    print(f"  shells {summary}")
    print(f"  mask   {int(mask.sum()):,} of {mask.size:,} voxels in brain")


if __name__ == "__main__":
    main()
