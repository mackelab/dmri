"""
Generate lightweight SVG figures used in the docs.

This script avoids external plotting dependencies by emitting simple SVGs for:
    - Tutorial plots (sim_ball.svg, sim_stick.svg, sim_multi.svg)
    - Model gallery plots (gallery_ball.svg, gallery_stick.svg, gallery_zeppelin.svg, gallery_dti.svg)

Run it before building the docs (the GitHub Pages workflow calls it).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _make_svg_plot(
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    fname: Path,
    title: str,
    xlabel: str = "b-value [s/mm²]",
    ylabel: str = "Signal",
    width: int = 720,
    height: int = 420,
    padding: int = 50,
) -> None:
    """Minimal SVG plotting helper to keep docs builds dependency-light."""
    x = np.asarray(x)
    xmin, xmax = float(x.min()), float(x.max())
    ymin = min(np.min(y) for _, y in series)
    ymax = max(np.max(y) for _, y in series)
    if math.isclose(ymin, ymax):
        ymin -= 0.1
        ymax += 0.1

    def sx(v):
        return (
            padding
            + (v - xmin) / (xmax - xmin) * (width - 2 * padding)
            if xmax > xmin
            else padding
        )

    def sy(v):
        return (
            height
            - padding
            - (v - ymin) / (ymax - ymin) * (height - 2 * padding)
            if ymax > ymin
            else height / 2
        )

    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" />',
        f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#e0e0e0" />',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#e0e0e0" />',
    ]
    for i, (label, y) in enumerate(series):
        pts = " ".join(f"{sx(a):.2f},{sy(b):.2f}" for a, b in zip(x, y))
        color = colors[i % len(colors)]
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}" />'
        )
        ly = padding - 15 - i * 18
        lx = padding
        elements.append(
            f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{color}" />'
        )
        elements.append(
            f'<text x="{lx+18}" y="{ly+11}" font-size="13" fill="#e0e0e0">{label}</text>'
        )
    elements.append(
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-weight="bold" fill="#e0e0e0">{title}</text>'
    )
    elements.append(
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13" fill="#e0e0e0">{xlabel}</text>'
    )
    elements.append(
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="13" fill="#e0e0e0" transform="rotate(-90 20,{height/2})">{ylabel}</text>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        + "".join(elements)
        + "</svg>"
    )
    fname.write_text(svg)


def _tutorial_plots(out_dir: Path) -> None:
    bvals = np.linspace(0, 3000, 100)
    lam_ball = 0.0018
    lam_par = 0.0015

    ball_signal = np.exp(-bvals * lam_ball)
    signal_aligned = np.exp(-bvals * lam_par * (0.8**2))
    signal_tilt = np.exp(-bvals * lam_par * (0.5**2))
    mixed = 0.5 * ball_signal + 0.5 * signal_aligned

    _make_svg_plot(
        bvals,
        [("Ball signal", ball_signal)],
        out_dir / "sim_ball.svg",
        title="Ball: isotropic decay",
    )
    _make_svg_plot(
        bvals,
        [("Aligned with g", signal_aligned), ("Tilted 45°", signal_tilt)],
        out_dir / "sim_stick.svg",
        title="Stick orientation effect",
    )
    _make_svg_plot(
        bvals,
        [("Ball+Stick", mixed)],
        out_dir / "sim_multi.svg",
        title="Multi-compartment mixture",
    )


def _gallery_plots(out_dir: Path) -> None:
    bvals = np.linspace(0, 3000, 80)
    ball = np.exp(-bvals * 0.0018)
    stick = np.exp(-bvals * 0.0015 * (0.7**2))
    lam_par, lam_perp, cos = 0.0015, 0.0005, 0.6
    zeppelin = np.exp(-bvals * (lam_perp + (lam_par - lam_perp) * (cos**2)))

    # simple DTI average over random directions
    bvecs = np.random.randn(30, 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    D = np.diag([0.0015, 0.0007, 0.0003])
    proj = np.einsum("bi,ij,bj->b", bvecs, D, bvecs)
    dti = np.exp(-bvals[None, :] * proj[:, None]).mean(axis=0)

    _make_svg_plot(bvals, [("Ball", ball)], out_dir / "gallery_ball.svg", title="Ball")
    _make_svg_plot(
        bvals, [("Stick", stick)], out_dir / "gallery_stick.svg", title="Stick"
    )
    _make_svg_plot(
        bvals,
        [("Zeppelin", zeppelin)],
        out_dir / "gallery_zeppelin.svg",
        title="Zeppelin",
    )
    _make_svg_plot(
        bvals, [("DTI (avg)", dti)], out_dir / "gallery_dti.svg", title="DTI"
    )


def main() -> None:
    out_dir = Path(__file__).parent / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    _tutorial_plots(out_dir)
    _gallery_plots(out_dir)
    print(f"Generated SVG figures under {out_dir}")


if __name__ == "__main__":
    main()
