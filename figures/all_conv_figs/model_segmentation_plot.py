from pathlib import Path
import colorsys
import hashlib
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import nibabel as nib
import numpy as np
import pandas as pd

def _model_summary_path(p_mask, data=0, data_path=None):
    p_mask_str = str(p_mask).replace(".", "")
    base_dir = Path(data_path) if data_path is not None else Path("outputs")
    return base_dir / f"data{data}_p{p_mask_str}" / f"preselected_models_summary_{p_mask}.csv"


def _load_model_summary(p_mask, data=0, data_path=None):
    return pd.read_csv(_model_summary_path(p_mask, data=data, data_path=data_path))


def get_model_freq(p_mask, data=0, return_models=False, data_path=None):
    df = _load_model_summary(p_mask, data=data, data_path=data_path)
    model_freq = df["Count"].values / df["Count"].sum()
    if return_models:
        return df["Model"].tolist(), model_freq
    return model_freq

def plot_model_freqs(
    p_values=None,
    fig_size=(6, 3),
    show=True,
    legend=False,
    output_png=None,
    output_svg=None,
    preselected_base_colors=None,
    preselected_factors=None,
    preselected_hls_factors=None,
    data=0,
    data_path=None,
    group_similar=False,
    group_base_colors=None,
    ball_color="lightgray",
):
    if p_values is None:
        p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0]
    models, first = get_model_freq(p_values[0], data=data, return_models=True, data_path=data_path)
    model_probs = [first]
    for p in p_values[1:]:
        model_probs.append(get_model_freq(p, data=data, data_path=data_path))
    model_probs = np.stack(model_probs, axis=0)
    if model_probs.shape[1] != len(models):
        raise ValueError("Model counts differ across p values.")

    preselected = _is_preselected(models)
    if group_similar:
        colors = _grouped_class_colors(
            models,
            base_colors=group_base_colors,
            ball_color=ball_color,
        )
        if preselected:
            labels = [_preselected_name(m) for m in models]
        else:
            labels = [_shorthand(m) for m in models]
    elif preselected:
        colors = _preselected_class_colors(
            models,
            ball_color=ball_color,
            preselected_base_colors=preselected_base_colors,
            preselected_factors=preselected_factors,
            preselected_hls_factors=preselected_hls_factors,
        )
        labels = [_preselected_name(m) for m in models]
    else:
        colors = _apply_special_class_colors(models, _make_distinct_palette(len(models)))
        labels = [_shorthand(m) for m in models]

    fig, ax = plt.subplots(figsize=fig_size)
    for idx, label in enumerate(labels):
        ax.plot(p_values, model_probs[:, idx], color=colors[idx], label=label, linewidth=1.6)

    ax.set_xlabel(r"$\lambda$", labelpad=-5)
    ax.set_ylabel("Freq.")
    ax.set_xlim(min(p_values), max(p_values))
    ax.set_ylim(0.0, model_probs.max() * 1.05)
    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=4,
            fontsize=8,
            frameon=False,
        )

    if output_png:
        fig.savefig(output_png, dpi=300)
    if output_svg:
        fig.savefig(output_svg)
    if show:
        plt.show()
    return fig, ax

def _load_probs(probs):
    if isinstance(probs, (str, Path)):
        return nib.load(str(probs)).get_fdata()
    return np.asarray(probs)


def _load_summary(summary):
    if isinstance(summary, (str, Path)):
        return pd.read_csv(summary)
    return summary


def _is_preselected(models):
    allowed = {"Stick", "Zeppelin", "Dti"}
    for model in models:
        toks = model.split(">")
        if toks[0] != "Ball":
            return False
        if len(toks) == 1:
            continue
        tail = toks[1:]
        if any(t != tail[0] for t in tail):
            return False
        if tail[0] not in allowed:
            return False
    return True


def _vary_lightness(base_color, factors):
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    out = []
    for f in factors:
        l2 = np.clip(l * f, 0, 1)
        out.append(colorsys.hls_to_rgb(h, l2, s))
    return out


def _vary_hls(base_color, hls_factors):
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    out = []
    for dh, lf, sf in hls_factors:
        h2 = (h + dh) % 1.0
        l2 = np.clip(l * lf, 0, 1)
        s2 = np.clip(s * sf, 0, 1)
        out.append(colorsys.hls_to_rgb(h2, l2, s2))
    return out


def _preselected_class_colors(
    class_paths,
    ball_color="lightgray",
    preselected_base_colors=None,
    preselected_factors=None,
    preselected_hls_factors=None,
):
    hls_factors = preselected_hls_factors or [
        (-0.04, 1.05, 1.20),
        (0.00, 0.92, 1.00),
        (0.04, 0.82, 0.90),
    ]
    factors = preselected_factors or [1.10, 1.00, 0.90]
    base_colors = {"Stick": "C0", "Zeppelin": "C2", "Dti": "C3"}
    if preselected_base_colors:
        base_colors.update(preselected_base_colors)
    if preselected_hls_factors or preselected_factors is None:
        s_cols = _vary_hls(base_colors["Stick"], hls_factors)
        z_cols = _vary_hls(base_colors["Zeppelin"], hls_factors)
        t_cols = _vary_hls(base_colors["Dti"], hls_factors)
    else:
        s_cols = _vary_lightness(base_colors["Stick"], factors)
        z_cols = _vary_lightness(base_colors["Zeppelin"], factors)
        t_cols = _vary_lightness(base_colors["Dti"], factors)
    class_colors = []
    for model in class_paths:
        if model == "Ball":
            class_colors.append(ball_color)
            continue
        special = _special_color_for_model(model)
        if special is not None:
            class_colors.append(special)
            continue
        toks = model.split(">")
        comp = toks[1]
        count = len(toks) - 1
        idx = min(max(count - 1, 0), len(factors) - 1)
        if comp == "Stick":
            class_colors.append(s_cols[idx])
        elif comp == "Zeppelin":
            class_colors.append(z_cols[idx])
        elif comp == "Dti":
            class_colors.append(t_cols[idx])
        else:
            class_colors.append(ball_color)
    return class_colors


def _make_distinct_palette(n):
    cm = plt.get_cmap("tab10")
    if n <= cm.N:
        return [cm(i) for i in range(n)]
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    cols = []
    for cmi in cmaps:
        cols.extend([cmi(i) for i in range(cmi.N)])
    if n > len(cols):
        raise ValueError(f"Need {n} colors, but palette provides only {len(cols)}.")
    return cols[:n]


def _preselected_name(model):
    if model == "Ball":
        return "Ball"
    toks = model.split(">")
    comp = toks[1]
    count = len(toks) - 1
    comp_abbr = {"Stick": "S", "Zeppelin": "Z", "Dti": "T"}
    return f"B{count}{comp_abbr.get(comp, comp[:1].upper())}"


_special_token_abbr = {
    "NoddiW": r"No$_{\mathcal{W}}$",
    "NoddiB": r"No$_{\mathcal{B}}$",
    "SandiW": r"Sa$_W$",
    "SandiB": r"Sa$_B$",
    "WatsonStick": r"S$_{\mathcal{W}}$",
    "BinghamStick": r"S$_{\mathcal{B}}$",
    "WatsonZeppelin": r"Z$_{\mathcal{W}}$",
    "BinghamZeppelin": r"Z$_{\mathcal{B}}$",
}

_special_token_colors = {
    "NoddiW": "#CC79A7",
    "NoddiB": "#E69F00",
    "SandiW": "#009E73",
    "SandiB": "#56B4E9",
    "WatsonStick": "#0072B2",
    "BinghamStick": "#D55E00",
}

_abbr_token = {
    "Ball": "B",
    "Zeppelin": "Z",
    "Dti": "T",
    "Stick": "S",
    **_special_token_abbr,
}


def _shorthand(model):
    toks = model.split(">")
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        j = i + 1
        while j < len(toks) and toks[j] == t:
            j += 1
        count = j - i
        base = _abbr_token.get(t, t[:1].upper())
        out.append(f"{count}{base}" if count > 1 else base)
        i = j
    return "".join(out)


def _strip_counts(label):
    return re.sub(r"\d+", "", label)


def _group_key(model):
    return _strip_counts(_shorthand(model))


def _blend_colors(color_a, color_b, weight=0.7):
    return tuple(
        (color_a[idx] * weight) + (color_b[idx] * (1.0 - weight)) for idx in range(3)
    )


def _average_colors(colors):
    if not colors:
        raise ValueError("colors must contain at least one color.")
    if len(colors) == 1:
        return colors[0]
    return tuple(sum(c[idx] for c in colors) / len(colors) for idx in range(3))


def _special_color_for_model(model):
    tokens = model.split(">")
    specials = [t for t in tokens if t in _special_token_colors]
    if not specials:
        return None
    base = _average_colors(
        [mcolors.to_rgb(_special_token_colors[token]) for token in specials]
    )
    if len(tokens) == 1 and len(specials) == 1:
        return base
    return _blend_colors(base, _stable_group_color(_group_key(model)), weight=0.7)


def _apply_special_class_colors(class_paths, class_colors):
    out = list(class_colors)
    for idx, model in enumerate(class_paths):
        special = _special_color_for_model(model)
        if special is not None:
            out[idx] = special
    return out


def _stable_group_color(group_key, saturation=(0.55, 0.75), lightness=(0.45, 0.65)):
    digest = hashlib.md5(group_key.encode("utf-8")).hexdigest()
    h = int(digest[:8], 16) / 2**32
    s = saturation[0] + (int(digest[8:16], 16) / 2**32) * (saturation[1] - saturation[0])
    l = lightness[0] + (int(digest[16:24], 16) / 2**32) * (lightness[1] - lightness[0])
    return colorsys.hls_to_rgb(h, l, s)


def _shade_variant(
    base_color,
    idx,
    count,
    lightness_span=0.32,
    saturation_span=0.26,
    hue_span=0.0,
):
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if count <= 1:
        return colorsys.hls_to_rgb(h, l, s)
    offset = (idx / (count - 1)) - 0.5
    h2 = (h + offset * hue_span) % 1.0
    l2 = np.clip(l + offset * lightness_span, 0, 1)
    s2 = np.clip(s * (1.0 + offset * saturation_span), 0, 1)
    return colorsys.hls_to_rgb(h2, l2, s2)


def _natural_sort_key(label):
    parts = re.split(r"(\d+)", label)
    return [int(p) if p.isdigit() else p for p in parts]


def _grouped_class_colors(class_paths, base_colors=None, ball_color="lightgray"):
    base_colors = dict(base_colors or {})
    group_keys = [_group_key(m) for m in class_paths]
    label_keys = [_shorthand(m) for m in class_paths]
    label_complexity = {}
    for model, label in zip(class_paths, label_keys):
        label_complexity.setdefault(label, len(model.split(">")))
    group_tokens = {}
    for key, model in zip(group_keys, class_paths):
        group_tokens.setdefault(key, set()).update(model.split(">"))
    for key, model in zip(group_keys, class_paths):
        if key in base_colors:
            continue
        special = _special_color_for_model(model)
        if special is not None:
            base_colors[key] = special
    base_by_key = {}
    for key in group_keys:
        if key in base_by_key:
            continue
        if key in base_colors:
            base_by_key[key] = base_colors[key]
        else:
            base_by_key[key] = _stable_group_color(key)
    group_labels = {}
    for key, label in zip(group_keys, label_keys):
        group_labels.setdefault(key, set()).add(label)
    group_label_index = {}
    for key, labels in group_labels.items():
        ordered = sorted(
            labels, key=lambda label: (label_complexity.get(label, 0), _natural_sort_key(label))
        )
        group_label_index[key] = {label: idx for idx, label in enumerate(ordered)}
    colors = []
    for key, label in zip(group_keys, label_keys):
        if label == "B":
            colors.append(ball_color)
            continue
        shade_kwargs = {}
        tokens = group_tokens.get(key, set())
        if "BinghamStick" in tokens and "Dti" in tokens:
            shade_kwargs = {
                "lightness_span": 0.48,
                "saturation_span": 0.4,
                "hue_span": 0.12,
            }
        colors.append(
            _shade_variant(
                base_by_key[key],
                group_label_index[key][label],
                len(group_label_index[key]),
                **shade_kwargs,
            )
        )
    return colors


def _prepare_segmentation_inputs(
    probs,
    summary,
    class_colors=None,
    cmap=None,
    names=None,
    background_color="white",
    ball_color="lightgray",
    preselected_base_colors=None,
    preselected_factors=None,
    preselected_hls_factors=None,
    show_percent=True,
    group_similar=None,
    group_base_colors=None,
):
    probs_arr = _load_probs(probs)
    summary_df = _load_summary(summary)
    if "Model" not in summary_df.columns:
        raise ValueError("summary must have a 'Model' column.")
    class_paths = summary_df["Model"].tolist()
    if probs_arr.shape[-1] != len(class_paths):
        raise ValueError(
            f"probs has {probs_arr.shape[-1]} channels but summary has {len(class_paths)} models."
        )

    preselected = _is_preselected(class_paths)
    if names is None:
        if preselected:
            names = {i + 1: _preselected_name(m) for i, m in enumerate(class_paths)}
        else:
            names = {i + 1: _shorthand(m) for i, m in enumerate(class_paths)}
    elif isinstance(names, (list, tuple)):
        if len(names) != len(class_paths):
            raise ValueError("names list must match number of classes.")
        names = {i + 1: names[i] for i in range(len(class_paths))}
    elif not isinstance(names, dict):
        raise ValueError("names must be a list, tuple, or dict mapping label -> name.")

    if show_percent and "Count" in summary_df.columns:
        counts = pd.to_numeric(summary_df["Count"], errors="coerce").to_numpy()
        total = np.nansum(counts)
        if total > 0:
            percents = counts / total * 100.0
            names = {
                i + 1: f"{names[i + 1]} ({percents[i]:.1f}%)"
                for i in range(len(class_paths))
            }

    if isinstance(cmap, str) and cmap.lower() in {"top10_grouped", "top10_group"}:
        group_similar = True
        cmap = None

    if group_similar is None:
        group_similar = isinstance(summary, (str, Path)) and "top10" in str(summary).lower()

    if class_colors is not None and cmap is not None:
        raise ValueError("Provide either class_colors or cmap, not both.")

    if group_similar and (class_colors is not None or cmap is not None):
        raise ValueError("group_similar cannot be combined with class_colors or cmap.")

    if class_colors is not None:
        if isinstance(class_colors, dict):
            class_colors = [class_colors[i] for i in range(1, len(class_paths) + 1)]
        if len(class_colors) != len(class_paths):
            raise ValueError("class_colors must have one entry per class.")
        color_list = [background_color] + list(class_colors)
        cmap = mcolors.ListedColormap(color_list, name="seg_custom_bg0")
    elif cmap is not None:
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        if isinstance(cmap_obj, mcolors.ListedColormap):
            colors = list(cmap_obj.colors)
            if len(colors) == len(class_paths) + 1:
                color_list = colors
            elif len(colors) >= len(class_paths):
                color_list = [background_color] + colors[: len(class_paths)]
            else:
                raise ValueError("ListedColormap has fewer colors than classes.")
        else:
            samples = [
                cmap_obj(i / max(len(class_paths) - 1, 1)) for i in range(len(class_paths))
            ]
            color_list = [background_color] + samples
        cmap = mcolors.ListedColormap(color_list, name="seg_custom_bg0")
    elif group_similar:
        class_colors = _grouped_class_colors(
            class_paths,
            base_colors=group_base_colors,
            ball_color=ball_color,
        )
        color_list = [background_color] + class_colors
        cmap = mcolors.ListedColormap(color_list, name="seg_grouped_top10_bg0")

    if preselected and cmap is None:
        class_colors = _preselected_class_colors(
            class_paths,
            ball_color=ball_color,
            preselected_base_colors=preselected_base_colors,
            preselected_factors=preselected_factors,
            preselected_hls_factors=preselected_hls_factors,
        )
        color_list = [background_color] + class_colors
        cmap = mcolors.ListedColormap(color_list, name="seg_grouped_bg0_white")
    elif cmap is None:
        palette = _apply_special_class_colors(class_paths, _make_distinct_palette(len(class_paths)))
        color_list = [background_color] + palette
        cmap = mcolors.ListedColormap(color_list, name="seg_distinct_bg0white")

    c = len(class_paths)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, c + 1.5, 1), cmap.N)

    def labels_from_probs(probs_in):
        lab = probs_in.argmax(axis=-1).astype(np.int32) + 1
        bg = probs_in.sum(axis=-1) == 0.0
        return np.where(bg, 0, lab)

    lbl = labels_from_probs(probs_arr)

    return lbl, cmap, norm, names, class_paths, preselected


def plot_model_segmentation(
    probs,
    summary,
    output_png=None,
    output_svg=None,
    fig_size=None,
    show=True,
    slice_indices=None,
    class_colors=None,
    cmap=None,
    names=None,
    imshow_kwargs=None,
    title_kwargs=None,
    colorbar_kwargs=None,
    colorbar_location="right",
    colorbar_width=None,
    colorbar_length=None,
    layout_kwargs=None,
    background_color="white",
    ball_color="lightgray",
    preselected_base_colors=None,
    preselected_factors=None,
    preselected_hls_factors=None,
    show_percent=True,
    group_similar=None,
    group_base_colors=None,
    axial_only=False,
    axial_coronal=False,
    axial_rotate_k=0,
    show_titles=True,
    fontsize=None,
):
    """Plot orthogonal slices of the model argmax map with a discrete colorbar.

    Customization hooks:
      - slice_indices: (x, y, z) tuple/list or dict with keys x/y/z.
      - class_colors: list (len=C) or dict {label: color} for labels 1..C.
      - cmap: colormap name or instance; used only if class_colors is None.
      - names: list (len=C) or dict {label: name} for colorbar labels.
      - imshow_kwargs / title_kwargs / colorbar_kwargs / layout_kwargs.
      - colorbar_location: "right" (default) or "bottom" for a horizontal colorbar.
      - colorbar_width: override colorbar thickness as a fraction of axes span.
      - colorbar_length: shrink colorbar length (0-1); smaller makes each color shorter.
      - preselected_hls_factors: list of (dh, lightness, saturation) factors.
      - group_similar: group labels by shorthand without counts (e.g., BZ and B2Z).
      - group_base_colors: dict mapping grouped labels (e.g., "BZ") to base colors.
      - axial_only: plot only the axial slice.
      - axial_coronal: plot only axial and coronal slices.
      - axial_rotate_k: rotate axial slice by 90 degrees k times (0, 1, 2, 3).
      - show_titles: toggle slice titles (e.g., "Axial (z=...)").
      - fontsize: override title and colorbar tick label size; None uses defaults.
    """
    if colorbar_location not in ("right", "bottom"):
        raise ValueError("colorbar_location must be 'right' or 'bottom'.")
    if colorbar_width is not None and colorbar_width <= 0:
        raise ValueError("colorbar_width must be positive or None.")
    if colorbar_length is not None and colorbar_length <= 0:
        raise ValueError("colorbar_length must be positive or None.")
    if axial_rotate_k not in (0, 1, 2, 3):
        raise ValueError("axial_rotate_k must be 0, 1, 2, or 3.")
    if axial_only and axial_coronal:
        raise ValueError("axial_only and axial_coronal cannot both be True.")
    lbl, cmap, norm, names, class_paths, preselected = _prepare_segmentation_inputs(
        probs,
        summary,
        class_colors=class_colors,
        cmap=cmap,
        names=names,
        background_color=background_color,
        ball_color=ball_color,
        preselected_base_colors=preselected_base_colors,
        preselected_factors=preselected_factors,
        preselected_hls_factors=preselected_hls_factors,
        show_percent=show_percent,
        group_similar=group_similar,
        group_base_colors=group_base_colors,
    )

    if fig_size is None:
        fig_size = (5, 3) if preselected else (5, 4)

    x_dim, y_dim, z_dim = lbl.shape
    x0, y0, z0 = x_dim // 2, y_dim // 2, z_dim // 2
    if slice_indices is not None:
        if isinstance(slice_indices, dict):
            x0 = slice_indices.get("x", x0)
            y0 = slice_indices.get("y", y0)
            z0 = slice_indices.get("z", z0)
        else:
            if len(slice_indices) != 3:
                raise ValueError("slice_indices must have three entries (x, y, z).")
            x0, y0, z0 = slice_indices
        for idx, dim, name in ((x0, x_dim, "x"), (y0, y_dim, "y"), (z0, z_dim, "z")):
            if not (0 <= idx < dim):
                raise ValueError(f"slice index {name}={idx} out of bounds for size {dim}.")

    sagittal = lbl[x0, :, :]
    coronal = lbl[:, y0, :]
    axial = lbl[:, :, z0]
    if axial_rotate_k:
        axial = np.rot90(axial, k=axial_rotate_k)

    def _title_kwargs(defaults):
        out = defaults.copy()
        if fontsize is not None:
            out.setdefault("fontsize", fontsize)
        if title_kwargs:
            out.update(title_kwargs)
        return out

    imshow_opts = {"origin": "lower", "interpolation": "nearest"}
    if imshow_kwargs:
        imshow_opts.update(imshow_kwargs)

    if axial_only:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=False)
        im2 = ax.imshow(axial.T, cmap=cmap, norm=norm, **imshow_opts)
        if show_titles:
            ax.set_title(f"Axial (z={z0})", **_title_kwargs({"pad": 2}))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if layout_kwargs:
            fig.subplots_adjust(**layout_kwargs)
    elif axial_coronal:
        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        gs = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 1.7],
            height_ratios=[1.0],
        )
        ax_l = fig.add_subplot(gs[0, 0])
        ax_r = fig.add_subplot(gs[0, 1])

        ax_l.imshow(coronal.T, cmap=cmap, norm=norm, **imshow_opts)
        im2 = ax_r.imshow(axial.T, cmap=cmap, norm=norm, **imshow_opts)

        if show_titles:
            ax_l.set_title(f"Coronal (y={y0})", **_title_kwargs({"pad": 1}))
            ax_r.set_title(f"Axial (z={z0})", **_title_kwargs({"pad": 2}))

        for ax in (ax_l, ax_r):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        adjust_opts = {
            "left": 0.05,
            "right": 0.90,
            "top": 0.93,
            "bottom": 0.06,
            "wspace": 0.04,
        }
        if layout_kwargs:
            adjust_opts.update(layout_kwargs)
        fig.subplots_adjust(**adjust_opts)
    else:
        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[1.0, 1.7],
            height_ratios=[1.0, 1.0],
        )

        ax_lt = fig.add_subplot(gs[0, 0])
        ax_lb = fig.add_subplot(gs[1, 0])
        ax_r = fig.add_subplot(gs[:, 1])

        ax_lt.imshow(sagittal.T, cmap=cmap, norm=norm, **imshow_opts)
        ax_lb.imshow(coronal.T, cmap=cmap, norm=norm, **imshow_opts)
        im2 = ax_r.imshow(axial.T, cmap=cmap, norm=norm, **imshow_opts)

        if show_titles:
            ax_lt.set_title(f"Sagittal (x={x0})", **_title_kwargs({"pad": 1}))
            ax_lb.set_title(f"Coronal (y={y0})", **_title_kwargs({"pad": 1}))
            ax_r.set_title(f"Axial (z={z0})", **_title_kwargs({"pad": 2}))

        for ax in (ax_lt, ax_lb, ax_r):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        adjust_opts = {
            "left": 0.04,
            "right": 0.88,
            "top": 0.93,
            "bottom": 0.05,
            "wspace": 0.03,
            "hspace": 0.00,
        }
        if layout_kwargs:
            adjust_opts.update(layout_kwargs)
        fig.subplots_adjust(**adjust_opts)

        gap = 0.0
        pos_top = ax_lt.get_position()
        pos_bot = ax_lb.get_position()
        y_bottom = pos_bot.y0
        y_top = pos_top.y1
        total_h = y_top - y_bottom
        new_h = (total_h - gap) / 2.0
        x0p = pos_bot.x0
        wp = pos_bot.width
        ax_lb.set_position([x0p, y_bottom, wp, new_h])
        ax_lt.set_position([x0p, y_bottom + new_h + gap, wp, new_h])

    c = len(class_paths)
    ticks = np.arange(1, c + 1)
    boundaries = np.arange(0.5, c + 1.5, 1)

    if axial_only:
        cb_axes = ax
    elif axial_coronal:
        cb_axes = [ax_l, ax_r]
    else:
        cb_axes = [ax_lt, ax_lb, ax_r]

    cb_kwargs = {
        "ax": cb_axes,
        "ticks": ticks,
        "boundaries": boundaries,
        "spacing": "proportional",
        "fraction": 0.06,
        "pad": 0.02,
    }
    if colorbar_location == "bottom":
        cb_kwargs.update(
            {
                "orientation": "horizontal",
                "fraction": 0.08,
                "pad": 0.08,
            }
        )
    if colorbar_kwargs:
        cb_kwargs.update(colorbar_kwargs)
    if colorbar_width is not None:
        cb_axes_list = cb_axes if isinstance(cb_axes, (list, tuple, np.ndarray)) else [cb_axes]
        cb_bbox = mtransforms.Bbox.union([ax.get_position() for ax in cb_axes_list])
        cb_kwargs["fraction"] = colorbar_width
    if colorbar_length is not None:
        cb_kwargs["shrink"] = colorbar_length
    cbar = fig.colorbar(im2, **cb_kwargs)
    if colorbar_width is not None:
        cpos = cbar.ax.get_position()
        if colorbar_location == "bottom":
            new_h = cb_bbox.height * colorbar_width
            cbar.ax.set_position([cpos.x0, cpos.y1 - new_h, cpos.width, new_h])
        else:
            new_w = cb_bbox.width * colorbar_width
            cbar.ax.set_position([cpos.x0, cpos.y0, new_w, cpos.height])
    if colorbar_location == "bottom":
        cbar.ax.set_xticklabels([names[i] for i in ticks])
        tick_params = {"axis": "x", "labelrotation": 90}
        cbar.ax.set_xlim(0.5, c + 0.5)
    else:
        cbar.ax.set_yticklabels([names[i] for i in ticks])
        tick_params = {}
        cbar.ax.set_ylim(0.5, c + 0.5)
    if fontsize is not None:
        tick_params["labelsize"] = fontsize
    tick_params.update({"colors": "black", "labelcolor": "black"})
    cbar.ax.tick_params(**tick_params)
    cbar.outline.set_edgecolor("black")

    if output_png:
        fig.savefig(output_png, dpi=300)
    if output_svg:
        fig.savefig(output_svg)
    if show:
        plt.show()

    return fig


def plot_model_segmentation_zgrid(
    probs,
    summary,
    z_slices=None,
    ncols=3,
    output_png=None,
    output_svg=None,
    fig_size=None,
    fig_title=None,
    show=True,
    class_colors=None,
    cmap=None,
    names=None,
    imshow_kwargs=None,
    title_kwargs=None,
    colorbar_kwargs=None,
    layout_kwargs=None,
    background_color="white",
    ball_color="lightgray",
    preselected_base_colors=None,
    preselected_factors=None,
    preselected_hls_factors=None,
    show_percent=True,
    group_similar=None,
    group_base_colors=None,
):
    """Plot multiple axial z slices in a grid with a shared colorbar."""
    if z_slices is None:
        z_slices = [5, 10, 15, 20, 25, 30, 35, 40, 49]
    z_slices = list(z_slices)
    if not z_slices:
        raise ValueError("z_slices must contain at least one index.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")

    lbl, cmap, norm, names, class_paths, preselected = _prepare_segmentation_inputs(
        probs,
        summary,
        class_colors=class_colors,
        cmap=cmap,
        names=names,
        background_color=background_color,
        ball_color=ball_color,
        preselected_base_colors=preselected_base_colors,
        preselected_factors=preselected_factors,
        preselected_hls_factors=preselected_hls_factors,
        show_percent=show_percent,
        group_similar=group_similar,
        group_base_colors=group_base_colors,
    )

    z_dim = lbl.shape[2]
    for z in z_slices:
        if not (0 <= z < z_dim):
            raise ValueError(f"slice index z={z} out of bounds for size {z_dim}.")

    ncols = min(ncols, len(z_slices))
    nrows = int(np.ceil(len(z_slices) / ncols))

    if fig_size is None:
        fig_size = (ncols * 2.2, nrows * 2.2) if not preselected else (ncols * 2.0, nrows * 2.0)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=fig_size,
        constrained_layout=False,
    )
    axes = np.array(axes).reshape(nrows, ncols)

    def _title_kwargs(defaults):
        out = defaults.copy()
        if title_kwargs:
            out.update(title_kwargs)
        return out

    imshow_opts = {"origin": "lower", "interpolation": "nearest"}
    if imshow_kwargs:
        imshow_opts.update(imshow_kwargs)

    ax_list = list(axes.ravel())
    for ax in ax_list[len(z_slices) :]:
        ax.axis("off")

    im2 = None
    for idx, z in enumerate(z_slices):
        ax = ax_list[idx]
        im2 = ax.imshow(lbl[:, :, z].T, cmap=cmap, norm=norm, **imshow_opts)
        ax.set_title(f"z={z}", **_title_kwargs({"fontsize": 9, "pad": 2}))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    adjust_opts = {
        "left": 0.04,
        "right": 0.90,
        "top": 0.95,
        "bottom": 0.05,
        "wspace": 0.02,
        "hspace": 0.02,
    }
    if layout_kwargs:
        adjust_opts.update(layout_kwargs)
    fig.subplots_adjust(**adjust_opts)

    c = len(class_paths)
    ticks = np.arange(1, c + 1)
    boundaries = np.arange(0.5, c + 1.5, 1)
    cb_axes = ax_list[: len(z_slices)]
    cb_kwargs = {
        "ax": cb_axes,
        "ticks": ticks,
        "boundaries": boundaries,
        "spacing": "proportional",
        "fraction": 0.05,
        "pad": 0.02,
    }
    if colorbar_kwargs:
        cb_kwargs.update(colorbar_kwargs)
    cbar = fig.colorbar(im2, **cb_kwargs)
    cbar.ax.set_yticklabels([names[i] for i in ticks])
    cbar.ax.tick_params(labelsize=7, colors="black", labelcolor="black")
    cbar.outline.set_edgecolor("black")
    cbar.ax.set_ylim(0.5, c + 0.5)
    if fig_title is not None:
        fig.suptitle(fig_title, y = 1.05)

    if output_png:
        fig.savefig(output_png, dpi=300)
    if output_svg:
        fig.savefig(output_svg)
    if show:
        plt.show()

    return fig, axes
