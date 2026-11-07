# Generate checkpoint-based cross-entropy distribution plots and statistics.

import csv
import re
from pathlib import Path
from .experiment_paths import method_dir
from .pdf_font_validation import verify_pdf_fonts

import matplotlib
matplotlib.use("Agg")
# embed TrueType glyphs instead of Matplotlib's default Type 3 bitmap fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


ARCHITECTURES = {"CNN": "cnn", "Transformer": "vit"}
FIGURE_GENERATOR_VERSION = "paper-kde-numpy-exact-20261101-v4"


def _experiment_dirs(architecture, noise_ratio, guide_noise_ratio):
    """Resolve learner and reference checkpoints using canonical names."""
    dirs = {
        method.upper(): method_dir(architecture, noise_ratio, method,
                                   'ResNet20' if method in ('rg', 'kd') else None,
                                   guide_noise_ratio).name
        for method in ('erm', 'rg', 'kd', 'spl', 'mentornet-pd')
    }
    return dirs, method_dir('ResNet20', guide_noise_ratio, 'erm').name


def _resolve_device(device_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return device


def _check_inputs(args, method_folders, selected_ids, guide_path):
    if args.x_max < 0 or not np.isfinite(args.x_max):
        raise ValueError('--x-max must be finite and >= 0 (0 selects the percentile-based grid)')
    if args.batch_size < 1 or args.grid_points < 50:
        raise ValueError("--batch-size must be positive; --grid-points must be >= 50")
    if not 0 < args.grid_percentile <= 100:
        raise ValueError("--grid-percentile must be in (0, 100]")
    if args.global_bandwidth <= 0 or args.subset_bandwidth <= 0:
        raise ValueError("KDE bandwidths must be strictly positive")
    for key, rate in (("noise-ratio", args.noise_ratio),
                      ("guide-noise-ratio", args.guide_noise_ratio)):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"--{key} must be within [0, 1]")
    if not guide_path.is_file():
        raise FileNotFoundError(f"ResNet20 guide checkpoint not found: {guide_path}")
    if guide_path.parent.name.startswith(("CNN_", "Transformer_")):
        raise ValueError(
            f"Guide path points to a learner checkpoint, not ResNet20: {guide_path}. "
            "Use --guide-dir ResNet20_n0.40_erm "
            "(adjust noise ratio as necessary)."
        )
    missing = []
    for method, folder in method_folders.items():
        ids = selected_ids[method]
        if len(ids) != 5 or len(set(ids)) != 5 or any(i < 1 for i in ids):
            raise ValueError(f"{method}: specify exactly five distinct positive run IDs, got {ids}")
        for run_id in ids:
            ckpt_path = args.results_dir / folder / f"best-model_r{run_id}.pt"
            if not ckpt_path.is_file():
                missing.append(f"{method} r{run_id}: {ckpt_path}")
    if missing:
        raise FileNotFoundError("Missing checkpoints; nothing evaluated:\n" + "\n".join(missing))






_VIT_ENCODER_TO_LAYERS = {
    "attention.attention.query": "attention.q_proj",
    "attention.attention.key": "attention.k_proj",
    "attention.attention.value": "attention.v_proj",
    "attention.output.dense": "attention.o_proj",
    "intermediate.dense": "mlp.fc1",
    "output.dense": "mlp.fc2",
    "layernorm_before": "layernorm_before",
    "layernorm_after": "layernorm_after",
}
_VIT_LAYERS_TO_ENCODER = {v: k for k, v in _VIT_ENCODER_TO_LAYERS.items()}
_VIT_LAYER_RE = re.compile(r"^vit\.(encoder\.layer|layers)\.(\d+)\.(.+)$")


def _convert_vit_state_dict(state_dict, model, checkpoint_path):
    """Map ViT checkpoint parameter names between supported layouts."""
    model_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    if checkpoint_keys == model_keys:
        return state_dict

    old_layout = any(k.startswith("vit.encoder.layer.") for k in checkpoint_keys)
    new_layout = any(k.startswith("vit.layers.") for k in model_keys)
    reverse_old_layout = any(k.startswith("vit.layers.") for k in checkpoint_keys)
    reverse_new_layout = any(k.startswith("vit.encoder.layer.") for k in model_keys)
    if old_layout and new_layout:
        source_prefix, target_prefix = "encoder.layer", "layers"
        rename = _VIT_ENCODER_TO_LAYERS
    elif reverse_old_layout and reverse_new_layout:
        source_prefix, target_prefix = "layers", "encoder.layer"
        rename = _VIT_LAYERS_TO_ENCODER
    else:

        return state_dict

    converted = {}
    renamed_count = 0
    for key, tensor in state_dict.items():
        replacement = key
        match = _VIT_LAYER_RE.fullmatch(key)
        if match and match.group(1) == source_prefix:
            block_id, name_and_type = match.group(2), match.group(3)
            name, separator, field = name_and_type.rpartition(".")
            if separator and field in ("weight", "bias") and name in rename:
                replacement = f"vit.{target_prefix}.{block_id}.{rename[name]}.{field}"
                renamed_count += 1
        if replacement in converted:
            raise RuntimeError(f"Duplicate ViT tensor key after remapping: {replacement} ({checkpoint_path})")
        converted[replacement] = tensor

    actual = set(converted)
    if actual != model_keys:
        missing = sorted(model_keys - actual)
        unexpected = sorted(actual - model_keys)
        raise RuntimeError(
            f"ViT checkpoint incompatible after key mapping: {checkpoint_path}\n"
            f"Missing ({len(missing)}): {missing[:8]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:8]}\n"
            "For a matching architecture, use the training dependency "
            "version: python -m pip install transformers==4.55.4"
        )
    print(f"[ViT compatibility] renamed {renamed_count} state_dict keys ({source_prefix} -> {target_prefix}); all keys matched", flush=True)
    return converted


def _load_checkpoint_model(model_class, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_params" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    params, _ = model_class.prepare_model_params(dict(checkpoint["model_params"]))
    model = model_class(**params)
    state_dict = checkpoint["state_dict"]
    if getattr(model_class, "model_name", None) == "Transformer":
        state_dict = _convert_vit_state_dict(state_dict, model, checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model_class.model_postprocess(model)
    return model.to(device).eval()


@torch.inference_mode()
def _test_ce_and_preds(model_class, checkpoint_path, test_loader, device):
    # Ordered, unaugmented Fashion-MNIST test examples, with CE on true labels.
    model = _load_checkpoint_model(model_class, checkpoint_path, device)
    losses, predictions = [], []
    for images, labels in test_loader:
        logits = model(images.to(device)).float().cpu()
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
        if not torch.isfinite(per_sample_ce).all():
            raise ValueError(f"Nonfinite per-sample CE in {checkpoint_path}")
        losses.append(per_sample_ce.numpy())
        predictions.append(logits.argmax(dim=1).numpy())
    del model
    if not losses:
        raise ValueError("Test loader produced no batches")
    return np.concatenate(losses), np.concatenate(predictions)


def _gaussian_kde_exact(samples, x_grid, bandwidth_factor):
    # Evaluate the original 1D Gaussian KDE without SciPy's compiled kernel.

    # Identical mathematical definition to
    # ``scipy.stats.gaussian_kde(samples, bw_method=factor)(grid)``:
    # bandwidth = sample_std(ddof=1) * factor, each observation has mass 1/n.
    
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if data.size < 2 or not np.isfinite(data).all():
        raise ValueError('Gaussian KDE requires at least two finite observations')
    if not (np.isfinite(bandwidth_factor) and bandwidth_factor > 0):
        raise ValueError('KDE bandwidth factor must be positive and finite')
    sigma = float(np.std(data, ddof=1)) * float(bandwidth_factor)
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError('Gaussian KDE requires nonzero finite sample variance')

    # Bound peak allocation to about 16 MB per block (two float64 arrays).
    block_size = max(1, min(grid.size, 1_000_000 // data.size))
    density = np.empty_like(grid)
    normalizer = 1.0 / (data.size * sigma * np.sqrt(2.0 * np.pi))
    for start in range(0, grid.size, block_size):
        segment = grid[start:start + block_size]
        z = (segment[:, None] - data[None, :]) / sigma
        # exp(-z*z/2), summed over all examples, is the Gaussian KDE.
        np.square(z, out=z)
        z *= -0.5
        np.exp(z, out=z)
        density[start:start + segment.size] = z.sum(axis=1) * normalizer
    if not np.isfinite(density).all():
        raise RuntimeError('Nonfinite KDE density; inspect per-example losses')
    return density


def _make_kde(losses, x_grid, bw_method, *, mode='auto', label=None, audit=None):
    #CE KDE Gaussian kernel.

    # Original mode evaluates the unmodified per-example losses using the exact
    # Gaussian KDE formula used in the paper plots. Auto mode changes only the PLOT, and only when a
    # truly extreme tail destroys visible density (its mass is retained).
    
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    grid = np.asarray(x_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 50 or not np.isfinite(grid).all() or not np.all(np.diff(grid) > 0):
        raise ValueError('KDE grid must be a finite, strictly increasing 1-D array with >=50 points')
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f'{label}: KDE requires at least two finite per-example losses')
    if not np.any((values >= grid[0]) & (values <= grid[-1])):
        raise ValueError(f'{label}: no losses inside the visible window. Increase --x-max.')
    if np.ptp(values) < 1e-12:
        raise ValueError(f'{label}: KDE requires non-degenerate per-example losses')
    if mode not in ('auto', 'original', 'robust'):
        raise ValueError(f'Unknown KDE mode: {mode}')
    in_range = values[(values >= grid[0]) & (values <= grid[-1])]
    fraction = in_range.size / values.size
    if in_range.size < 2:
        raise ValueError(f'{label}: fewer than two losses within [{grid[0]:.3g},{grid[-1]:.3g}]. '
                         'Increase --x-max to include observations.')
    core_std = float(np.std(in_range, ddof=1))
    full_std = float(np.std(values, ddof=1))
    sd_ratio = full_std / max(core_std, 1e-12)

    raw = None if mode == 'robust' else _gaussian_kde_exact(values, grid, bw_method)
    raw_peak = None if raw is None else float(raw.max())
    tail_fallback = mode == 'auto' and sd_ratio >= 8 and raw_peak < 0.20
    use_window = mode == 'robust' or tail_fallback
    if use_window:
        if in_range.size < 10 or core_std <= 1e-12:
            raise ValueError(f'{label}: insufficient variability within the displayed KDE range')
        # Keep probability mass
        density = _gaussian_kde_exact(in_range, grid, bw_method) * fraction
    else:
        density = raw
    peak = float(density.max())
    mode_used = 'robust-fallback' if tail_fallback else ('robust' if use_window else 'original')
    print(f'[KDE] {label or "sample"}: n={values.size}; median={np.median(values):.4g}; '
          f'p99.5={np.percentile(values, 99.5):.4g}; max={values.max():.4g}; '
          f'window={fraction:.1%}; full/core SD={sd_ratio:.2f}; '
          f'raw_peak={raw_peak}; plotted_peak={peak:.4g}; '
          f'calculator=NumPy-exact; mode={mode_used}', flush=True)
    if fraction < 0.70:
        print(f'[KDE WARNING] {label}: {1-fraction:.1%} of observations are outside '
              'the visible x-axis window; consider a larger --x-max.', flush=True)
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError(f'{label}: no usable KDE density in the display window; '
                           'check losses and --x-max')
    if audit is not None:
        audit.append({'curve': label, 'n': int(values.size),
                      'median_ce': float(np.median(values)),
                      'p995_ce': float(np.percentile(values, 99.5)),
                      'max_ce': float(values.max()),
                      'visible_fraction': fraction, 'sd_ratio': sd_ratio,
                      'raw_peak': raw_peak, 'plotted_peak': peak,
                      'kde_mode': mode_used, 'backend': 'numpy-exact'})
    return density


def _stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "sd_sample": float(v.std(ddof=1)) if len(v) > 1 else None,
            "n": int(len(v))}


def _plot_kde(path, x_grid, reference_density, method_densities, *, shade=False, png=False):
    # 7x5 layout and labels
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'font.size': 16, 'axes.titlesize': 16,
                         'axes.labelsize': 16, 'xtick.labelsize': 16,
                         'ytick.labelsize': 16, 'legend.fontsize': 16}):
        fig, ax = plt.subplots(figsize=(7, 5))
        if reference_density is not None:
            ax.plot(x_grid, reference_density, linewidth=3, label='Guide')
        for method, densities in method_densities.items():
            curves = np.stack(densities)
            center = curves.mean(axis=0)
            display_name = {'RG': 'RG', 'MENTORNET-PD': 'MentorNet-PD'}.get(method, method)
            (line,) = ax.plot(x_grid, center, label=display_name)
            if shade and len(curves) > 1:
                sd = curves.std(axis=0, ddof=1)
                ax.fill_between(x_grid, np.maximum(0, center - sd), center + sd,
                                alpha=.13, color=line.get_color(), linewidth=0)
        ax.set_xlabel('Cross-Entropy per Sample')
        ax.set_ylabel('Density')
        ax.grid(True)
        ax.legend()
        fig.savefig(path, bbox_inches='tight')
        verify_pdf_fonts(path)
        if png:
            fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f'[saved] {path}')


def run_error_distributions(args):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from models.resnet20 import MyModel as ResNet20
    if args.architecture == "CNN":
        from models.cnn2d import MyModel as model_class
    else:
        from models.transformer import MyModel as model_class

    device = _resolve_device(args.device)
    if args.guide_noise_ratio is None:
        args.guide_noise_ratio = args.noise_ratio
    defaults, default_guide_dir = _experiment_dirs(
        args.architecture, args.noise_ratio, args.guide_noise_ratio)
    selected_methods = tuple(m.upper() for m in args.methods if m != "reference")
    include_reference = "reference" in args.methods
    method_dirs = {
        m: getattr(args, f"{m.lower().replace('-', '_')}_dir") or defaults[m]
        for m in selected_methods
    }
    selected_ids = {
        m: getattr(args, f"{m.lower().replace('-', '_')}_runs")
        for m in selected_methods
    }
    guide_dir = args.guide_dir or default_guide_dir
    guide_path = (args.guide_checkpoint if args.guide_checkpoint is not None
                  else args.results_dir / guide_dir / "best-model.pt")
    _check_inputs(args, method_dirs, selected_ids, guide_path)
    file_tag = ARCHITECTURES[args.architecture]
    noise_tag = f"{round(100 * args.noise_ratio):02d}noise"
    print(f"[config] learner={args.architecture} | plotted={','.join(args.methods)} | guide=ResNet20 | "
          f"noise={args.noise_ratio:.0%} | guide noise={args.guide_noise_ratio:.0%}", flush=True)


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    test_dataset = datasets.FashionMNIST(root=str(args.data_dir), train=False,
                                         transform=test_transform, download=not args.no_download)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    labels = np.asarray(test_dataset.targets, dtype=np.int64)
    print(f"[test] {len(labels)} clean, ordered examples | device={device}")

    print(f"[guide] {guide_path}", flush=True)
    guide_ce, guide_pred = _test_ce_and_preds(ResNet20, guide_path, test_loader, device)
    if len(guide_ce) != len(labels):
        raise RuntimeError("Guide predictions have a different number of test samples")
    guide_wrong = guide_pred != labels
    n_wrong = int(guide_wrong.sum())
    if n_wrong < 2:
        raise ValueError("Guide has fewer than two misclassified test examples")
    print(f"[guide] misclassified {n_wrong}/{len(labels)} clean test samples")
    print(f"[guide] test accuracy = {100 * (1 - n_wrong / len(labels)):.2f}%")

    ce_by_method = {}
    subset_by_method = {}
    rows = []
    all_loss_arrays = [guide_ce]

    for method, folder in method_dirs.items():
        ce_by_method[method] = []
        subset_by_method[method] = []
        for rid in selected_ids[method]:
            ckpt_path = args.results_dir / folder / f"best-model_r{rid}.pt"
            print(f"[{method} r{rid}] {ckpt_path}", flush=True)
            ce, preds = _test_ce_and_preds(model_class, ckpt_path, test_loader, device)
            if len(ce) != len(guide_ce):
                raise RuntimeError(f"Length mismatch in {ckpt_path}")
            ce_wrong = ce[guide_wrong]
            delta_wrong = ce_wrong.astype(np.float64) - guide_ce[guide_wrong]
            ce_by_method[method].append(ce)
            subset_by_method[method].append(ce_wrong)
            all_loss_arrays.append(ce)
            rho = float(spearmanr(guide_ce, ce).statistic)
            if not np.isfinite(rho):
                raise ValueError(f"Spearman undefined for {ckpt_path}")
            rows.append({
                "method": method, "run_id": rid,
                "test_accuracy": float(np.mean(preds == labels)),
                "spearman_global": rho,
                "ks_global": float(ks_2samp(guide_ce, ce).statistic),
                "wasserstein_global": float(wasserstein_distance(guide_ce, ce)),
                "ks_guide_miss": float(ks_2samp(guide_ce[guide_wrong], ce_wrong).statistic),
                "wasserstein_guide_miss": float(wasserstein_distance(guide_ce[guide_wrong], ce_wrong)),
                "fraction_lower_ce_guide_miss": float(np.mean(delta_wrong < 0)),
                "mean_ce_difference_guide_miss": float(delta_wrong.mean()),
            })

    if args.x_max > 0:
        upper = float(args.x_max)
        grid_reason = 'explicit paper window'
    elif args.grid_source == "reference":
        upper = float(np.percentile(guide_ce, args.grid_percentile))
        grid_reason = 'reference percentile'
    else:
        upper = max(float(np.percentile(arr, args.grid_percentile)) for arr in all_loss_arrays)
        grid_reason = 'all-distributions percentile'
    if not np.isfinite(upper) or upper <= 0:
        raise RuntimeError("Invalid CE distribution / grid range")
    x_grid = np.linspace(0, upper, args.grid_points)
    print(f"[KDE grid] 0..{upper:.6g}; step={x_grid[1]-x_grid[0]:.6g}; "
          f"source={grid_reason}; guide median={np.median(guide_ce):.6g}; "
          f"guide p99.5={np.percentile(guide_ce,99.5):.6g}; "
          f"guide max={np.max(guide_ce):.6g}", flush=True)
    if x_grid[1] - x_grid[0] > .05:
        raise ValueError('KDE grid is too sparse for a readable plot '
                         f'(step={x_grid[1]-x_grid[0]:.4g}); '
                         'rerun with --x-max 3.0 (or another sensible CE limit).')
    # Keep per-curve diagnostics.
    audit = []
    by_global = {
        method: [_make_kde(ce, x_grid, args.global_bandwidth,
                           mode=args.kde_mode, label=f'{method} r{rid} global', audit=audit)
                 for rid, ce in zip(selected_ids[method], runs)]
        for method, runs in ce_by_method.items()
    }
    by_subset = {
        method: [_make_kde(ce, x_grid, args.subset_bandwidth,
                           mode=args.kde_mode, label=f'{method} r{rid} guide-wrong', audit=audit)
                 for rid, ce in zip(selected_ids[method], runs)]
        for method, runs in subset_by_method.items()
    }
    ref_global = (_make_kde(guide_ce, x_grid, args.global_bandwidth,
                            mode=args.kde_mode, label='Guide global', audit=audit)
                  if include_reference else None)
    ref_subset = (_make_kde(guide_ce[guide_wrong], x_grid, args.subset_bandwidth,
                            mode=args.kde_mode, label='Guide guide-misclassified', audit=audit)
                  if include_reference else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_pdf = args.output_dir / f'fig_global_error_dist_{file_tag}.pdf'
    subset_pdf = args.output_dir / f'fig_guide_miss_error_dist_{file_tag}.pdf'
    _plot_kde(global_pdf, x_grid, ref_global, by_global,
              shade=args.show_run_variation, png=args.png)
    _plot_kde(subset_pdf, x_grid, ref_subset, by_subset,
              shade=args.show_run_variation, png=args.png)

    import shutil
    l_suffix = '' if args.architecture == 'Transformer' else '_cnn'
    for source, stem in ((global_pdf, 'global-error-distribution'),
                         (subset_pdf, 'reference-error-distribution')):
        alias = args.output_dir / f'{stem}{l_suffix}.pdf'
        shutil.copyfile(source, alias)
        print(f'[saved alias] {alias}', flush=True)
    diagnostic_path = args.output_dir / f'{file_tag}_{noise_tag}_kde_audit.csv'
    with diagnostic_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(f'[saved] {diagnostic_path}', flush=True)

    row_path = args.output_dir / f"{file_tag}_{noise_tag}_alignment_per_run.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {row_path}")

    metric_keys = [k for k in rows[0] if k not in ("method", "run_id")]
    summary_rows = []
    for method in selected_methods:
        runs = [r for r in rows if r["method"] == method]
        for metric in metric_keys:
            stats = _stats([r[metric] for r in runs])
            summary_rows.append({"method": method, "metric": metric, **stats})
            if metric != "test_accuracy":
                print(f"{method:<4} {metric:<24} {stats['mean']:.4f} ± {stats['sd_sample']:.4f} (n=5)")
    summary_path = args.output_dir / f"{file_tag}_{noise_tag}_alignment_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[saved] {summary_path}")
