from pathlib import Path
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# Embed TrueType glyphs instead of Matplotlib's default Type 3 bitmap fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from .pdf_font_validation import verify_pdf_fonts


def _mean_accuracy(folder):
    csv_path = folder / 'experiment-results.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing required experimental results: {csv_path}')
    df = pd.read_csv(csv_path)
    if 'accuracy' not in df or 'run_id' not in df:
        raise ValueError(f'Missing accuracy/run_id columns: {csv_path}')
    if len(df) != 5 or set(df['run_id'].astype(int)) != set(range(1,6)):
        raise ValueError(f'Expected exactly five distinct seeds at {csv_path}, found {len(df)} records')
    return float(df['accuracy'].astype(float).mean())


def _sweep(output, xs, rg, erm, xlabel, png, *, y_min, y_max, y_step):
    # Render a CNN sweep with an independently chosen y-axis.

    with plt.rc_context({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
        'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16,
    }):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, rg, marker='o', linewidth=2.0, markersize=6, label='RG')
        ax.plot(xs, erm, marker='o', linestyle='--', linewidth=2.0,
                markersize=6, label='ERM')

        # Both x-axes are identical and each y-axis is fixed to its own range.
        ax.set_xlim(-0.025, 0.825)
        ax.set_xticks((0.0, 0.4, 0.8))
        ax.set_ylim(y_min, y_max)
        n_steps = round((y_max - y_min) / y_step)
        ax.set_yticks([round(y_min + i * y_step, 10)
                       for i in range(n_steps + 1)])
        ax.yaxis.set_minor_locator(MultipleLocator(y_step / 2))

        fmt = FuncFormatter(lambda x, _: f'{x * 100:.0f}%')
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Accuracy')
        ax.set_axisbelow(True)
        ax.grid(True, which='major', linestyle='--', alpha=0.35)
        ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.15)
        ax.legend(loc='upper right' if output.stem == 'cnn_guide_sweep'
                  else 'lower left')
        fig.tight_layout()

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches='tight')
        verify_pdf_fonts(output)
        if png:
            fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved CNN sweep, y-axis {y_min:.0%}-{y_max:.0%}] {output}', flush=True)


def generate_sweeps(results_dir, output_dir, *, png=False):
    # Regenerate CNN PDFs from final five-seed CSVs.
    root = Path(results_dir)
    out = Path(output_dir)
    noise = (0.0, .4, .8)
    noise_rg = [_mean_accuracy(root / f'CNN_n{n:.2f}_rg_guide-ResNet20_n0.00') for n in noise]
    noise_erm = [_mean_accuracy(root / f'CNN_n{n:.2f}_erm') for n in noise]
    guide_rg = [_mean_accuracy(root / f'CNN_n0.40_rg_guide-ResNet20_n{n:.2f}') for n in noise]
    guide_erm = [_mean_accuracy(root / 'CNN_n0.40_erm')]*3
    _sweep(out / 'cnn_noise_sweep.pdf', noise, noise_rg, noise_erm,
                  'Label noise ratio', png, y_min=0.75, y_max=0.95, y_step=0.05)
    _sweep(out / 'cnn_guide_sweep.pdf', noise, guide_rg, guide_erm,
                  'Guide label noise ratio', png, y_min=0.89, y_max=0.93, y_step=0.01)
    print('[CNN learner noise] RG:',noise_rg,'ERM:',noise_erm,flush=True)
    print('[CNN guide noise, learner fixed 40%] RG:', guide_rg,'ERM:', guide_erm,flush=True)
    return {'noise':(noise_rg,noise_erm),'guide':(guide_rg,guide_erm)}


def _selected_rg_params_from_csv(rg_folder):
    # Read RG parameters actually used by all final seeds.

    csv_path = Path(rg_folder) / 'experiment-results.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f'RG final-run CSV not found: {csv_path}')
    rows = pd.read_csv(csv_path)
    required = {'run_id', 'gamma', 'delta', 'lambda', 'selected_trial'}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f'{csv_path}: missing columns {sorted(missing)}')
    if len(rows) != 5 or set(rows['run_id'].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError(f'{csv_path}: expected exactly five final seeds r1..r5')
    params = {}
    for key in ('gamma', 'delta', 'lambda'):
        values = pd.to_numeric(rows[key], errors='raise').to_numpy(dtype=float)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f'{csv_path}: non-finite {key}')
        if max(abs(float(v) - float(values[0])) for v in values) > 1e-8 * max(1.0, abs(float(values[0]))):
            raise ValueError(f'{csv_path}: inconsistent {key} across final seeds: {values.tolist()}')
        params[key] = float(values[0])
    if params['delta'] <= 0 or params['gamma'] <= 0:
        raise ValueError(f'{csv_path}: invalid gamma/delta: {params}')
    trials = pd.to_numeric(rows['selected_trial'], errors='raise').to_numpy(dtype=int)
    if len(set(trials.tolist())) != 1:
        raise ValueError(f'{csv_path}: inconsistent selected_trial across final seeds')
    params['selected_trial'] = int(trials[0])
    return params


def generate_r_distribution(results_dir, output_dir, *, device='auto', data_dir='data'):
    # Create the eight-panel PDF.

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from core.dataset_builder import DatasetBuilder
    from core.evaluation_distributions import _load_checkpoint_model, _resolve_device
    from core.experiment_paths import method_dir

    path = Path(results_dir)
    dev = _resolve_device(device)
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
    ])
    original = datasets.FashionMNIST(root=str(data_dir), train=True,
                                      transform=test_transform, download=False)
    indices = np.random.default_rng(42).permutation(len(original))[:54000]
    clean = np.asarray(original.targets, dtype=np.int64)[indices]
    
    conditions = [
        ('CNN', .4, .4), ('CNN', .8, .8),
        ('Transformer', .4, .4), ('Transformer', .8, .8),
        ('MLP', .4, .4), ('MLP', .8, .8),
        ('CNN', .4, 0.), ('CNN', .8, 0.),
    ]
    loader = DataLoader(Subset(original, indices), batch_size=256,
                        shuffle=False, num_workers=0)
    learner_logits_cache = {}
    ref_cache = {}
    panel_data = {}
    for arch, noise, guide_noise in conditions:
        print(f'{arch} learner={noise:.0%}, guide={guide_noise:.0%}', flush=True)
        ds = DatasetBuilder(noise_ratio=noise, baseline_noise_ratio=guide_noise,
                            baseline_method='ResNet20', seed=42, device=str(dev))
        ds.build()
        labels = ds.train_dataset.labels.cpu().long()
        guide_logits = ds.baseline_train_outputs.cpu().float()
        if len(labels) != len(clean) or len(guide_logits) != len(clean):
            raise RuntimeError('Train split or guide logits alignment failed')
        labels_numpy = labels.numpy()
        changed = labels_numpy != clean
        if (arch, noise) not in learner_logits_cache:
            from importlib import import_module
            module = {'CNN':'models.cnn2d','MLP':'models.mlp_model',
                      'Transformer':'models.transformer'}[arch]
            cls = import_module(module).MyModel
            ckpt = path / method_dir(arch,noise,'erm').name / 'best-model.pt'
            if not ckpt.exists():
                raise FileNotFoundError(f'Requires selected ERM checkpoint: {ckpt}')
            model = _load_checkpoint_model(cls,ckpt,dev)
            outputs = []
            with torch.inference_mode():
                for xb,_ in loader:
                    outputs.append(model(xb.to(dev)).float().cpu())
            learner_logits_cache[(arch,noise)] = torch.cat(outputs)
            del model
        rg_folder = path / method_dir(arch,noise,'rg','ResNet20',guide_noise).name
        params = _selected_rg_params_from_csv(rg_folder)
        gamma = params['gamma']; delta = params['delta']
        print(f'[RG parameters] {rg_folder.name}: '
              f'gamma={gamma:.8g}, delta={delta:.8g}, '
              f'lambda={params["lambda"]:.8g}, '
              f'selected_trial={params["selected_trial"]}; '
              'source=experiment-results.csv', flush=True)
        with torch.no_grad():
            learner_ce = F.cross_entropy(learner_logits_cache[(arch,noise)], labels,
                                         reduction='none').numpy()
            guide_ce = F.cross_entropy(guide_logits, labels,
                                       reduction='none').numpy()
        r = learner_ce - gamma*guide_ce
        panel_data[(arch,noise,guide_noise)] = (r,changed,delta)
        del ds

    import numpy as np
    # Match CNN, ViT, MLP, then the 2 clean guides.
    rank = {'CNN':0,'Transformer':1,'MLP':2}
    keys = sorted(panel_data,key=lambda k:(0 if k[1]==k[2] else 1,rank[k[0]],k[1]))
    ncol = 3
    nrow = int(np.ceil(len(keys)/ncol))
    fig, axes = plt.subplots(nrow,ncol,figsize=(5*ncol,3.6*nrow),squeeze=False)
    for ax,key in zip(axes.ravel(),keys):
        r,changed,delta = panel_data[key]
        lo,hi = np.percentile(r,(.5,99.5))
        bins = np.linspace(lo,hi,80)
        ax.hist(r[~changed],bins=bins,alpha=.6,density=True,label='clean')
        ax.hist(r[changed],bins=bins,alpha=.6,density=True,label='corrupted')
        if lo<0:
            ax.axvspan(lo,0.,color='.55',alpha=.30,zorder=0)
        ax.axvline(0.,color='k',lw=1,zorder=3)
        ax.axvline(delta,color='r',ls='--',lw=1,zorder=3,
                   label=rf'$\delta = {delta:.1e}$')
        ymax=ax.get_ylim()[1]
        ax.set_ylim(0,ymax*1.22)
        if lo<0:
            ax.text(lo+(0-lo)/2,ymax*1.10,r'$\omega_i = 1$',
                    ha='center',va='center',fontsize=9,color='.10')
        off_corrupt=float((r[changed]<=0).mean())
        off_clean=float((r[~changed]<=0).mean())
        arch_display='ViT' if key[0]=='Transformer' else key[0]
        ax.set_title(f'{arch_display} $\\rho$={key[1]:.0%} | guide $\\rho$={key[2]:.0%}\\n'
                     +rf'$\omega_i=1$ on {off_corrupt:.1%} of corrupted, {off_clean:.1%} of clean',
                     fontsize=10)
        # Resolve explicit newline: formatting above uses literal slash-n so replace.
        ax.title.set_text(ax.get_title().replace('\\n','\n'))
        ax.set_xlabel(r'$r_i = L_i - \gamma L_i^{guide}$')
        ax.set_ylabel('density')
        ax.set_xlim(lo,hi)
        ax.legend(fontsize=8,loc='upper right',framealpha=.9)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    fig.tight_layout()
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('.pdf','.png'):
        target=out/('guide_memorization_r_dist'+ext)
        with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(target,bbox_inches='tight',**({'dpi':200} if ext=='.png' else {}))
        if ext == '.pdf':
            verify_pdf_fonts(target)
        print(f'[saved figure] {target}',flush=True)
    plt.close(fig)
