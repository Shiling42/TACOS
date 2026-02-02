#!/usr/bin/env python3
"""
Generate all figures for the paper in both PNG and SVG formats.

This script runs all the figure-generating scripts and produces:
- PNG files (for web/preview)
- SVG files (for publication)

Usage:
    python scripts/generate_all_figures.py --outdir notes
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path, args):
    """Run a Python script with given arguments."""
    cmd = [sys.executable, str(script_path)] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
    else:
        print(f"  OK")
    return result.returncode == 0


def add_svg_save_to_matplotlib():
    """Monkey-patch to also save SVG when saving PNG."""
    import matplotlib.pyplot as plt
    _original_savefig = plt.savefig

    def savefig_with_svg(fname, *args, **kwargs):
        _original_savefig(fname, *args, **kwargs)
        if str(fname).endswith('.png'):
            svg_fname = str(fname).replace('.png', '.svg')
            kwargs_svg = {k: v for k, v in kwargs.items() if k != 'dpi'}
            kwargs_svg['format'] = 'svg'
            _original_savefig(svg_fname, *args, **kwargs_svg)
            print(f"  Also saved: {svg_fname}")

    plt.savefig = savefig_with_svg


def main():
    parser = argparse.ArgumentParser(description="Generate all figures")
    parser.add_argument('--outdir', type=str, default='notes')
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip slow simulations')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).parent

    print("=" * 70)
    print("Generating all figures (PNG + SVG)")
    print("=" * 70)

    # List of (script, arguments)
    figure_scripts = [
        ('plot_self_assembly_from_efm.py',
         ['--out', str(outdir / 'fig_self_assembly_efm.png'), '--DeltaMu_over_RT', '2.0']),

        ('plot_schlogl_bounds.py',
         ['--out', str(outdir / 'fig_schlogl_bounds.png')]),

        ('plot_chiral_bounds.py',
         ['--out', str(outdir / 'fig_chiral_bounds.png')]),

        ('plot_self_assembly_probe_bound_from_efm.py',
         ['--out', str(outdir / 'fig_self_assembly_probe.png'), '--DeltaMu_over_RT', '2.0']),

        ('benchmark_efm_enumeration.py',
         ['--out', str(outdir / 'fig_EFM_search.png'), '--n-max', '12']),

        ('plot_affinity_bounds_validation.py',
         ['--outdir', str(outdir), '--DeltaMu_over_RT', '2.0', '--n', '200']),
    ]

    # Slow scripts (simulation-based)
    if not args.skip_slow:
        figure_scripts.append(
            ('sample_self_assembly_relaxation_panels.py',
             ['--outdir', str(outdir), '--DeltaMu_over_RT', '2.0', '--n', '300', '--seed', '42'])
        )

    success_count = 0
    for script_name, script_args in figure_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"\nSkipping {script_name} (not found)")
            continue

        print(f"\n--- {script_name} ---")
        if run_script(script_path, script_args):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"Completed: {success_count}/{len(figure_scripts)} scripts")
    print("=" * 70)

    # Now convert all PNG to SVG that weren't already converted
    print("\nConverting remaining PNG files to SVG...")

    # Use matplotlib to re-save as SVG
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    for png_file in outdir.glob('*.png'):
        svg_file = png_file.with_suffix('.svg')
        if not svg_file.exists():
            # For raster images, we can't truly vectorize
            # but we can embed them in SVG
            print(f"  Note: {png_file.name} is raster, SVG would just embed it")

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(outdir.glob('*')):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:50} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
