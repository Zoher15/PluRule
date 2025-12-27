#!/usr/bin/env python3
"""
Central plotting configuration for all visualization scripts.

Ensures consistent styling across all plots to match ACL paper format.
All plotting scripts should import and use these settings.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# ACL PAPER DIMENSIONS
# ============================================================================
# ACL uses a4paper with 2.5cm (0.984in) margins on all sides
# A4: 8.27" x 11.69"
# Margins: 2.5cm = 0.984"
# Usable text width: 8.27 - 2*(0.984) = 6.3"

SINGLE_COLUMN_WIDTH = 3.15  # inches (one column in 2-column layout)
TWO_COLUMN_WIDTH = 6.3      # inches (full text width)

# Figure heights by plot type
FIGURE_HEIGHT_BARPLOT = 3.4125        # inches (bar and distribution charts)
FIGURE_HEIGHT_DISTRIBUTION = 2.0      # inches (same as bar plots)
FIGURE_HEIGHT_SCATTER = 3.15          # inches (scatter/2D plots - square-ish subplots)

# Publication quality settings
PUBLICATION_DPI = 300  # DPI for PDF/PNG output (publication-quality resolution)

# ============================================================================
# FONTS & TYPOGRAPHY
# ============================================================================
# ACL template uses:
# - Body: Times Roman (via \usepackage{times})
# - Monospace: Inconsolata (via \usepackage{inconsolata})

# Configure matplotlib to use Times Roman like ACL
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.monospace'] = ['Courier New', 'Courier', 'monospace']

# Font sizes consistent with ACL (11pt body, smaller for figures)
mpl.rcParams['font.size'] = 9        # Default text size for figure content
mpl.rcParams['axes.labelsize'] = 9   # Axis labels
mpl.rcParams['axes.titlesize'] = 10  # Subplot titles (slightly larger)
mpl.rcParams['xtick.labelsize'] = 8  # Tick labels (smaller)
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.titlesize'] = 11

# ============================================================================
# FIGURE STYLING
# ============================================================================
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 0.5

# Grid and spines
mpl.rcParams['axes.grid'] = False
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.minor.width'] = 0.25
mpl.rcParams['ytick.minor.width'] = 0.25

# Legend styling
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.9
mpl.rcParams['legend.edgecolor'] = 'black'

# Save settings for PDF (editable in Illustrator)
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts
mpl.rcParams['ps.fonttype'] = 42   # TrueType for PostScript

# ============================================================================
# FIGURE TEMPLATES
# ============================================================================

def create_two_column_figure(figsize=None, plot_type='barplot', **kwargs):
    """Create a two-column figure with standard ACL dimensions.

    Args:
        figsize: Override figure size tuple (width, height)
        plot_type: 'barplot', 'distribution', or 'scatter' to auto-select height

    Returns:
        (fig, (ax_left, ax_right)): Figure and left/right axes
    """
    if figsize is None:
        if plot_type == 'scatter':
            height = FIGURE_HEIGHT_SCATTER
        elif plot_type == 'distribution':
            height = FIGURE_HEIGHT_DISTRIBUTION
        else:  # barplot (default)
            height = FIGURE_HEIGHT_BARPLOT
        figsize = (TWO_COLUMN_WIDTH, height)

    fig, axes = plt.subplots(1, 2, figsize=figsize, **kwargs)
    fig.patch.set_facecolor('white')

    return fig, axes


def create_single_column_figure(figsize=None, plot_type='barplot', **kwargs):
    """Create a single-column figure with standard ACL dimensions.

    Args:
        figsize: Override figure size tuple (width, height)
        plot_type: 'barplot', 'distribution', or 'scatter' to auto-select height

    Returns:
        (fig, ax): Figure and axes
    """
    if figsize is None:
        if plot_type == 'scatter':
            height = FIGURE_HEIGHT_SCATTER
        elif plot_type == 'distribution':
            height = FIGURE_HEIGHT_DISTRIBUTION
        else:  # barplot (default)
            height = FIGURE_HEIGHT_BARPLOT
        figsize = (SINGLE_COLUMN_WIDTH, height)

    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    fig.patch.set_facecolor('white')

    return fig, ax


# ============================================================================
# COMMON COLORS & PALETTES
# ============================================================================
# Paul Tol's scientifically-inspired color palettes (used in cluster plots)
# See: https://personal.sron.nl/~pault/colourschemes.pdf

# Vibrant palette (9 colors)
VIBRANT_COLORS = [
    '#EE7733',  # orange
    '#0077BB',  # blue
    '#33BBEE',  # cyan
    '#EE3377',  # magenta
    '#CC3311',  # red
    '#009988',  # teal
    '#BBBBBB',  # gray
]

# ============================================================================
# SAVE UTILITIES
# ============================================================================

def save_figure(fig, output_path, dpi=None, bbox_inches='tight'):
    """Save figure in both PDF and PNG formats.

    Args:
        fig: Matplotlib figure
        output_path: Path without extension (will add .pdf and .png)
        dpi: Resolution in DPI (defaults to PUBLICATION_DPI)
        bbox_inches: 'tight' to remove excess whitespace
    """
    if dpi is None:
        dpi = PUBLICATION_DPI

    output_path = Path(output_path)

    # Save PDF (editable)
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches=bbox_inches)
    print(f"✓ Saved: {pdf_path} ({dpi} DPI)")

    # Save PNG (quick preview)
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, format='png', dpi=dpi, bbox_inches=bbox_inches)
    print(f"✓ Saved: {png_path} ({dpi} DPI)")


# ============================================================================
# LAYOUT HELPERS
# ============================================================================

def apply_tight_layout(fig, **kwargs):
    """Apply tight layout with padding suitable for ACL papers."""
    defaults = {'pad': 0.3, 'w_pad': 0.5, 'h_pad': 0.5}
    defaults.update(kwargs)
    fig.tight_layout(**defaults)


def add_subplot_labels(axes, labels=None, loc='upper left', fontweight='normal', fontsize=10):
    """Add (a), (b), (c) style labels to subplots (no box).

    Args:
        axes: List of axes or single axis
        labels: List of labels (defaults to a, b, c, ...)
        loc: Location ('upper left', 'upper right', etc.)
        fontweight: 'bold' or 'normal'
        fontsize: Font size for labels
    """
    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes))]

    for ax, label in zip(axes, labels):
        ax.text(0.02, 0.98, f'({label})', transform=ax.transAxes,
               fontsize=fontsize, fontweight=fontweight,
               verticalalignment='top', horizontalalignment='left')
