from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "cnn" / "plots"

DATASETS = (
    ("MNIST", "1 × 28 × 28", "32 × 14 × 14", "64 × 7 × 7", "128 × 3 × 3", "10 classes"),
    ("ORL", "3 × 112 × 92", "32 × 56 × 46", "64 × 28 × 23", "128 × 14 × 11", "40 classes"),
    ("CIFAR-10", "3 × 32 × 32", "32 × 16 × 16", "64 × 8 × 8", "128 × 4 × 4", "10 classes"),
)

COLORS = {
    "background": "#F5F7FB",
    "ink": "#172B4D",
    "muted": "#53657D",
    "input": "#E7F0FF",
    "conv": "#E7F7F2",
    "pool": "#D2F0E7",
    "pool_text": "#176B56",
    "global": "#FFF2D8",
    "dense": "#F0EAFE",
    "output": "#E6EDFF",
    "border": "#C7D3E3",
    "arrow": "#73839A",
    "accent": "#315F9B",
    "white": "#FFFFFF",
}


def add_card(
    ax: plt.Axes,
    y: float,
    height: float,
    facecolor: str,
    edgecolor: str = COLORS["border"],
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.75, y),
            14.5,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.16",
            linewidth=1.2,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
    )


def add_arrow(ax: plt.Axes, start_y: float, end_y: float) -> None:
    ax.annotate(
        "",
        xy=(8, end_y),
        xytext=(8, start_y),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["arrow"], "lw": 1.8},
    )


def add_feature_stage(
    ax: plt.Axes,
    *,
    y: float,
    number: int,
    channels: str,
    feature_index: int,
) -> None:
    height = 1.0
    add_card(ax, y, height, COLORS["conv"])
    ax.text(
        1.05,
        y + 0.69,
        f"BLOCK {number}",
        color=COLORS["accent"],
        fontsize=9,
        fontweight="bold",
        va="center",
    )
    ax.text(
        1.05,
        y + 0.36,
        "Convolution → ReLU → 2×2 max pool",
        color=COLORS["ink"],
        fontsize=11,
        fontweight="semibold",
        va="center",
    )
    ax.text(
        1.05,
        y + 0.12,
        f"3×3 kernel · stride 1 · padding 1 · {channels} output channels",
        color=COLORS["muted"],
        fontsize=8.5,
        va="center",
    )
    for column, dataset in enumerate(DATASETS):
        x = 6.05 + column * 2.95
        ax.text(
            x,
            y + 0.70,
            dataset[0],
            color=COLORS["muted"],
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            x,
            y + 0.36,
            dataset[feature_index],
            color=COLORS["ink"],
            fontsize=9.5,
            fontweight="semibold",
            ha="center",
            va="center",
        )
    ax.add_patch(
        FancyBboxPatch(
            (13.55, y + 0.27),
            1.35,
            0.44,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=0,
            facecolor=COLORS["pool"],
        )
    )
    ax.text(
        14.225,
        y + 0.49,
        "POOL",
        color=COLORS["pool_text"],
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )


def create_plot() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        0.8,
        9.62,
        "Convolutional Neural Network",
        fontsize=24,
        color=COLORS["ink"],
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.82,
        9.22,
        "Best test-performing configuration (deeper_adamw) · shared architecture, dataset-specific input/output sizes",
        fontsize=11,
        color=COLORS["muted"],
        va="center",
    )

    input_y = 8.18
    add_card(ax, input_y, 0.76, COLORS["input"])
    ax.text(
        1.05,
        input_y + 0.38,
        "INPUT",
        fontsize=10,
        color=COLORS["accent"],
        fontweight="bold",
        va="center",
    )
    for column, dataset in enumerate(DATASETS):
        x = 6.05 + column * 2.95
        ax.text(
            x,
            input_y + 0.51,
            dataset[0],
            color=COLORS["muted"],
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            x,
            input_y + 0.23,
            dataset[1],
            color=COLORS["ink"],
            fontsize=10,
            fontweight="semibold",
            ha="center",
            va="center",
        )
    add_arrow(ax, input_y, 7.98)

    stages = ((7.0, 1, "32", 2), (5.66, 2, "64", 3), (4.32, 3, "128", 4))
    for y, number, channels, shape_index in stages:
        add_feature_stage(
            ax, y=y, number=number, channels=channels, feature_index=shape_index
        )
    add_arrow(ax, 6.98, 6.66)
    add_arrow(ax, 5.64, 5.32)
    add_arrow(ax, 4.30, 3.94)

    add_card(ax, 3.08, 0.72, COLORS["global"])
    ax.text(
        1.05,
        3.44,
        "GLOBAL AVERAGE POOL",
        fontsize=10,
        color=COLORS["ink"],
        fontweight="bold",
        va="center",
    )
    ax.text(
        7.35,
        3.44,
        "128 × 1 × 1 feature map → 128-value vector",
        fontsize=10,
        color=COLORS["ink"],
        ha="center",
        va="center",
    )
    add_arrow(ax, 3.06, 2.88)

    add_card(ax, 2.02, 0.76, COLORS["dense"])
    ax.text(
        1.05,
        2.40,
        "CLASSIFIER",
        fontsize=10,
        color=COLORS["ink"],
        fontweight="bold",
        va="center",
    )
    ax.text(
        7.45,
        2.40,
        "Flatten → Linear 128→256 → ReLU → Linear 256→classes",
        fontsize=10.5,
        color=COLORS["ink"],
        ha="center",
        va="center",
    )
    add_arrow(ax, 2.00, 1.78)

    add_card(ax, 0.92, 0.76, COLORS["output"])
    ax.text(
        1.05,
        1.30,
        "OUTPUT LOGITS",
        fontsize=10,
        color=COLORS["accent"],
        fontweight="bold",
        va="center",
    )
    for column, dataset in enumerate(DATASETS):
        x = 6.05 + column * 2.95
        ax.text(
            x,
            1.30,
            f"{dataset[5]} · cross-entropy loss",
            color=COLORS["ink"],
            fontsize=9.5,
            ha="center",
            va="center",
        )

    ax.text(
        0.82,
        0.44,
        "Training setup: AdamW · learning rate 0.001 · 10 epochs · batch size 64",
        fontsize=9.5,
        color=COLORS["muted"],
        va="center",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    png_path = OUTPUT_DIR / "model_architecture.png"
    svg_path = OUTPUT_DIR / "model_architecture.svg"
    fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, svg_path


if __name__ == "__main__":
    for path in create_plot():
        print(f"Saved architecture diagram to {path}")
