import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

LEGEND_FONTSIZE = 15
LEGEND_MAX_COLUMNS = 4
LEGEND_BELOW_ANCHOR = (0.5, -0.18)

BEIGE = "#F5F5F0"
CARDINAL = "#8C1C13"
BLUSHED_BRICK = "#BF4342"
DUSTY_TAUPE = "#A78A7F"
MAUVE_BARK = "#735751"
CORAL = "#E26D5C"
BROWN = "#6B4226"
INK = "#2A211E"

CATEGORICAL_CYCLE = [CARDINAL, DUSTY_TAUPE, MAUVE_BARK, BLUSHED_BRICK]

CLASS_CYCLE = [CARDINAL, DUSTY_TAUPE, CORAL, MAUVE_BARK, BLUSHED_BRICK]


def class_color(class_index: int) -> str:
    return CLASS_CYCLE[class_index % len(CLASS_CYCLE)]

SERIF_PREFERENCE = [
    "New Computer Modern",
    "NewComputerModern10",
    "CMU Serif",
    "Latin Modern Roman",
    "STIX Two Text",
    "STIXGeneral",
    "cmr10",
    "DejaVu Serif",
]

SEQUENTIAL_CMAP_NAME = "house_warm"
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    SEQUENTIAL_CMAP_NAME, [BEIGE, DUSTY_TAUPE, BLUSHED_BRICK, CARDINAL]
)

HOUSE_RC = {
    "font.family": "serif",
    "font.serif": SERIF_PREFERENCE,
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": LEGEND_FONTSIZE,
    "legend.title_fontsize": LEGEND_FONTSIZE + 1,
    "figure.titlesize": 22,
    "figure.titleweight": "bold",
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "axes.linewidth": 1.2,
    "figure.facecolor": BEIGE,
    "axes.facecolor": BEIGE,
    "savefig.facecolor": BEIGE,
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": DUSTY_TAUPE,
    "grid.alpha": 0.35,
    "legend.frameon": False,
    "axes.prop_cycle": cycler(color=CATEGORICAL_CYCLE),
    "figure.dpi": 120,
    "savefig.dpi": 300,
}


def apply_house_style():
    if SEQUENTIAL_CMAP_NAME not in matplotlib.colormaps:
        matplotlib.colormaps.register(SEQUENTIAL_CMAP)
    matplotlib.rcParams.update(HOUSE_RC)


def sentence_case(text: str) -> str:
    def is_technical(token: str) -> bool:
        letters = "".join(character for character in token if character.isalpha())
        if not letters:
            return True
        if "$" in token or "_" in token or any(c.isdigit() for c in token):
            return True
        if letters.isupper():
            return True
        return any(character.isupper() for character in letters[1:])

    sentence_started = False
    cased_tokens = []

    def cased(token: str) -> str:
        nonlocal sentence_started
        if is_technical(token):
            if any(character.isalpha() for character in token):
                sentence_started = True
            return token
        lowered = token.lower()
        if sentence_started:
            return lowered
        for position, character in enumerate(lowered):
            if character.isalpha():
                sentence_started = True
                return lowered[:position] + character.upper() + lowered[position + 1 :]
        return lowered

    for token in text.split(" "):
        cased_tokens.append(cased(token))
    return " ".join(cased_tokens)


def finalize_house_style(fig: Figure):
    def relocate_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        existing = ax.get_legend()
        if not handles:
            if existing is not None:
                existing.remove()
            return
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=LEGEND_BELOW_ANCHOR,
            ncol=min(len(labels), LEGEND_MAX_COLUMNS),
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )

    def restyle(ax):
        if ax.get_title():
            ax.set_title(sentence_case(ax.get_title()))
        if ax.get_xlabel():
            ax.set_xlabel(sentence_case(ax.get_xlabel()))
        if ax.get_ylabel():
            ax.set_ylabel(sentence_case(ax.get_ylabel()))
        relocate_legend(ax)

    for ax in fig.axes:
        restyle(ax)

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        suptitle.set_text(sentence_case(suptitle.get_text()))
