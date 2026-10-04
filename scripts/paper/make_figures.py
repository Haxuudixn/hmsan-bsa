"""Render publication-ready PNG figures for the HMSAN-BSA paper materials.

Uses Pillow only (matplotlib is not installed in the project venv), so the same
script runs under both the project venv and the bundled runtime interpreter.

Figures
  fig_training_curves.png   train/val accuracy, macro-F1 and loss per epoch
  fig_class_distribution.png 19-class block distribution (log scale)
  fig_segment_size.png      blocks per training segment, cap = 3000
  fig_ocr_coverage.png      image-block OCR coverage by corpus batch
  fig_confusion_matrix.png  test-set confusion matrix (needs eval_test.json)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
]
EARLY_BATCH = {
    "training_data_2026-08-04 (1).zip",
    "training_data_2026-08-04.zip",
    "training_data_2026-08-08 (1).zip",
    "training_data_2026-08-08 (2).zip",
    "training_data_2026-08-08.zip",
    "training_data_2026-08-16 (2).zip",
    "training_data_2026-08-16(3).zip",
    "training_data_2026-08-16.zip",
    "training_data_2026-08-19.zip",
}
PALETTE = {
    "blue": (37, 99, 235),
    "orange": (234, 88, 12),
    "green": (22, 163, 74),
    "red": (220, 38, 38),
    "grey": (107, 114, 128),
    "light": (229, 231, 235),
    "ink": (17, 24, 39),
}


def load_font(size, bold=False):
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Chart:
    """Minimal plotting canvas: axes, grids, lines, bars, scatter."""

    def __init__(self, width=1500, height=950, margin=(110, 60, 60, 90)):
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.width, self.height = width, height
        self.left, self.top, self.right, self.bottom = margin
        self.font = load_font(22)
        self.small = load_font(18)
        self.title_font = load_font(28)
        self.plot_w = width - self.left - self.right
        self.plot_h = height - self.top - self.bottom

    def xy(self, x, y, xlim, ylim, rect=None):
        left, top, plot_w, plot_h = rect or (self.left, self.top, self.plot_w, self.plot_h)
        x0, x1 = xlim
        y0, y1 = ylim
        px = left + (x - x0) / (x1 - x0) * plot_w
        py = top + plot_h - (y - y0) / (y1 - y0) * plot_h
        return px, py

    def frame(self, xlim, ylim, xticks, yticks, xlabel="", ylabel="", rect=None,
              y_format="{:.2f}", tick_labels=None):
        left, top, plot_w, plot_h = rect or (self.left, self.top, self.plot_w, self.plot_h)
        d = self.draw
        d.rectangle([left, top, left + plot_w, top + plot_h], outline=PALETTE["ink"], width=2)
        for value in yticks:
            _, py = self.xy(xlim[0], value, xlim, ylim, rect)
            d.line([left, py, left + plot_w, py], fill=PALETTE["light"], width=1)
            label = y_format.format(value)
            d.text((left - 12 - d.textlength(label, font=self.small), py - 10),
                   label, font=self.small, fill=PALETTE["ink"])
        for idx, value in enumerate(xticks):
            px, _ = self.xy(value, ylim[0], xlim, ylim, rect)
            d.line([px, top + plot_h, px, top + plot_h + 6], fill=PALETTE["ink"], width=2)
            label = str(value if tick_labels is None else tick_labels[idx])
            d.text((px - d.textlength(label, font=self.small) / 2, top + plot_h + 12),
                   label, font=self.small, fill=PALETTE["ink"])
        if xlabel:
            d.text((left + plot_w / 2 - d.textlength(xlabel, font=self.font) / 2,
                    top + plot_h + 44), xlabel, font=self.font, fill=PALETTE["ink"])
        if ylabel:
            d.text((left - 78, top - 34), ylabel, font=self.font, fill=PALETTE["ink"])

    def series(self, xs, ys, color, xlim, ylim, rect=None, width=4, dashed=False, markers=True):
        points = [self.xy(x, y, xlim, ylim, rect) for x, y in zip(xs, ys) if y is not None]
        if dashed:
            for start, end in zip(points[:-1], points[1:]):
                self._dashed_line(start, end, color, width)
        elif len(points) > 1:
            self.draw.line(points, fill=color, width=width, joint="curve")
        if markers:
            for px, py in points:
                self.draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color)

    def _dashed_line(self, start, end, color, width, dash=12, gap=9):
        x0, y0 = start
        x1, y1 = end
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        steps = max(int(length // (dash + gap)), 1)
        for step in range(steps + 1):
            t0 = (step * (dash + gap)) / length
            t1 = min(t0 + dash / length, 1.0)
            if t0 >= 1.0:
                break
            self.draw.line(
                [x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                 x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1],
                fill=color, width=width,
            )

    def legend(self, entries, position, rect=None):
        left, top, plot_w, plot_h = rect or (self.left, self.top, self.plot_w, self.plot_h)
        d = self.draw
        x, y = position
        box_w = max(d.textlength(text, font=self.small) for _, text in entries) + 60
        box_h = 30 * len(entries) + 14
        d.rectangle([x, y, x + box_w, y + box_h], fill="white", outline=PALETTE["grey"])
        for idx, (color, text) in enumerate(entries):
            cy = y + 22 + idx * 30
            d.line([x + 14, cy, x + 44, cy], fill=color, width=4)
            d.ellipse([x + 24, cy - 5, x + 34, cy + 5], fill=color)
            d.text((x + 52, cy - 11), text, font=self.small, fill=PALETTE["ink"])

    def title(self, text, rect=None):
        left, top, plot_w, _ = rect or (self.left, self.top, self.plot_w, self.plot_h)
        self.draw.text((left + plot_w / 2 - self.draw.textlength(text, font=self.title_font) / 2, 18),
                       text, font=self.title_font, fill=PALETTE["ink"])

    def save(self, path):
        self.image.save(path, dpi=(200, 200))
        print(f"  wrote {path}")


def nice_ticks(low, high, count=5, integer=True):
    if high <= low:
        high = low + 1
    step = (high - low) / count
    if integer:
        step = max(1, math.ceil(step))
        low = math.floor(low / step) * step
        return [low + step * i for i in range(int((high - low) / step) + 1)]
    return [low + step * i for i in range(count + 1)]


def fig_training_curves(history, out_dir):
    epochs = [entry["epoch"] for entry in history]
    train_acc = [entry["train"]["accuracy"] for entry in history]
    val_acc = [entry["val"]["accuracy"] for entry in history]
    train_f1 = [entry["train"]["macro_f1"] for entry in history]
    val_f1 = [entry["val"]["macro_f1"] for entry in history]
    train_loss = [entry["train"]["loss"] for entry in history]
    val_loss = [entry["val"]["val_loss"] for entry in history]

    chart = Chart(width=1800, height=880, margin=(120, 70, 60, 100))
    chart.title("Training dynamics (8 epochs, train 303 / val 65 segments)")

    half = (chart.plot_w - 110) / 2
    rect_a = (chart.left, chart.top, half, chart.plot_h)
    rect_b = (chart.left + half + 110, chart.top, half, chart.plot_h)

    xlim = (min(epochs) - 0.3, max(epochs) + 0.3)
    ylim = (0.7, 1.0)
    ticks = [0.7 + 0.05 * i for i in range(7)]
    chart.frame(xlim, ylim, epochs, ticks, xlabel="epoch", ylabel="accuracy / macro-F1",
                rect=rect_a, tick_labels=None)
    chart.series(epochs, train_acc, PALETTE["blue"], xlim, ylim, rect_a, dashed=True)
    chart.series(epochs, val_acc, PALETTE["blue"], xlim, ylim, rect_a)
    chart.series(epochs, train_f1, PALETTE["green"], xlim, ylim, rect_a, dashed=True)
    chart.series(epochs, val_f1, PALETTE["green"], xlim, ylim, rect_a)
    chart.legend([(PALETTE["blue"], "train / val accuracy"),
                  (PALETTE["green"], "train / val macro-F1")],
                 (chart.left + 40, chart.top + 24), rect_a)

    loss_high = max(max(train_loss), max(val_loss)) * 1.1
    loss_ticks = nice_ticks(0, loss_high, 5, integer=False)
    chart.frame(xlim, (0, loss_ticks[-1]), epochs, loss_ticks, xlabel="epoch",
                ylabel="loss", rect=rect_b)
    chart.series(epochs, train_loss, PALETTE["orange"], xlim, (0, loss_ticks[-1]), rect_b, dashed=True)
    chart.series(epochs, val_loss, PALETTE["red"], xlim, (0, loss_ticks[-1]), rect_b)
    chart.legend([(PALETTE["orange"], "train total loss"),
                  (PALETTE["red"], "val loss")],
                 (chart.left + half + 150, chart.top + 24), rect_b)
    chart.save(out_dir / "fig_training_curves.png")


def fig_class_distribution(classes, out_dir):
    classes = sorted(classes, key=lambda entry: -entry["blocks"])
    chart = Chart(width=1600, height=1000, margin=(330, 70, 90, 90))
    chart.title("Block-level class distribution (456,602 blocks, 19 classes)")
    log_high = math.log10(max(entry["blocks"] for entry in classes)) + 0.15
    bar_h = (chart.plot_h - 20) / len(classes) - 6
    x_ticks = [1, 10, 100, 1000, 10000, 100000]
    for value in x_ticks:
        px, _ = chart.xy(math.log10(value), 0, (0, log_high), (0, 1))
        chart.draw.line([px, chart.top, px, chart.top + chart.plot_h],
                        fill=PALETTE["light"], width=1)
        label = f"{value:,}"
        chart.draw.text((px - chart.draw.textlength(label, font=chart.small) / 2,
                         chart.top + chart.plot_h + 10), label,
                        font=chart.small, fill=PALETTE["ink"])
    chart.draw.rectangle([chart.left, chart.top, chart.left + chart.plot_w,
                          chart.top + chart.plot_h], outline=PALETTE["ink"], width=2)
    for idx, entry in enumerate(classes):
        y = chart.top + 10 + idx * (bar_h + 6)
        px, _ = chart.xy(math.log10(max(entry["blocks"], 1)), 0, (0, log_high), (0, 1))
        chart.draw.rectangle([chart.left, y, px, y + bar_h], fill=PALETTE["blue"])
        chart.draw.text((chart.left - 300, y + bar_h / 2 - 11),
                        f"{entry['id']:2d} {entry['label']}", font=chart.small,
                        fill=PALETTE["ink"])
        count = f"{entry['blocks']:,}"
        chart.draw.text((px + 8, y + bar_h / 2 - 11), count, font=chart.small,
                        fill=PALETTE["ink"])
    chart.draw.text((chart.left + chart.plot_w / 2 - 60,
                     chart.top + chart.plot_h + 44),
                    "blocks (log scale)", font=chart.font, fill=PALETTE["ink"])
    chart.save(out_dir / "fig_class_distribution.png")


def fig_segment_size(segmentation, out_dir, cap=3000):
    stats = segmentation["blocks_per_segment"]
    chart = Chart(width=1500, height=880, margin=(130, 70, 70, 100))
    chart.title("Segments after page-range capping (cap = 3,000 blocks / 150 pages)")

    values = [stats["p50"], stats["p90"], stats["p99"], stats["max"]]
    labels = ["p50", "p90", "p99", "max"]
    extra = [f"mean {stats['mean']:.0f}", f"min {stats['min']:.0f}"]
    xlim = (0, len(values))
    ylim = (0, max(values) * 1.15)
    yticks = nice_ticks(0, ylim[1], 5)
    chart.frame(xlim, ylim, list(range(len(values))), yticks,
                xlabel="percentile over 434 segments", ylabel="blocks per segment",
                tick_labels=labels, y_format="{:.0f}")
    for idx, value in enumerate(values):
        x0, y0 = chart.xy(idx + 0.2, 0, xlim, ylim)
        x1, y1 = chart.xy(idx + 0.8, value, xlim, ylim)
        color = PALETTE["red"] if idx == 3 else PALETTE["blue"]
        chart.draw.rectangle([x0, y1, x1, y0], fill=color)
        chart.draw.text(((x0 + x1) / 2 - chart.draw.textlength(f"{value:.0f}", font=chart.small) / 2,
                         y1 - 28), f"{value:.0f}", font=chart.small, fill=PALETTE["ink"])
    _, cap_y = chart.xy(0, cap, xlim, ylim)
    chart.draw.line([chart.left, cap_y, chart.left + chart.plot_w, cap_y],
                    fill=PALETTE["green"], width=3)
    chart.draw.text((chart.left + chart.plot_w - 260, cap_y - 30),
                    "cap 3,000 blocks", font=chart.small, fill=PALETTE["green"])
    for idx, text in enumerate(extra):
        chart.draw.text((chart.left + 10, chart.top + 12 + idx * 26), text,
                        font=chart.small, fill=PALETTE["grey"])
    chart.save(out_dir / "fig_segment_size.png")


def fig_ocr_coverage(packages, out_dir):
    groups = {"早期包（08-04 ~ 08-19，9 包）": {"with": 0, "without": 0},
              "OCR 补齐包（08-27 ~ 09-17，29 包）": {"with": 0, "without": 0}}
    for entry in packages:
        key = ("早期包（08-04 ~ 08-19，9 包）" if entry["zipname"] in EARLY_BATCH
               else "OCR 补齐包（08-27 ~ 09-17，29 包）")
        groups[key]["with"] += entry["ocr_filled"]
        groups[key]["without"] += entry["image"] + entry["mixed"] - entry["ocr_filled"]

    chart = Chart(width=1500, height=880, margin=(140, 70, 70, 110))
    chart.title("OCR coverage of image-type blocks by corpus batch")
    names = list(groups)
    totals = [groups[name]["with"] + groups[name]["without"] for name in names]
    ylim = (0, max(totals) * 1.2)
    yticks = nice_ticks(0, ylim[1], 5)
    chart.frame((0, len(names)), ylim, list(range(len(names))), yticks,
                xlabel="", ylabel="image-type blocks", tick_labels=names,
                y_format="{:.0f}")
    for idx, name in enumerate(names):
        with_ocr = groups[name]["with"]
        without = groups[name]["without"]
        x0, y0 = chart.xy(idx + 0.25, 0, (0, len(names)), ylim)
        x1, y1 = chart.xy(idx + 0.5, with_ocr, (0, len(names)), ylim)
        x2, y2 = chart.xy(idx + 0.75, with_ocr + without, (0, len(names)), ylim)
        chart.draw.rectangle([x0, y1, x1, y0], fill=PALETTE["green"])
        chart.draw.rectangle([x1, y2, x2, y0], fill=PALETTE["light"], outline=PALETTE["grey"])
        ratio = with_ocr / max(with_ocr + without, 1)
        chart.draw.text((x0 - 10, y1 - 32), f"{with_ocr:,}", font=chart.small, fill=PALETTE["green"])
        chart.draw.text((x1 + 6, y2 - 32), f"{without:,}", font=chart.small, fill=PALETTE["grey"])
        chart.draw.text((x0 - 10, y0 + 14), f"OCR 覆盖 {ratio*100:.1f}%", font=chart.small,
                        fill=PALETTE["ink"])
    chart.legend([(PALETTE["green"], "image blocks with OCR text"),
                  (PALETTE["light"], "image blocks without OCR text")],
                 (chart.left + 40, chart.top + 20))
    chart.save(out_dir / "fig_ocr_coverage.png")


def fig_confusion_matrix(eval_data, out_dir, model_key):
    entry = next((item for item in eval_data["models"] if item["checkpoint"] == model_key), None)
    if entry is None:
        print(f"  skip confusion matrix: {model_key} not in eval report")
        return
    matrix = entry["confusion_matrix"]
    labels = eval_data["class_labels"]
    n = len(labels)
    cell = 62
    margin_left, margin_top = 300, 120
    chart = Chart(width=margin_left + n * cell + 80, height=margin_top + n * cell + 80,
                  margin=(margin_left, margin_top, 60, 60))
    chart.title(f"Test-set confusion matrix (row-normalised) - {model_key}")
    row_sums = [max(sum(row), 1) for row in matrix]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ratio = value / row_sums[i]
            base = (255, 255, 255)
            shade = (37, 99, 235)
            color = tuple(int(base[k] + (shade[k] - base[k]) * (ratio ** 0.6)) for k in range(3))
            x0 = margin_left + j * cell
            y0 = margin_top + i * cell
            chart.draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color,
                                 outline=PALETTE["light"])
            if value:
                text = f"{ratio*100:.0f}" if ratio >= 0.005 else "."
                ink = "white" if ratio > 0.55 else PALETTE["ink"]
                chart.draw.text((x0 + cell / 2 - chart.draw.textlength(text, font=chart.small) / 2,
                                 y0 + cell / 2 - 11), text, font=chart.small, fill=ink)
        chart.draw.text((margin_left - 290, margin_top + i * cell + cell / 2 - 11),
                        f"{i:2d} {labels[i]}", font=chart.small, fill=PALETTE["ink"])
    for j, label in enumerate(labels):
        x0 = margin_left + j * cell
        for depth, char in enumerate(label):
            chart.draw.text((x0 + cell / 2 - 9, margin_top - 26 * (len(label) - depth) - 6),
                            char, font=chart.small, fill=PALETTE["ink"])
    chart.save(out_dir / f"fig_confusion_matrix_{model_key.replace('.pt','')}.png")


def main():
    parser = argparse.ArgumentParser(description="render paper figures")
    parser.add_argument("--stats", default="docs/paper_materials/dataset_stats.json")
    parser.add_argument("--history", default="outputs/hmsan_bsa_final/history.json")
    parser.add_argument("--eval", dest="eval_path", default="docs/paper_materials/eval_test.json")
    parser.add_argument("--out_dir", default="docs/paper_materials/figures")
    parser.add_argument("--confusion_model", default="best_model.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    history = json.loads(Path(args.history).read_text(encoding="utf-8"))

    fig_training_curves(history, out_dir)
    fig_class_distribution(stats["classes"], out_dir)
    fig_segment_size(stats["segmentation"], out_dir,
                     cap=stats["segmentation"]["max_blocks_per_sample"])
    fig_ocr_coverage(stats["packages"], out_dir)

    eval_file = Path(args.eval_path)
    if eval_file.exists():
        fig_confusion_matrix(json.loads(eval_file.read_text(encoding="utf-8")),
                             out_dir, args.confusion_model)
    else:
        print(f"  no eval report at {eval_file}; confusion matrix skipped")


if __name__ == "__main__":
    main()