import csv, glob
import numpy as np
from collections import defaultdict

style_metrics = defaultdict(lambda: {
    "RSI_mean_raw": [], "RSI_median_raw": [], "RSI_p90_raw": [],
    "SCAR_mean_raw": [], "SCAR_median_raw": [], "SCAR_p90_raw": []
})
content_metrics = defaultdict(lambda: {
    "RSI_mean_raw": [], "RSI_median_raw": [], "RSI_p90_raw": [],
    "SCAR_mean_raw": [], "SCAR_median_raw": [], "SCAR_p90_raw": []
})

for path in glob.glob("csv/RSI_SCAR_content*.csv"):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            ci = int(row[0])
            si = int(row[1])
            rsi_vals = np.array([float(x) for x in row[2].split(';') if x])
            scar_vals = np.array([float(x) for x in row[4].split(';') if x])

            RSI_mean_raw   = rsi_vals.mean()
            RSI_median_raw = np.median(rsi_vals)
            RSI_p90_raw    = np.percentile(rsi_vals, 90)

            SCAR_mean_raw   = scar_vals.mean()
            SCAR_median_raw = np.median(scar_vals)
            SCAR_p90_raw    = np.percentile(scar_vals, 90)

            for metrics, idx in [(style_metrics, si), (content_metrics, ci)]:
                metrics[idx]["RSI_mean_raw"].append(RSI_mean_raw)
                metrics[idx]["RSI_median_raw"].append(RSI_median_raw)
                metrics[idx]["RSI_p90_raw"].append(RSI_p90_raw)
                metrics[idx]["SCAR_mean_raw"].append(SCAR_mean_raw)
                metrics[idx]["SCAR_median_raw"].append(SCAR_median_raw)
                metrics[idx]["SCAR_p90_raw"].append(SCAR_p90_raw)

def collapse(metrics_dict):
    collapsed = {}
    for idx, vals in metrics_dict.items():
        collapsed[idx] = {k: float(np.mean(v)) for k, v in vals.items()}
    return collapsed

style_avg = collapse(style_metrics)
content_avg = collapse(content_metrics)

def rank_block(name, avg_dict, metric, topn=10):
    top = sorted(((i, m[metric]) for i, m in avg_dict.items()), key=lambda x: -x[1])[:topn]
    bottom = sorted(((i, m[metric]) for i, m in avg_dict.items()), key=lambda x: x[1])[:topn]

    lines = []
    lines.append(f"=== Top {topn} {name} by {metric} ===")
    for idx, val in top:
        lines.append(f"{name[:-1]} {idx}: {val:.4f}")
    lines.append(f"=== Bottom {topn} {name} by {metric} ===")
    for idx, val in bottom:
        lines.append(f"{name[:-1]} {idx}: {val:.4f}")
    lines.append("")
    return "\n".join(lines)

metrics_list = ["RSI_mean_raw", "RSI_median_raw", "RSI_p90_raw",
                "SCAR_mean_raw", "SCAR_median_raw", "SCAR_p90_raw"]

with open("rankings_summary.txt", "w") as f:
    for metric in metrics_list:
        f.write(f"##### Rankings for {metric} #####\n\n")
        f.write(rank_block("Styles", style_avg, metric))
        f.write(rank_block("Contents", content_avg, metric))
        f.write("\n\n")

def write_csv(filename, name, avg_dict):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{name[:-1]}_idx"] + list(next(iter(avg_dict.values())).keys())
        writer.writerow(header)
        for idx, metrics in sorted(avg_dict.items()):
            writer.writerow([idx] + [f"{metrics[k]:.6f}" for k in header[1:]])

write_csv("style_metrics.csv", "Styles", style_avg)
write_csv("content_metrics.csv", "Contents", content_avg)
