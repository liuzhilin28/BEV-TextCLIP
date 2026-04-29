import re
from pathlib import Path
import matplotlib.pyplot as plt

# 改这里：优先读你单独保存的本次日志
LOG_PATH = Path("logs/train.log")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

train_pat = re.compile(r"Epoch\s+(\d+)\s*-\s*Avg Loss:\s*([0-9]+(?:\.[0-9]+)?)")
val_pat_new = re.compile(
    r"Val Loss:\s*([0-9]+(?:\.[0-9]+)?),\s*Val Accuracy:\s*([0-9]+(?:\.[0-9]+)?),\s*Val mIoU:\s*([0-9]+(?:\.[0-9]+)?)"
)
val_pat_old = re.compile(
    r"Val Loss:\s*([0-9]+(?:\.[0-9]+)?),\s*Val Accuracy:\s*([0-9]+(?:\.[0-9]+)?)"
)

def parse_log(log_path: Path):
    epochs = []
    train_loss = []
    val_loss = []
    val_acc = []
    val_miou = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_train = train_pat.search(line)
            if m_train:
                epochs.append(int(m_train.group(1)))
                train_loss.append(float(m_train.group(2)))
                continue

            m_val_new = val_pat_new.search(line)
            if m_val_new:
                val_loss.append(float(m_val_new.group(1)))
                val_acc.append(float(m_val_new.group(2)))
                val_miou.append(float(m_val_new.group(3)))
                continue

            m_val_old = val_pat_old.search(line)
            if m_val_old:
                val_loss.append(float(m_val_old.group(1)))
                val_acc.append(float(m_val_old.group(2)))
                val_miou.append(None)
                continue

    if not epochs:
        raise RuntimeError(f"没有从 {log_path} 里解析到 epoch 数据，请检查日志路径或日志格式。")

    n = min(len(epochs), len(train_loss), len(val_loss), len(val_acc))
    epochs = epochs[:n]
    train_loss = train_loss[:n]
    val_loss = val_loss[:n]
    val_acc = val_acc[:n]
    val_miou = val_miou[:n]

    return epochs, train_loss, val_loss, val_acc, val_miou

def plot_curve(x, y, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"找不到日志文件: {LOG_PATH}")

    epochs, train_loss, val_loss, val_acc, val_miou = parse_log(LOG_PATH)

    plot_curve(epochs, train_loss, "Train Loss vs Epoch", "Train Loss", OUT_DIR / "train_loss.png")
    plot_curve(epochs, val_loss, "Val Loss vs Epoch", "Val Loss", OUT_DIR / "val_loss.png")
    plot_curve(epochs, val_acc, "Val Accuracy vs Epoch", "Val Accuracy", OUT_DIR / "val_accuracy.png")

    if any(v is not None for v in val_miou):
        x_miou = []
        y_miou = []
        for e, m in zip(epochs, val_miou):
            if m is not None:
                x_miou.append(e)
                y_miou.append(m)
        if x_miou:
            plot_curve(x_miou, y_miou, "Val mIoU vs Epoch", "Val mIoU", OUT_DIR / "val_miou.png")

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")
    plt.plot(epochs, val_acc, marker="o", label="Val Accuracy")

    if any(v is not None for v in val_miou):
        x_miou = []
        y_miou = []
        for e, m in zip(epochs, val_miou):
            if m is not None:
                x_miou.append(e)
                y_miou.append(m)
        if x_miou:
            plt.plot(x_miou, y_miou, marker="o", label="Val mIoU")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics Overview")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "metrics_overview.png", dpi=200)
    plt.close()

    print(f"已保存到: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()