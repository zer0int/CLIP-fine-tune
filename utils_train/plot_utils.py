from __future__ import annotations

import os
from typing import List, Dict

def _plot_dict_of_series(
    series_dict: Dict[str, List[float]],
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    use_log_scale: bool = True,
    topk: int = 40,
):
    if not series_dict:
        return
    
    import matplotlib.pyplot as plt
    
    # top-k by max value (keeps plot readable)
    sorted_layers = sorted(series_dict.items(), key=lambda item: (max(item[1]) if item[1] else 0.0), reverse=True)
    sorted_layers = sorted_layers[:max(1, int(topk))]

    plt.figure(figsize=(20, 10))
    for layer_name, norms in sorted_layers:
        plt.plot(norms, label=layer_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if use_log_scale:
        plt.yscale('log')
    plt.title(title)
    plt.legend(loc='upper right', fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_gradient_norms(gradient_norms_unscaled, gradient_rms_unscaled, epoch, plots_folder, topk=40):
    _plot_dict_of_series(
        gradient_norms_unscaled,
        out_path=os.path.join(plots_folder, f"grad_unscaled_e{epoch}.png"),
        title=f"Unscaled grad L2 norms (epoch {epoch})",
        xlabel="Logged step (sparse)",
        ylabel="L2 norm (unscaled)",
        use_log_scale=True,
        topk=topk,
    )
    _plot_dict_of_series(
        gradient_rms_unscaled,
        out_path=os.path.join(plots_folder, f"grad_rms_unscaled_e{epoch}.png"),
        title=f"Unscaled grad RMS (epoch {epoch})",
        xlabel="Logged step (sparse)",
        ylabel="RMS (unscaled)",
        use_log_scale=True,
        topk=topk,
    )

def plot_training_info(
    epoch_ids: List[int],
    training_losses: List[float],
    validation_losses: List[float],
    logits_diag_train: List[float],
    logits_off_train: List[float],
    logits_diag_val: List[float],
    logits_off_val: List[float],
    plots_folder: str,
):
    if not epoch_ids:
        return

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    if len(training_losses) == len(epoch_ids):
        plt.plot(epoch_ids, training_losses, label='Training Loss')
    if len(validation_losses) == len(epoch_ids):
        plt.plot(epoch_ids, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs (0-based epoch ids)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    
    # logits mean(diag) and mean(offdiag) (train/val)
    if len(logits_diag_train) == len(epoch_ids):
        plt.plot(epoch_ids, logits_diag_train, label='Train logits diag mean')
    if len(logits_off_train) == len(epoch_ids):
        plt.plot(epoch_ids, logits_off_train, label='Train logits offdiag mean')
    if len(logits_diag_val) == len(epoch_ids):
        plt.plot(epoch_ids, logits_diag_val, label='Val logits diag mean')
    if len(logits_off_val) == len(epoch_ids):
        plt.plot(epoch_ids, logits_off_val, label='Val logits offdiag mean')

    plt.title('Logit stats Over Epochs (diag vs offdiag)')
    plt.xlabel('Epoch')
    plt.ylabel('Logits')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "combined_training_plots.png"))
    plt.close()


def plot_teacher_info(epochs, teacher_cos, teacher_fit_cos, teacher_fit_mse, plots_folder, out_name="teacher_stats.png"):
    if not epochs:
        return

    import os
    import math
    import matplotlib.pyplot as plt

    # on resume -> dedupe repeated epochs (last write wins)
    def _dedupe(x, y):
        d = {}
        for xi, yi in zip(x, y):
            d[int(xi)] = float(yi)
        xs = sorted(d.keys())
        ys = [d[k] for k in xs]
        return xs, ys

    epochs_u = [int(e) for e in epochs]

    plt.figure(figsize=(12, 6))

    if teacher_cos is not None and len(teacher_cos) == len(epochs_u):
        x, y = _dedupe(epochs_u, teacher_cos)
        plt.plot(x, y, label="teacher_cos(val): cos(cls, c_hat)")

    if teacher_fit_cos is not None and len(teacher_fit_cos) == len(epochs_u):
        x, y = _dedupe(epochs_u, teacher_fit_cos)
        plt.plot(x, y, label="teacher_fit_cos(val): cos(C_true, C_pred)")

    if teacher_fit_mse is not None and len(teacher_fit_mse) == len(epochs_u):
        x, y = _dedupe(epochs_u, teacher_fit_mse)
        plt.plot(x, y, label="teacher_fit_mse(val)")

    plt.xlabel("Epoch (0-based; -1 is pre)")
    plt.ylabel("Value")
    plt.title("Teacher stats over epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, out_name))
    plt.close()


def plot_probe_info(epochs, lin_probe_accs, zs_accs, plots_folder):
    if not epochs:
        return
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    if lin_probe_accs:
        plt.plot(epochs, lin_probe_accs, label="linear_probe_acc")
    if zs_accs:
        plt.plot(epochs, zs_accs, label="zero_shot_acc")
    plt.xlabel("Epoch (0-based; -1 is pre)")
    plt.ylabel("Accuracy")
    plt.title("Quick probe over epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "quick_probe.png"))
    plt.close()


def plot_tiny_benchmark(history, plots_folder: str):
    """
    history:
      dict folder -> dict keys "epochs","acc","margin"
    """
    if not history:
        return

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    for folder, h in history.items():
        if h["epochs"]:
            plt.plot(h["epochs"], h["acc"], label=f"acc: {folder}")
    plt.xlabel("Epoch (0-based; -1 is pre)")
    plt.ylabel("Accuracy")
    plt.title("Tiny benchmark accuracy (bird vs bee vs text)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "tiny_benchmark_acc.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    for folder, h in history.items():
        if h["epochs"]:
            plt.plot(h["epochs"], h["margin"], label=f"margin: {folder}")
    plt.xlabel("Epoch (0-based; -1 is pre)")
    plt.ylabel("Mean margin (correct - best other)")
    plt.title("Tiny benchmark margin")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "tiny_benchmark_margin.png"))
    plt.close()