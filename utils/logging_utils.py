"""학습 진행 로그: 콘솔 + 파일 + latest_status.json (로컬/EC2 공통)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import config


class TrainLogger:
    def __init__(self, model_type: str, run_name: str | None = None):
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_type = model_type
        self.run_name = run_name or f"{model_type}_{stamp}"
        self.log_path = config.LOG_DIR / f"train_{self.run_name}.log"
        self.status_path = config.LOG_DIR / "latest_status.json"
        self._file = self.log_path.open("a", encoding="utf-8")
        self._write(f"=== Train start: {self.run_name} ===")
        self._write(f"Device profile: {config.runtime_profile()}")

    def _write(self, msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def info(self, msg: str):
        self._write(msg)

    def batch_progress(
        self,
        epoch: int,
        total_epochs: int,
        batch_idx: int,
        num_batches: int,
        train_loss: float,
        best_val_acc: float = 0.0,
    ):
        """epoch 도중 latest_status.json 갱신 (CustomCNN 등 긴 epoch용)."""
        pct_epoch = 100.0 * batch_idx / max(num_batches, 1)
        status = {
            "updated_at": datetime.now().isoformat(),
            "model": self.model_type,
            "run_name": self.run_name,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch": batch_idx,
            "total_batches": num_batches,
            "epoch_progress_pct": round(pct_epoch, 1),
            "progress_pct": round(
                100.0 * ((epoch - 1) + batch_idx / max(num_batches, 1)) / max(total_epochs, 1),
                1,
            ),
            "train_loss_running": round(train_loss, 6),
            "best_val_acc": round(best_val_acc, 6),
            "log_file": str(self.log_path),
        }
        self.status_path.write_text(
            json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if batch_idx == 1 or batch_idx == num_batches or batch_idx % 50 == 0:
            self._write(
                f"Epoch {epoch:02d}/{total_epochs} | batch {batch_idx}/{num_batches} "
                f"({pct_epoch:.0f}%) | loss {train_loss:.4f}"
            )

    def epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        best_val_acc: float,
        extra: str = "",
    ):
        tag = f" {extra}" if extra else ""
        self._write(
            f"Epoch {epoch:02d}/{total_epochs}{tag} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
            f"Best Val {best_val_acc:.4f}"
        )
        status = {
            "updated_at": datetime.now().isoformat(),
            "model": self.model_type,
            "run_name": self.run_name,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "progress_pct": round(100.0 * epoch / max(total_epochs, 1), 1),
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "best_val_acc": round(best_val_acc, 6),
            "log_file": str(self.log_path),
        }
        self.status_path.write_text(
            json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def checkpoint_saved(self, val_acc: float, train_acc: float):
        self._write(f"  >> checkpoint saved (Val Acc {val_acc:.4f}, Train Acc {train_acc:.4f})")

    def close(self, best_val_acc: float):
        self._write(f"=== Train done. Best Val Acc: {best_val_acc:.4f} ===")
        self._file.close()
