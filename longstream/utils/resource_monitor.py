"""
longstream/utils/resource_monitor.py

轻量级后台资源监控器。
推理开始前调用 monitor.start(output_dir)，结束后调用 monitor.stop()。

在 <output_dir>/resource_monitor/ 下生成三个 CSV：
  cpu_timeseries.csv   — 系统+进程 CPU/RAM 时序（论文绘图用）
  gpu_timeseries.csv   — GPU 利用率/显存/功耗时序（论文绘图用，依赖 nvidia-smi）
  cpu_process_top.csv  — 每采样周期 CPU 占用最高的前 N 个进程（debug 用）

依赖：psutil（可选）；nvidia-smi CLI（可选）；不引入 torch / CUDA tensor。
"""

from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from datetime import datetime

_PSUTIL_AVAILABLE = False
try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass


class ResourceMonitor:
    """
    后台资源监控器，使用 daemon 线程运行，不阻塞主进程退出。

    用法::

        monitor = ResourceMonitor(interval_sec=2.0)
        monitor.start("/path/to/output")
        try:
            run_inference(...)
        finally:
            monitor.stop()

    或者通过工厂函数从 YAML 配置创建::

        monitor = from_cfg(cfg.get("monitoring", {}))
        if monitor is not None:
            monitor.start(output_dir)
    """

    def __init__(
        self,
        interval_sec: float = 2.0,
        top_process_count: int = 10,
        write_process_top: bool = True,
        write_cpu_timeseries: bool = True,
        write_gpu_timeseries: bool = True,
    ):
        self.interval_sec = float(interval_sec)
        self.top_process_count = int(top_process_count)
        self.write_process_top = bool(write_process_top)
        self.write_cpu_timeseries = bool(write_cpu_timeseries)
        self.write_gpu_timeseries = bool(write_gpu_timeseries)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_dir: str | None = None

    # ------------------------------------------------------------------ #
    #  public                                                              #
    # ------------------------------------------------------------------ #

    def start(self, output_dir: str) -> None:
        """启动监控线程。若已在运行则忽略。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._output_dir = output_dir
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="ResourceMonitor", daemon=True
        )
        self._thread.start()
        print(
            f"[ResourceMonitor] 已启动，采样间隔 {self.interval_sec}s，"
            f"输出目录: {os.path.join(output_dir, 'resource_monitor')}",
            flush=True,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """停止监控线程，等待最多 timeout 秒。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        print("[ResourceMonitor] 已停止。", flush=True)

    # ------------------------------------------------------------------ #
    #  internal                                                            #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        out_dir = os.path.join(self._output_dir, "resource_monitor")
        os.makedirs(out_dir, exist_ok=True)

        cpu_file = gpu_file = proc_file = None
        cpu_writer = gpu_writer = proc_writer = None
        start_time = time.monotonic()

        try:
            # --- open CSV files ---
            if self.write_cpu_timeseries and _PSUTIL_AVAILABLE:
                cpu_file = open(
                    os.path.join(out_dir, "cpu_timeseries.csv"),
                    "w", newline="", buffering=1,
                )
                cpu_writer = csv.writer(cpu_file)
                cpu_writer.writerow([
                    "timestamp", "elapsed_s",
                    "cpu_total_percent", "cpu_user_percent", "cpu_system_percent",
                    "cpu_iowait_percent",
                    "load1", "load5", "load15",
                    "ram_used_gb", "ram_percent",
                    "swap_used_gb", "swap_percent",
                    "process_cpu_percent", "process_rss_mb", "process_threads",
                ])

            if self.write_gpu_timeseries:
                gpu_file = open(
                    os.path.join(out_dir, "gpu_timeseries.csv"),
                    "w", newline="", buffering=1,
                )
                gpu_writer = csv.writer(gpu_file)
                gpu_writer.writerow([
                    "timestamp", "elapsed_s",
                    "gpu_index", "gpu_name",
                    "gpu_util_percent", "mem_util_percent",
                    "mem_used_mb", "mem_total_mb",
                    "temperature_c", "power_w",
                ])

            if self.write_process_top and _PSUTIL_AVAILABLE:
                proc_file = open(
                    os.path.join(out_dir, "cpu_process_top.csv"),
                    "w", newline="", buffering=1,
                )
                proc_writer = csv.writer(proc_file)
                proc_writer.writerow([
                    "timestamp", "elapsed_s",
                    "pid", "ppid", "name", "username",
                    "cpu_percent", "rss_mb", "mem_percent",
                    "num_threads", "cmdline",
                ])

            # Prime psutil cpu_percent (first call always returns 0.0)
            if _PSUTIL_AVAILABLE:
                psutil.cpu_percent(interval=None, percpu=False)
                psutil.cpu_times_percent(interval=None)
                try:
                    self_proc = psutil.Process()
                    self_proc.cpu_percent(interval=None)
                except Exception:
                    self_proc = None
            else:
                self_proc = None

            # --- sampling loop ---
            while not self._stop_event.wait(timeout=self.interval_sec):
                ts = datetime.now().isoformat(timespec="seconds")
                elapsed = round(time.monotonic() - start_time, 1)

                if cpu_writer is not None:
                    self._write_cpu_row(cpu_writer, ts, elapsed, self_proc)

                if gpu_writer is not None:
                    self._write_gpu_rows(gpu_writer, ts, elapsed)

                if proc_writer is not None:
                    self._write_proc_top(proc_writer, ts, elapsed)

        finally:
            for f in (cpu_file, gpu_file, proc_file):
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    #  row writers                                                         #
    # ------------------------------------------------------------------ #

    def _write_cpu_row(self, writer, ts: str, elapsed: float, self_proc) -> None:
        try:
            cpu_pct = psutil.cpu_percent(interval=None, percpu=False)
            times = psutil.cpu_times_percent(interval=None)
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            try:
                load1, load5, load15 = psutil.getloadavg()
                load1 = round(load1, 2)
                load5 = round(load5, 2)
                load15 = round(load15, 2)
            except AttributeError:
                # Windows does not support getloadavg
                load1 = load5 = load15 = ""

            proc_cpu = proc_rss = proc_threads = ""
            if self_proc is not None:
                try:
                    proc_cpu = round(self_proc.cpu_percent(interval=None), 1)
                    mi = self_proc.memory_info()
                    proc_rss = round(mi.rss / 1024 ** 2, 1)
                    proc_threads = self_proc.num_threads()
                except Exception:
                    pass

            writer.writerow([
                ts, elapsed,
                round(cpu_pct, 1),
                round(getattr(times, "user", 0), 1),
                round(getattr(times, "system", 0), 1),
                round(getattr(times, "iowait", 0), 1),
                load1, load5, load15,
                round(mem.used / 1024 ** 3, 2),
                round(mem.percent, 1),
                round(swap.used / 1024 ** 3, 2),
                round(swap.percent, 1),
                proc_cpu, proc_rss, proc_threads,
            ])
        except Exception:
            pass

    def _write_gpu_rows(self, writer, ts: str, elapsed: float) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                writer.writerow([ts, elapsed] + parts[:8])
        except Exception:
            pass

    def _write_proc_top(self, writer, ts: str, elapsed: float) -> None:
        try:
            procs = []
            for p in psutil.process_iter(
                ["pid", "ppid", "name", "username", "cpu_percent",
                 "memory_info", "memory_percent", "num_threads"]
            ):
                try:
                    procs.append(p.info)
                except Exception:
                    pass

            procs.sort(
                key=lambda x: float(x.get("cpu_percent") or 0),
                reverse=True,
            )

            for info in procs[: self.top_process_count]:
                try:
                    p_obj = psutil.Process(info["pid"])
                    try:
                        cmdline = " ".join(p_obj.cmdline())[:200]
                    except Exception:
                        cmdline = ""

                    rss_mb = ""
                    mi = info.get("memory_info")
                    if mi is not None:
                        rss_mb = round(mi.rss / 1024 ** 2, 1)

                    writer.writerow([
                        ts, elapsed,
                        info.get("pid", ""),
                        info.get("ppid", ""),
                        info.get("name", ""),
                        info.get("username", ""),
                        round(float(info.get("cpu_percent") or 0), 1),
                        rss_mb,
                        round(float(info.get("memory_percent") or 0), 2),
                        info.get("num_threads", ""),
                        cmdline,
                    ])
                except Exception:
                    pass
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  factory                                                             #
# ------------------------------------------------------------------ #

def from_cfg(monitoring_cfg: dict) -> "ResourceMonitor | None":
    """
    从 YAML monitoring 配置字典创建 ResourceMonitor。
    若 enabled=false 或 psutil 不可用则返回 None。

    期望的 YAML 结构::

        monitoring:
          enabled: true
          interval_sec: 2.0
          top_process_count: 10
          write_process_top: true
          write_cpu_timeseries: true
          write_gpu_timeseries: true
    """
    if not bool(monitoring_cfg.get("enabled", False)):
        return None
    if not _PSUTIL_AVAILABLE and not monitoring_cfg.get("write_gpu_timeseries", True):
        print(
            "[ResourceMonitor] 警告：psutil 未安装，CPU 监控不可用。"
            "请运行: pip install psutil",
            flush=True,
        )
        return None
    return ResourceMonitor(
        interval_sec=float(monitoring_cfg.get("interval_sec", 2.0)),
        top_process_count=int(monitoring_cfg.get("top_process_count", 10)),
        write_process_top=bool(monitoring_cfg.get("write_process_top", True)),
        write_cpu_timeseries=bool(monitoring_cfg.get("write_cpu_timeseries", True)),
        write_gpu_timeseries=bool(monitoring_cfg.get("write_gpu_timeseries", True)),
    )


# ------------------------------------------------------------------ #
#  critical operation profiler                                         #
# ------------------------------------------------------------------ #

class CriticalOperationProfiler:
    """
    轻量级上下文管理器，记录代码块的挂钟时间（Wall Time）及进程级 CPU 时间。

    结果追加写入 <output_dir>/resource_monitor/critical_ops.csv，
    与 ResourceMonitor 的时序 CSV 共用同一目录，便于对齐分析。

    用法::

        CriticalOperationProfiler.initialize(output_root)
        with CriticalOperationProfiler("run_inference_cfg"):
            run_inference_cfg(cfg)
    """

    _output_path: "str | None" = None
    _lock = threading.Lock()
    _is_initialized = False

    @classmethod
    def initialize(cls, output_dir: str) -> None:
        """初始化 CSV 文件（写入表头）。应在推理开始前调用一次。"""
        os.makedirs(os.path.join(output_dir, "resource_monitor"), exist_ok=True)
        cls._output_path = os.path.join(
            output_dir, "resource_monitor", "critical_ops.csv"
        )
        with cls._lock:
            if not cls._is_initialized:
                with open(cls._output_path, "w", newline="", buffering=1) as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["operation", "duration_sec", "cpu_user_sec", "cpu_sys_sec"]
                    )
                cls._is_initialized = True

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = 0.0
        self.start_cpu = None

    def __enter__(self) -> "CriticalOperationProfiler":
        self.start_time = time.perf_counter()
        if _PSUTIL_AVAILABLE:
            try:
                self.start_cpu = psutil.Process().cpu_times()
            except Exception:
                self.start_cpu = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.perf_counter() - self.start_time
        cpu_user: "float | str" = ""
        cpu_sys: "float | str" = ""
        if self.start_cpu is not None and _PSUTIL_AVAILABLE:
            try:
                end_cpu = psutil.Process().cpu_times()
                cpu_user = round(end_cpu.user - self.start_cpu.user, 4)
                cpu_sys = round(end_cpu.system - self.start_cpu.system, 4)
            except Exception:
                pass

        if self._output_path is not None:
            with self._lock:
                try:
                    with open(self._output_path, "a", newline="", buffering=1) as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                self.operation_name,
                                round(duration, 4),
                                cpu_user,
                                cpu_sys,
                            ]
                        )
                except Exception:
                    pass
        print(
            f"[Profiler] {self.operation_name} took {duration:.4f}s"
            f" (CPU: user {cpu_user}s, sys {cpu_sys}s)",
            flush=True,
        )
