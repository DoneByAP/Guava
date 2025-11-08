#!/usr/bin/env python3
"""
EnergyMonitor — cross-platform system + GPU telemetry with FLOPs and task tagging.

Adds:
- FLOP estimation (transformer rule-of-thumb)
- TFLOPs/s + TFLOPs/J in meta
- Windows CPU temp via OHM (if available)
- psutil + NVML fallback
- Task tagging via .task("name") so parallel workloads are attributable
- Static model/task info for heartbeats
- Per-step train_event packets for orchestrator aggregation
"""

from __future__ import annotations
import os, json, time, socket, platform, contextlib
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except Exception:
    psutil = None

# NVML
_nvml_loaded = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_loaded = True
except Exception:
    _nvml_loaded = False


def _read_cpu_temp_windows() -> Optional[float]:
    """
    Try to read CPU temp from OpenHardwareMonitor WMI if available (Windows only).
    Returns float °C or None.
    """
    try:
        import wmi
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        for s in w.Sensor():
            if s.SensorType == u"Temperature" and "CPU" in s.Name:
                return float(s.Value)
    except Exception:
        return None
    return None


class EnergyMonitor:
    def __init__(
        self,
        role: str,
        udp_host: str,
        udp_port: int,
        gpu_ids: Optional[List[int]] = None,
        net_j_per_byte: float = 4e-9,
        hostname: Optional[str] = None,
        model_d_model: Optional[int] = None,
        model_layers: Optional[int] = None,
        enable_error_tracking: bool = True,

    ):
        """
        Lightweight, fire-and-forget telemetry publisher.
        We do *not* block training. UDP only.

        role: "worker", "orchestrator", etc.
        udp_host/udp_port: where orchestrator is listening (master_port+8)
        gpu_ids: which local GPU indices to sample
        model_d_model/model_layers: used for FLOP estimation heuristic
        """
        self.role = role
        self.addr = (udp_host, udp_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.gpu_ids = gpu_ids or []
        self.net_j_per_byte = float(net_j_per_byte)
        self.hostname = hostname or socket.gethostname()

        # live model+workload identity
        self.model_name: Optional[str] = None     # "ChatTransformerLM"
        self.params: Optional[int] = None         # total param count
        self.model_d_model = model_d_model or 0   # width for FLOP est
        self.model_layers  = model_layers  or 0   # depth for FLOP est

        # task tagging
        # example values: "train.gpu0", "infer.gpu1", "upload", etc.
        self.current_task: str = "default"

        # these persist across calls so heartbeat is never empty
        self.static_task: Optional[str]   = None
        self.static_model: Optional[str]  = None
        self.static_params: Optional[int] = None

        # rolling state for net energy calc
        self._last_tokens = 0
        self._last_net = self._read_net_bytes()
        self._last_t = time.time()

        self.enable_error_tracking = enable_error_tracking
        if self.enable_error_tracking:
            self._error_tracker = get_error_tracker()
            self._error_tracker.attach_energy_monitor(self)
        else:
            self._error_tracker = None
    # ------------------------------------------------------------------
    # Static identity setters
    # ------------------------------------------------------------------
    def set_model_info(self, model_name: str, params: int):
        """
        Backwards-compatible setter.
        Records model identity & parameter count for telemetry.
        """
        try:
            self.model_name = model_name
        except Exception:
            self.model_name = str(model_name)
        try:
            self.params = int(params)
        except Exception:
            self.params = None

        # Also update static_* so heartbeat isn't null.
        self.static_model = self.model_name
        self.static_params = self.params

    def set_static_info(self, task: str, model_name: Optional[str], param_count: Optional[int]):
        """
        Called by worker after CONTROL_ACK (once it knows who it is).
        task:        "train.gpu0", "infer.gpu1", etc.
        model_name:  class name of the model on this worker
        param_count: total parameters of that model
        """
        self.static_task   = task
        self.static_model  = model_name
        self.static_params = param_count

        # keep current_task/model_name/params in sync too
        if task:
            self.current_task = task
        if model_name:
            self.model_name = model_name
        if param_count is not None:
            try:
                self.params = int(param_count)
            except Exception:
                self.params = None

    # ------------------------------------------------------------------
    # FLOP math
    # ------------------------------------------------------------------
    def estimate_flops(self, tokens: int) -> float:
        """
        Very rough transformer FLOP estimate per forward+backward:
        FLOPs ≈ 6 * L * D^2 * tokens
        L: number of layers
        D: hidden dim
        tokens: #tokens processed this interval
        """
        if not (tokens and self.model_layers and self.model_d_model):
            return 0.0
        L, D = self.model_layers, self.model_d_model
        return 6 * L * (D ** 2) * tokens

    def flops_metrics(self, flops: float, joules: float, dt: float) -> Dict[str, float]:
        """
        Produce TFLOPs/s, TFLOPs/J and include raw FLOPs.
        Avoids div-by-zero.
        """
        if flops <= 0 or dt <= 0:
            return {
                "TFLOPs/s":    0.0,
                "TFLOPs/J":    0.0,
                "total_flops": float(flops),
            }
        tflops_s = flops / dt / 1e12
        tflops_j = flops / max(joules, 1e-12) / 1e12
        return {
            "TFLOPs/s":    float(f"{tflops_s:.9f}"),
            "TFLOPs/J":    float(f"{tflops_j:.9f}"),
            "total_flops": float(flops),
        }

    # ------------------------------------------------------------------
    # Low-level samplers
    # ------------------------------------------------------------------
    def _sample_cpu(self) -> Dict[str, Optional[float]]:
        temp = None

        # Windows OHM path (OpenHardwareMonitor)
        if platform.system() == "Windows":
            try:
                from OHM import Computer
                pc = Computer()
                pc.CPUEnabled = True
                pc.Open()
                temps = []
                for hw in pc.Hardware:
                    if "cpu" in getattr(hw, "HardwareType", "").lower():
                        hw.Update()
                        for s in hw.Sensors:
                            if getattr(s, "SensorType", "").lower() == "temperature":
                                temps.append(s.Value)
                if temps:
                    temp = sum(temps) / len(temps)
            except Exception:
                pass

            # WMI fallback (still Windows)
            if temp is None:
                t2 = _read_cpu_temp_windows()
                if t2 is not None:
                    temp = t2

        # psutil load/freq, cross-platform
        if psutil:
            try:
                load = psutil.cpu_percent()
                freq = getattr(psutil.cpu_freq(), "current", None)
            except Exception:
                load, freq = None, None
        else:
            load = freq = None

        # psutil temps on Linux/mac
        if temp is None and psutil:
            try:
                temps = getattr(psutil, "sensors_temperatures", lambda: {})() or {}
                for arr in temps.values():
                    if arr:
                        temp = arr[0].current
                        break
            except Exception:
                temp = None

        return {"load": load, "freq": freq, "temp": temp}

    def _sample_ram(self) -> Dict[str, Optional[float]]:
        if not psutil:
            return {}
        try:
            v = psutil.virtual_memory()
            return {"percent": v.percent, "used": v.used, "total": v.total}
        except Exception:
            return {}

    def _sample_disk(self) -> Dict[str, Optional[float]]:
        if not psutil:
            return {}
        try:
            io = psutil.disk_io_counters()
            return {"read": io.read_bytes, "write": io.write_bytes}
        except Exception:
            return {}

    def _read_net_bytes(self) -> int:
        if not psutil:
            return 0
        try:
            c = psutil.net_io_counters()
            return int(c.bytes_sent + c.bytes_recv)
        except Exception:
            return 0

    def _sample_net(self, now: float) -> Tuple[int, float]:
        cur = self._read_net_bytes()
        delta = max(0, cur - self._last_net)
        self._last_net = cur
        return delta, delta * self.net_j_per_byte

    def _sample_gpus(self) -> List[Dict[str, Optional[float]]]:
        if not (_nvml_loaded and self.gpu_ids):
            return []
        out = []
        for gid in self.gpu_ids:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(gid)
                p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # watts
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                c = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                out.append({"id": gid, "power": p, "temp": t, "clock": c})
            except Exception:
                out.append({"id": gid, "power": None, "temp": None, "clock": None})
        return out

    # ------------------------------------------------------------------
    # Internal packet builder / sender
    # ------------------------------------------------------------------
    def _snapshot_packet(
        self,
        event: str,
        step: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        override_task: Optional[str] = None,
        override_model: Optional[str] = None,
        override_params: Optional[int] = None,
        include_errors: bool = True,
    ) -> Dict[str, Any]:
        """
        Build the structured telemetry dict we'll send via UDP.
        This is the SINGLE source of truth so "heartbeat" and "train_step"
        always carry model/params/task if we know them.
        """
        cpu   = self._sample_cpu()
        ram   = self._sample_ram()
        disk  = self._sample_disk()
        net_b, net_j = self._sample_net(time.time())
        gpus  = self._sample_gpus()

        # choose labels in priority order: explicit override -> static_* -> live fields -> fallback
        task_label   = override_task   or self.static_task   or self.current_task or "worker.unknown"
        model_label  = override_model  or self.static_model  or self.model_name
        params_label = override_params or self.static_params or self.params

        pkt = {
            "t": time.time(),
            "host": self.hostname,
            "role": self.role,

            "task": task_label,          # "train.gpu0", "infer.gpu1", ...
            "event": event,              # "heartbeat", "train_step", "forward_end", ...
            "step": step,                # global step or None

            "model": model_label,        # model class name
            "params": params_label,      # total parameter count (int)

            "cpu": cpu,
            "ram": ram,
            "disk": disk,
            "net": {"bytes": net_b, "joules": net_j},
            "gpu": gpus,
            "meta": extra_meta or {},
        }
        if include_errors and self.enable_error_tracking and self._error_tracker:
            try:
                self._error_tracker.inject_errors_into_meta(pkt["meta"], max_errors=3)
            except Exception:
                # Never let error tracking crash telemetry
                pass
        # ============================================================
        
        return pkt
    
    def _send_udp(self, payload: Dict[str, Any]) -> None:
        try:
            self.sock.sendto(json.dumps(payload).encode(), self.addr)
        except Exception:
            # UDP is best-effort: never raise
            pass

    # ------------------------------------------------------------------
    # Public emitters
    # ------------------------------------------------------------------
    def send(self, event: str, step: Optional[int] = None, meta: Optional[Dict[str, Any]] = None):
        """
        Backwards-compatible fire-and-forget event.
        Keeps your old API working.
        """
        pkt = self._snapshot_packet(event=event, step=step, extra_meta=meta)
        self._send_udp(pkt)

    def sample(self, evt: str):
        """Alias for send(evt) with no step/meta."""
        return self.send(evt)

    def background(self, evt: str = "heartbeat", interval: float = 1.0):
        """
        Periodic telemetry loop (daemon-friendly).
        Always includes task/model/params if we know them.
        """
        while True:
            pkt = self._snapshot_packet(event=evt)
            self._send_udp(pkt)
            time.sleep(interval)

    def step_event(
        self,
        task: str,
        step: Optional[int],
        model_name: Optional[str],
        param_count: Optional[int],
        total_flops: float,
    ):
        """
        Emit a 'train_step' packet. The orchestrator uses this to line up:
        - energy draw (watts -> joules)
        - total_flops (for TFLOPs/J, J/token)
        - which GPU/task did the work
        """
        meta = {"total_flops": total_flops}
        pkt = self._snapshot_packet(
            event="train_step",
            step=step,
            extra_meta=meta,
            override_task=task,
            override_model=model_name,
            override_params=param_count,
        )
        self._send_udp(pkt)

    # ------------------------------------------------------------------
    # Context helpers for marking code regions with FLOP estimates
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def mark(self, tag: str, step: Optional[int] = None, tokens: Optional[int] = None):
        """
        Run a code region with automatic '<tag>_start' and '<tag>_end' events.
        We also attach FLOP metrics on *_end based on token count.
        """
        # remember tokens for FLOP estimate
        self._last_tokens = tokens or self._last_tokens or 0

        # START event (lightweight)
        pkt_start = self._snapshot_packet(event=f"{tag}_start", step=step)
        self._send_udp(pkt_start)

        t0 = time.time()
        try:
            yield
        finally:
            dt = max(time.time() - t0, 1e-9)
            flops = self.estimate_flops(self._last_tokens)

            # crude joule proxy from network bytes only
            # (for better accuracy, orchestrator sums GPU watts over dt)
            _, net_j = self._sample_net(time.time())

            meta = {
                "dt": dt,
                "tokens": self._last_tokens,
                **self.flops_metrics(flops, net_j, dt),
            }

            pkt_end = self._snapshot_packet(
                event=f"{tag}_end",
                step=step,
                extra_meta=meta,
            )
            self._send_udp(pkt_end)

    @contextlib.contextmanager
    def task(self, name: str):
        """
        Temporarily override self.current_task so everything inside
        (heartbeats, mark(), step_event(), etc.) gets attributed
        to that logical task.

        Example:
            with mon.task(f"train.gpu{gid}"):
                with mon.mark("forward", step=step, tokens=tokens):
                    ...
        """
        prev = self.current_task
        self.current_task = name
        try:
            yield
        finally:
            self.current_task = prev


# ============================================================================
# ERROR TRACKING & REPORTING SUBSYSTEM
# ============================================================================
"""
Centralized error tracking that feeds into Energy Monitor telemetry.

DESIGN:
- ErrorTracker is a singleton that any module can import
- Errors are automatically captured with full context (traceback, step, GPU, etc.)
- Errors flow into the meta={} section of telemetry packets
- Users can import and use in their launch scripts

USAGE IN GUAVA MODULES:
    from .energy_monitor import get_error_tracker
    
    try:
        # ... your code ...
    except Exception as e:
        get_error_tracker().report_error(
            error=e,
            context="network_worker",
            gpu_id=self.gpu_id,
            step=self.global_step,
            severity="critical"
        )
        raise  # Re-raise if needed

USAGE IN USER LAUNCH SCRIPTS:
    from guava.energy_monitor import get_error_tracker, ErrorSeverity
    
    try:
        # ... user's training code ...
    except ValueError as e:
        get_error_tracker().report_error(
            error=e,
            context="user_dataloader",
            severity=ErrorSeverity.WARNING,
            metadata={"batch_idx": batch_idx}
        )
"""

import traceback
import threading
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class ErrorReport:
    """Structured error report that gets serialized into telemetry."""
    
    timestamp: float
    error_type: str
    error_message: str
    context: str
    severity: str
    
    # Optional fields
    gpu_id: Optional[int] = None
    step: Optional[int] = None
    traceback: Optional[str] = None
    hostname: Optional[str] = None
    
    # User-provided metadata
    metadata: Optional[Dict[str, Any]] = None
    
    # Recovery info
    recovery_attempted: bool = False
    recovery_success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization, excluding None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


class ErrorTracker:
    """
    Thread-safe error tracking singleton.
    
    Buffers errors and integrates with EnergyMonitor to include them
    in telemetry packets automatically.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._errors: List[ErrorReport] = []
        self._errors_lock = threading.Lock()
        self._max_buffer_size = 1000  # Prevent unbounded growth
        self._energy_monitor: Optional['EnergyMonitor'] = None
        self._hostname = socket.gethostname()
    
    def attach_energy_monitor(self, monitor: 'EnergyMonitor') -> None:
        """Link this tracker to an EnergyMonitor instance."""
        self._energy_monitor = monitor
    
    def report_error(
        self,
        error: Exception,
        context: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        gpu_id: Optional[int] = None,
        step: Optional[int] = None,
        include_traceback: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        recovery_attempted: bool = False,
        recovery_success: Optional[bool] = None,
    ) -> None:
        """
        Report an error to the tracking system.
        
        Args:
            error: The exception object
            context: Where the error occurred (e.g., "network_worker", "user_dataloader")
            severity: Error severity level
            gpu_id: GPU ID if relevant
            step: Training step if relevant
            include_traceback: Whether to include full traceback
            metadata: Additional context (dict)
            recovery_attempted: Whether automatic recovery was tried
            recovery_success: Whether recovery succeeded (True/False/None)
        """
        
        # Build traceback string
        tb_str = None
        if include_traceback:
            try:
                tb_str = ''.join(traceback.format_exception(
                    type(error), error, error.__traceback__
                ))
                # Truncate to prevent massive packets
                if len(tb_str) > 4000:
                    tb_str = tb_str[:3900] + "\n... (truncated)"
            except Exception:
                tb_str = str(error)
        
        # Create error report
        report = ErrorReport(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            severity=severity.value if isinstance(severity, ErrorSeverity) else severity,
            gpu_id=gpu_id,
            step=step,
            traceback=tb_str,
            hostname=self._hostname,
            metadata=metadata,
            recovery_attempted=recovery_attempted,
            recovery_success=recovery_success,
        )
        
        # Buffer the error
        with self._errors_lock:
            self._errors.append(report)
            
            # Enforce buffer size limit (FIFO)
            if len(self._errors) > self._max_buffer_size:
                self._errors.pop(0)
        
        # If we have an energy monitor, send immediately as telemetry event
        if self._energy_monitor:
            try:
                self._energy_monitor.send(
                    event="error_report",
                    step=step,
                    meta=report.to_dict()
                )
            except Exception:
                # Never let error reporting crash the main code
                pass
    
    def get_recent_errors(self, n: int = 10) -> List[ErrorReport]:
        """Get the N most recent errors."""
        with self._errors_lock:
            return self._errors[-n:]
    
    def get_errors_since(self, timestamp: float) -> List[ErrorReport]:
        """Get all errors since a given timestamp."""
        with self._errors_lock:
            return [e for e in self._errors if e.timestamp >= timestamp]
    
    def get_errors_by_context(self, context: str) -> List[ErrorReport]:
        """Get all errors from a specific context."""
        with self._errors_lock:
            return [e for e in self._errors if e.context == context]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorReport]:
        """Get all errors of a specific severity."""
        severity_str = severity.value if isinstance(severity, ErrorSeverity) else severity
        with self._errors_lock:
            return [e for e in self._errors if e.severity == severity_str]
    
    def clear_errors(self) -> None:
        """Clear the error buffer."""
        with self._errors_lock:
            self._errors.clear()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a statistical summary of errors."""
        with self._errors_lock:
            if not self._errors:
                return {
                    "total_errors": 0,
                    "by_severity": {},
                    "by_context": {},
                    "by_error_type": {},
                    "recovery_stats": {
                        "attempted": 0,
                        "successful": 0,
                        "failed": 0,
                    }
                }
            
            # Count by severity
            by_severity = {}
            for e in self._errors:
                by_severity[e.severity] = by_severity.get(e.severity, 0) + 1
            
            # Count by context
            by_context = {}
            for e in self._errors:
                by_context[e.context] = by_context.get(e.context, 0) + 1
            
            # Count by error type
            by_error_type = {}
            for e in self._errors:
                by_error_type[e.error_type] = by_error_type.get(e.error_type, 0) + 1
            
            # Recovery stats
            recovery_attempted = sum(1 for e in self._errors if e.recovery_attempted)
            recovery_successful = sum(1 for e in self._errors if e.recovery_success is True)
            recovery_failed = sum(1 for e in self._errors if e.recovery_success is False)
            
            return {
                "total_errors": len(self._errors),
                "by_severity": by_severity,
                "by_context": by_context,
                "by_error_type": by_error_type,
                "oldest_error": self._errors[0].timestamp,
                "newest_error": self._errors[-1].timestamp,
                "recovery_stats": {
                    "attempted": recovery_attempted,
                    "successful": recovery_successful,
                    "failed": recovery_failed,
                }
            }
    
    def inject_errors_into_meta(self, meta: Dict[str, Any], max_errors: int = 5) -> None:
        """
        Inject recent errors into a metadata dict (for telemetry).
        
        This is called automatically by EnergyMonitor when building packets.
        """
        recent = self.get_recent_errors(max_errors)
        if recent:
            meta["recent_errors"] = [e.to_dict() for e in recent]
            meta["error_summary"] = self.get_error_summary()


# Global singleton accessor
_error_tracker_instance = None

def get_error_tracker() -> ErrorTracker:
    """Get the global ErrorTracker singleton."""
    global _error_tracker_instance
    if _error_tracker_instance is None:
        _error_tracker_instance = ErrorTracker()
    return _error_tracker_instance


# ============================================================================
# ENHANCED ENERGYMONITOR WITH ERROR TRACKING
# ============================================================================


# ============================================================================
# DECORATOR: @track_errors - Drop-in error tracking for any function
# ============================================================================
from functools import wraps

def track_errors(
    context: str = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    reraise: bool = True,
    recovery_attempted: bool = False,
):
    """
    Decorator that automatically reports exceptions to the error tracker.
    
    Usage:
        @track_errors(context="network_worker.forward_pass", severity=ErrorSeverity.CRITICAL)
        def _handle_train_data_parallel(self, msg):
            # ... your code ...
    
        @track_errors(context="orchestrator.broadcast", reraise=False)
        def _broadcast_ctrl(self, msg):
            # ... your code ...
    
    Args:
        context: Where the error occurred (auto-generated from function name if not provided)
        severity: Error severity level
        reraise: Whether to re-raise the exception after reporting (default: True)
        recovery_attempted: Whether this function attempts recovery
    
    Returns:
        Decorated function that auto-reports errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Auto-generate context from function/class if not provided
            actual_context = context
            if actual_context is None:
                # Try to get class name if it's a method
                if args and hasattr(args[0], '__class__'):
                    class_name = args[0].__class__.__name__
                    actual_context = f"{class_name}.{func.__name__}"
                else:
                    actual_context = func.__name__
            
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                # Extract metadata from self if available
                self_obj = args[0] if args else None
                gpu_id = getattr(self_obj, 'gpu_id', None)
                step = getattr(self_obj, '_step_counter', None) or getattr(self_obj, 'global_step', None)
                
                # Build metadata
                metadata = {
                    "function": func.__name__,
                    "module": func.__module__,
                }
                
                # Add CUDA-specific metadata if it's a CUDA error
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    metadata["cuda_error"] = True
                    metadata["error_category"] = "gpu_failure"
                
                # Add connection-specific metadata
                if any(x in str(e).lower() for x in ["connection", "socket", "timeout"]):
                    metadata["error_category"] = "network_failure"
                
                # Report to error tracker
                try:
                    get_error_tracker().report_error(
                        error=e,
                        context=actual_context,
                        severity=severity,
                        gpu_id=gpu_id,
                        step=step,
                        include_traceback=True,
                        metadata=metadata,
                        recovery_attempted=recovery_attempted,
                    )
                except Exception:
                    # Never let error reporting crash the main code
                    pass
                
                # Re-raise if requested
                if reraise:
                    raise
                
                # Otherwise, return None (error was swallowed)
                return None
        
        return wrapper
    return decorator