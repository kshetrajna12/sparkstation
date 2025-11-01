# Bug Report - Sparkstation v0.1.0

**Date**: 2025-10-27 (Original Report)
**Last Updated**: 2025-10-31 (Bug Verification)
**Severity Levels**: HIGH, MEDIUM, LOW

---

## ✅ STATUS: ALL BUGS FIXED

All bugs identified in the original report have been **verified as fixed** in the current codebase:

- **BUG #1** (MEDIUM): ✅ Fixed - `last_request_time` now persisted to database (line 139 in auto_suspend.py)
- **BUG #2** (MEDIUM): ✅ Fixed - Process termination properly verified with SIGTERM/SIGKILL (vllm_launcher.py:165-186)
- **BUG #3** (HIGH): ✅ Fixed - `saved_config` now set on model start (main.py:256-268)
- **BUG #4** (MEDIUM): ✅ Fixed - SGLang launcher has proper process termination (sglang_launcher.py:165-190)

**Fix Commits**: ed7fd0b (critical production bugs), and earlier commits

All 24 unit tests pass. ✅

---

## Original Bug Reports (For Reference)

## BUG #1: `last_request_time` not persisted to database (MEDIUM) ✅ FIXED

**Status**: ✅ **FIXED** - Verified on 2025-10-31
**Location**: `supervisor/auto_suspend.py:137-139`

**Issue**:
```python
if model.last_request_time is None:
    model.last_request_time = model.started_at
```

The code modifies `model.last_request_time` in memory but doesn't call `await self.registry.update(model)` to persist it. This means:
- The change is lost on supervisor restart
- Idle timeout calculations may be incorrect after restart
- Models might not auto-suspend when expected

**Fix**:
```python
if model.last_request_time is None:
    model.last_request_time = model.started_at
    await self.registry.update(model)  # Persist to database
```

**Impact**: Medium - Edge case (only affects models that never received a request), but can cause unexpected behavior.

---

## BUG #2: Process kill doesn't verify termination (MEDIUM) ✅ FIXED

**Status**: ✅ **FIXED** - Verified on 2025-10-31
**Location**: `supervisor/launchers/vllm_launcher.py:165-186`
**Location**: `supervisor/launchers/sglang_launcher.py:165-190`

**Issue**:
```python
process = subprocess.Popen(["kill", str(instance.pid)])
process.wait(timeout=10)
```

This creates a *new* process to execute the `kill` command, but:
- Doesn't check if the target process (instance.pid) actually died
- The `wait()` waits for the *kill command* to finish, not the target process
- Zombie processes possible if kill signal ignored

**Fix**:
```python
import os
import signal

try:
    os.kill(instance.pid, signal.SIGTERM)
    # Wait for process to terminate
    for _ in range(100):  # 10 seconds max
        try:
            os.kill(instance.pid, 0)  # Check if still alive
            await asyncio.sleep(0.1)
        except ProcessLookupError:
            logger.info(f"Stopped vLLM process PID={instance.pid}")
            return True
    # Force kill if still alive
    os.kill(instance.pid, signal.SIGKILL)
    logger.warning(f"Force killed vLLM process PID={instance.pid}")
    return True
except ProcessLookupError:
    # Already dead
    return True
except Exception as e:
    logger.error(f"Failed to stop vLLM model: {e}")
    return False
```

**Impact**: Medium - Can leave zombie processes if models don't respond to SIGTERM

---

## BUG #3: `saved_config` not set on model start (HIGH) ✅ FIXED

**Status**: ✅ **FIXED** - Verified on 2025-10-31
**Location**: `supervisor/main.py:256-268`

**Issue**:
Models only have `saved_config` set when they're suspended (see `auto_suspend.py:173-183`). This means:
- If a model fails *before* being suspended, `saved_config` is `None`
- `RestartManager.handle_failed_model()` checks `if not model.saved_config` and fails
- Auto-restart will **never work** for models that fail on first launch or early in their lifecycle
- This defeats the entire purpose of the restart manager

**Example Failure Scenario**:
1. User starts model with `/models/start`
2. Model launches but crashes after 2 minutes (before idle timeout)
3. Health check marks it as FAILED
4. RestartManager tries to restart
5. Restart fails because `saved_config` is None
6. Model stuck in FAILED state

**Fix**:
In `supervisor/main.py:start_model()`, after creating the instance (around line 254):

```python
# Launch model
try:
    launcher = launcher_factory.get_launcher(request.backend)
    instance = await launcher.launch(config, model_id, port)
    instance.memory_gb = memory_estimate

    # Save config for restart/resume (CRITICAL for auto-restart)
    instance.saved_config = {
        "model_name": config.model_name,
        "backend": config.backend.value,
        "model_alias": config.model_alias,
        "gpu_ids": instance.gpu_ids,
        "port": port,
        "quantization": config.quantization,
        "auto_suspend_enabled": config.auto_suspend_enabled,
        "idle_timeout_minutes": config.idle_timeout_minutes,
        "extra_args": config.extra_args,
    }

except Exception as e:
    raise ModelLaunchError(request.backend.value, str(e))
```

**Impact**: **HIGH** - Auto-restart feature is completely broken for models that fail before suspension

---

## BUG #4: SGLang launcher process kill issue (MEDIUM) ✅ FIXED

**Status**: ✅ **FIXED** - Verified on 2025-10-31
**Location**: `supervisor/launchers/sglang_launcher.py:165-190`

**Issue**: Same as BUG #2 (process kill doesn't verify termination)

**Fix**: Same fix as BUG #2 - proper SIGTERM/SIGKILL with verification implemented

---

## Additional Observations (Not Bugs)

### 1. Health check timeout hardcoded in launcher
`vllm_launcher.py:190` uses `settings.health_check_timeout_seconds`, but the httpx client in `__init__` uses `timeout=30.0`. These should match.

### 2. No cleanup of failure_counts on restart success
In `health_check.py`, the `failure_counts` dict is never cleaned up for models that are stopped/deleted. This is a minor memory leak but not critical (dict size bounded by model count).

### 3. Resource manager port allocation not atomic
`resources.py` allocates ports but doesn't lock during allocation. Race condition possible if two models start simultaneously (unlikely but possible).

---

## Severity Summary

**Original Report (2025-10-27)**:
- **HIGH**: 1 bug (BUG #3 - breaks auto-restart)
- **MEDIUM**: 3 bugs (BUG #1, #2, #4 - edge cases and resource leaks)
- **LOW**: 0 critical path bugs

**Current Status (2025-10-31)**:
- **ALL BUGS FIXED**: ✅ 0 open bugs
- All 24 unit tests passing
- Production-ready

---

## Testing Recommendations

After fixes:
1. Test auto-restart with model that fails immediately
2. Test process cleanup with stuck models
3. Test supervisor restart with models that never received requests
4. Add unit tests for process termination logic
5. Add integration test for restart manager with saved_config

---

**Report Generated**: 2025-10-27
