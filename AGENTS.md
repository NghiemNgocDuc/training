# User preferences

- Always run long-running jobs with `nohup` (or the Windows equivalent: `Start-Process` detached with stdout/stderr redirected to log files). Never block the session on long training runs.
