$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$logsDir = Join-Path $scriptDir "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$py = "python"
$epochs = "15"
$seed = "42"
$common = @("aqm-spice2/freesolv/neighbor_regularization/finetune_nbr.py",
            "--seed", $seed, "--epochs", $epochs, "--patience", "30",
            "--device", "cpu", "--track_groups")

# Runs: (out_rel, extra_args, log_name) - all under smoke_test/
$runs = @(
    @("smoke_test/baseline/lambda0_seed42",      @("--lambda_nbr", "0"),      "smoke_baseline_lambda0_seed42"),
    @("smoke_test/raw/lambda0.001_seed42",       @("--lambda_nbr", "0.001"),  "smoke_raw_lambda0.001_seed42"),
    @("smoke_test/raw/lambda0.01_seed42",        @("--lambda_nbr", "0.01"),   "smoke_raw_lambda0.01_seed42"),
    @("smoke_test/normalized/lambda0.1_seed42",  @("--lambda_nbr", "0.1", "--normalize_nbr"), "smoke_normalized_lambda0.1_seed42"),
    @("smoke_test/normalized/lambda1.0_seed42",  @("--lambda_nbr", "1.0", "--normalize_nbr"), "smoke_normalized_lambda1.0_seed42")
)

foreach ($r in $runs) {
    $outRel = $r[0]; $extra = $r[1]; $logName = $r[2]
    $log = Join-Path $logsDir "$logName.log"
    $argsList = @($common[0]) + $extra + @("--out", $outRel) + $common[1..($common.Length-1)]
    Write-Output ("=== {0} -> {1} ===" -f $outRel, $log)
    Write-Output ("{0} {1}" -f $py, ($argsList -join " "))
    & $py $argsList *> $log
    $code = $LASTEXITCODE
    Write-Output "exit code: $code"
    if ($code -ne 0) { Write-Output "RUN FAILED - stopping driver"; exit 1 }
}
Write-Output "ALL SMOKE RUNS DONE"