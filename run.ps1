# Dev shortcuts for the fixed-point stress-evaluation script (Windows / no make needed).
# ASCII-only on purpose: Windows PowerShell 5.1 mis-parses UTF-8 (no BOM) multibyte comments.
# Usage:
#   .\run.ps1 run             # run the script
#   .\run.ps1 serve          # start the evaluation server (open http://localhost:8000/)
#   .\run.ps1 ros2           # run as ROS2 node (ROS_MODE=both for video)
#   .\run.ps1 clean           # delete session outputs (log / graph / __pycache__) in the data dir
#   .\run.ps1 clean-profiles  # delete learned baselines + person IDs (reset fixed-point data)
#   .\run.ps1 clean-all       # delete everything above
#   .\run.ps1 show-profiles   # pretty-print the saved profiles
param([string]$task = "run")

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

# Data lives outside the repo (config.DATA_DIR / MIRS_DATA_DIR); delegate to python like the Makefile.
function Invoke-Py([string]$code) { & $python -c $code }

$cleanCode = "import os,shutil,config;[os.remove(p) for p in (config.STRESS_CSV_PATH,config.STRESS_GRAPH_PATH) if os.path.exists(p)];shutil.rmtree(os.path.join(config.REPO_DIR,'__pycache__'),ignore_errors=True)"
$profCode = "import os,config;[os.remove(p) for p in (config.PROFILES_JSON_PATH,config.SESSION_HISTORY_PATH) if os.path.exists(p)]"

switch ($task) {
    "run" {
        & $python peroson_detect.py
    }
    "serve" {
        & $python stress_server.py
    }
    "ros2" {
        $mode = if ($env:ROS_MODE) { $env:ROS_MODE } else { "ros2" }
        & $python peroson_detect.py --output-mode $mode
    }
    "paths" {
        Invoke-Py "import config;print('DATA_DIR :',config.DATA_DIR);print('MODEL_DIR:',config.MODEL_DIR)"
    }
    "clean" {
        Invoke-Py $cleanCode
        Write-Host "Removed session outputs."
    }
    "clean-profiles" {
        Invoke-Py $profCode
        Write-Host "Removed fixed-point data (person_profiles.json / session_history.jsonl)."
    }
    "clean-all" {
        Invoke-Py $cleanCode
        Invoke-Py $profCode
        Write-Host "Removed all outputs and fixed-point data."
    }
    "show-profiles" {
        Invoke-Py "import json,config;print(json.dumps(json.load(open(config.PROFILES_JSON_PATH,encoding='utf-8')),indent=2,ensure_ascii=False))"
    }
    default {
        Write-Host "unknown task: $task"
        Write-Host "targets: run | serve | ros2 | paths | clean | clean-profiles | clean-all | show-profiles"
        exit 1
    }
}
