@echo off
title 2048 DESTROYER - WSL ROCm GPU
wsl.exe -d Ubuntu-24.04 --cd "%~dp0" -- bash scripts/run_gpu_wsl.sh
if errorlevel 1 pause
