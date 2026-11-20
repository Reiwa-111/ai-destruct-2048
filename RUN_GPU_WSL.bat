@echo off
title 2048 DESTROYER - WSL ROCm GPU
wsl -d Ubuntu-24.04 -- bash /mnt/q/PROJECTS/2048-DESTROYER/scripts/run_gpu_wsl.sh
if errorlevel 1 pause
