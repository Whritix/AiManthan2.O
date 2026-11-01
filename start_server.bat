@echo off
cd C:\Games\Indore\Indore
set PYTHONPATH=C:\Games\Indore\Indore
set INDORE_DETECTOR=unified
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
