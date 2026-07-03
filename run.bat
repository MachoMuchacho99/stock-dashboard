@echo off
cd /d "%~dp0"
rem Prefer the Kronos-Forecast venv (has torch for the Forecast tab).
set "KF_PY=C:\Users\river\Projects\Kronos-Forecast\.venv\Scripts\python.exe"
if exist "%KF_PY%" (
    "%KF_PY%" -m streamlit run "%~dp0app.py"
) else (
    python -m streamlit run "%~dp0app.py"
)
