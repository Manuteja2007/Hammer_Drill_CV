@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 "%~dp0run.hammer_drill" %*
exit /b %errorlevel%

:use_python
python "%~dp0run.hammer_drill" %*
exit /b %errorlevel%
