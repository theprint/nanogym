@echo off
REM Stop the NanoGym dashboard server (whatever is listening on port 7669)
set PORT=7669
echo Looking for a server on port %PORT%...

set FOUND=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
    set FOUND=1
    echo Stopping process %%p
    taskkill /f /pid %%p
)

if not defined FOUND echo No server found listening on port %PORT%.
