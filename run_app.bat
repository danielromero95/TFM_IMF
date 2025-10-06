@echo off
setlocal EnableExtensions

REM go to script dir
cd /d "%~dp0"

REM locate conda.bat (Miniforge)
set "CONDA=%USERPROFILE%\AppData\Local\miniforge3\condabin\conda.bat"
if not exist "%CONDA%" (
  echo ERROR: Miniforge not found at: %CONDA%
  echo Install Miniforge (user mode) and retry.
  exit /b 1
)

REM env name
set "ENV_NAME=gym_env"

REM check if env exists (try python -V; no redirections raras)
call "%CONDA%" run -n %ENV_NAME% python -V >nul
if errorlevel 1 (
  echo Creating environment %ENV_NAME%...
  set "ENVFILE=environment.yml"
  if exist environment.lock.yml set "ENVFILE=environment.lock.yml"
  call "%CONDA%" env create -n %ENV_NAME% -f "%ENVFILE%"
  if errorlevel 1 (
    echo ERROR: environment creation failed.
    pause
    exit /b 1
  )
)

REM launch app
echo Launching app...
call "%CONDA%" run -n %ENV_NAME% python -m src.gui.main

endlocal
