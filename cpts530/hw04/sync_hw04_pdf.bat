@echo off
REM Script to sync Hw04_Cpts530 PDF from documentsCreated_repo
REM Author: Aryan Ritwajeet Jha
REM Date: October 30, 2025

echo === HW04 PDF Sync Script ===
echo.

REM Define paths
set "DOCUMENTS_CREATED_REPO=C:\Users\Aryan Ritwajeet Jha\Documents\documents_general\documentsCreated\documentsCreated_repo"
set "SOURCE_DIR=%DOCUMENTS_CREATED_REPO%\journals"
set "DEST_DIR=C:\Users\Aryan Ritwajeet Jha\Documents\documents_general\structured_courses\cpts530\hw04"

REM Step 1: Pull latest changes from documentsCreated_repo
echo Step 1: Pulling latest changes from documentsCreated_repo...
cd /d "%DOCUMENTS_CREATED_REPO%" || (
    echo Error: Could not navigate to documentsCreated_repo
    exit /b 1
)

git fetch
git pull origin main 2>nul || git pull origin master 2>nul || (
    echo Warning: Could not pull changes (might already be up to date^)
)

echo [32mRepository updated[0m
echo.

REM Step 2: Check if source file exists
echo Step 2: Checking for Hw04_Cpts530 file...

if exist "%SOURCE_DIR%\Hw04_Cpts530.pdf" (
    set "SOURCE_FILE=%SOURCE_DIR%\Hw04_Cpts530.pdf"
    echo [32mFound: Hw04_Cpts530.pdf[0m
) else if exist "%SOURCE_DIR%\Hw04_Cpts530.PDF" (
    set "SOURCE_FILE=%SOURCE_DIR%\Hw04_Cpts530.PDF"
    echo [32mFound: Hw04_Cpts530.PDF[0m
) else (
    echo [31mError: Hw04_Cpts530 file not found in %SOURCE_DIR%[0m
    echo Looking for files in the directory...
    dir "%SOURCE_DIR%\*hw04*" /b 2>nul
    exit /b 1
)

echo.

REM Step 3: Copy file to destination
echo Step 3: Copying file to hw04 folder...
echo Source: %SOURCE_FILE%

copy /Y "%SOURCE_FILE%" "%DEST_DIR%\" >nul || (
    echo [31mError: Could not copy file[0m
    exit /b 1
)

echo [32mFile copied successfully to:[0m
echo   %DEST_DIR%\Hw04_Cpts530.pdf
echo.

REM Step 4: Verify copy
echo Step 4: Verifying copy...
if exist "%DEST_DIR%\Hw04_Cpts530.pdf" (
    echo [32mCopy verified![0m
) else if exist "%DEST_DIR%\Hw04_Cpts530.PDF" (
    echo [32mCopy verified![0m
) else (
    echo [31mError: Destination file not found after copy[0m
    exit /b 1
)

echo.
echo [32m=== Sync Complete ===[0m
echo You can now work with the latest version of Hw04_Cpts530
pause
