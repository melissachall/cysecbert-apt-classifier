@echo off
echo 🚀 Deploying APT Classification System...
echo Current time: %DATE% %TIME%
echo Current user: %USERNAME%
echo Current directory: %CD%

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python3 not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv apt_classifier_env

REM Activate virtual environment
echo 📦 Activating virtual environment...
call apt_classifier_env\Scripts\activate.bat

REM Upgrade pip
echo 🔥 Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 🔥 Installing dependencies...
if exist requirements_interface.txt (
    pip install -r requirements_interface.txt
    echo ✅ Dependencies installed!
) else (
    echo ⚠️ requirements_interface.txt not found, installing manually...
    pip install streamlit torch transformers pandas numpy plotly seaborn matplotlib scikit-learn PyPDF2 python-docx tqdm fastapi uvicorn python-multipart
)

REM Verify model file
if not exist "best_cysecbert_max_performance.pt" (
    echo ❌ Model file not found!
    echo 📁 Please ensure best_cysecbert_max_performance.pt is in the current directory
    dir *.pt 2>nul
    pause
    exit /b 1
)

echo ✅ Model file found!

REM Verify interface file
if not exist "apt_classification_interface.py" (
    echo ❌ Interface file not found!
    pause
    exit /b 1
)

echo ✅ Interface file found!

REM Check API file
set API_AVAILABLE=false
if exist "apt_classification_api.py" (
    set API_AVAILABLE=true
    echo ✅ API file found!
) else (
    echo ⚠️ API file not found. Only Streamlit option available.
)

echo.
echo 🚀 All dependencies installed successfully!
echo.
echo Choose deployment option:
echo 1. Streamlit Interface (Web UI) - Recommended for demo
if "%API_AVAILABLE%"=="true" (
    echo 2. FastAPI REST API - For integration
    echo 3. Both services
    set /p choice="Enter your choice (1-3): "
) else (
    echo Only option 1 is available
    set choice=1
)

if "%choice%"=="1" (
    echo 🌐 Starting Streamlit interface...
    echo 📱 Access at: http://localhost:8501
    echo ℹ️ Press Ctrl+C to stop
    streamlit run apt_classification_interface.py
) else if "%choice%"=="2" (
    if "%API_AVAILABLE%"=="true" (
        echo 🔌 Starting FastAPI server...
        echo 📱 API at: http://localhost:8000
        echo 📚 Docs at: http://localhost:8000/docs
        python apt_classification_api.py
    ) else (
        echo ❌ API file not available
        pause
        exit /b 1
    )
) else if "%choice%"=="3" (
    if "%API_AVAILABLE%"=="true" (
        echo 🚀 Starting both services...
        echo 📱 Streamlit: http://localhost:8501
        echo 🔌 API: http://localhost:8000
        start /b python apt_classification_api.py
        timeout /t 3 >nul
        streamlit run apt_classification_interface.py
    ) else (
        echo ❌ API file not available
        pause
        exit /b 1
    )
) else (
    echo ❌ Invalid choice
    pause
    exit /b 1
)