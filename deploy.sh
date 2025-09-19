#!/bin/bash

# APT Classification System Deployment Script
echo "🚀 Deploying APT Classification System..."
echo "Current time: $(date)"
echo "Current user: $USER"
echo "Current directory: $(pwd)"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3.8+"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv apt_classifier_env

# Activate virtual environment based on OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source apt_classifier_env/Scripts/activate
    echo "✅ Virtual environment activated (Windows)"
else
    # Linux/Mac
    source apt_classifier_env/bin/activate
    echo "✅ Virtual environment activated (Linux/Mac)"
fi

# Upgrade pip
echo "📥 Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
if [ -f "requirements_interface.txt" ]; then
    pip install -r requirements_interface.txt
    echo "✅ Dependencies from requirements_interface.txt installed!"
else
    echo "⚠️ requirements_interface.txt not found, installing manually..."
    pip install streamlit torch transformers pandas numpy plotly seaborn matplotlib scikit-learn PyPDF2 python-docx tqdm fastapi uvicorn python-multipart
fi

# Verify model file exists
echo "🔍 Checking for model file..."
if [ ! -f "best_cysecbert_max_performance.pt" ]; then
    echo "❌ Model file not found! Please ensure best_cysecbert_max_performance.pt is in the current directory."
    echo "📁 Current directory contents:"
    ls -la *.pt 2>/dev/null || echo "No .pt files found"
    exit 1
fi

echo "✅ Model file found: best_cysecbert_max_performance.pt"

# Verify interface file exists
if [ ! -f "apt_classification_interface.py" ]; then
    echo "❌ Interface file not found! Please ensure apt_classification_interface.py is in the current directory."
    exit 1
fi

echo "✅ Interface file found!"

# Check if API file exists for option 2 and 3
API_AVAILABLE=false
if [ -f "apt_classification_api.py" ]; then
    API_AVAILABLE=true
    echo "✅ API file found!"
else
    echo "⚠️ API file not found. Only Streamlit option will be available."
fi

echo ""
echo "🚀 All dependencies installed successfully!"
echo ""

# Option to run services
echo "Choose deployment option:"
echo "1. Streamlit Interface (Web UI) - Recommended for demo"

if [ "$API_AVAILABLE" = true ]; then
    echo "2. FastAPI REST API - For integration"
    echo "3. Both services"
    read -p "Enter your choice (1-3): " choice
else
    echo "Only option 1 is available (API file missing)"
    choice=1
fi

echo ""

case $choice in
    1)
        echo "🌐 Starting Streamlit interface..."
        echo "📱 Access the interface at: http://localhost:8501"
        echo "⏹️  Press Ctrl+C to stop"
        echo ""
        streamlit run apt_classification_interface.py
        ;;
    2)
        if [ "$API_AVAILABLE" = true ]; then
            echo "🔌 Starting FastAPI server..."
            echo "📱 API will be available at: http://localhost:8000"
            echo "📚 API documentation at: http://localhost:8000/docs"
            echo "⏹️  Press Ctrl+C to stop"
            echo ""
            python apt_classification_api.py
        else
            echo "❌ API file not available"
            exit 1
        fi
        ;;
    3)
        if [ "$API_AVAILABLE" = true ]; then
            echo "🚀 Starting both services..."
            echo "📱 Streamlit: http://localhost:8501"
            echo "🔌 API: http://localhost:8000"
            echo "📚 API docs: http://localhost:8000/docs"
            echo "⏹️  Press Ctrl+C to stop both services"
            echo ""
            
            # Start API in background
            python apt_classification_api.py &
            API_PID=$!
            
            # Wait a moment for API to start
            sleep 3
            
            # Start Streamlit (foreground)
            streamlit run apt_classification_interface.py
            
            # Kill API when Streamlit stops
            kill $API_PID 2>/dev/null
        else
            echo "❌ API file not available"
            exit 1
        fi
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac