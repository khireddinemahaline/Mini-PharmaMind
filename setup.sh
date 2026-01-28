#!/bin/bash
# PharmaMind Setup Script
# This script automates the setup process for the PharmaMind application

set -e  # Exit on error

echo "🧬 PharmaMind Setup Script"
echo "=========================="
echo ""

# Check if Python 3.10+ is installed
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Error: Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"
echo ""

# Check if uv is installed
echo "Checking for uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your API keys and configuration"
    echo ""
fi

# Install dependencies
echo "Installing Python dependencies..."
uv sync
echo "✅ Dependencies installed"
echo ""

# Setup Prisma
echo "Setting up database with Prisma..."
if command -v npx &> /dev/null; then
    npx prisma generate
    echo "✅ Prisma client generated"
    echo ""
    
    read -p "Do you want to push the database schema? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        npx prisma db push
        echo "✅ Database schema pushed"
    fi
else
    echo "⚠️  Node.js/npx not found. Please install Node.js to use Prisma."
    echo "   Or manually run: npx prisma generate && npx prisma db push"
fi
echo ""

# Create necessary directories
echo "Creating required directories..."
mkdir -p session_state
mkdir -p generated_reports
mkdir -p mlruns
echo "✅ Directories created"
echo ""

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update your .env file with your API keys"
echo "2. Start the application with: chainlit run orcastration/main_chainlit.py -w --host 0.0.0.0 --port 8000"
echo "3. (Optional) Start MLflow server: mlflow server --host 0.0.0.0 --port 5000"
echo ""
echo "For more information, see README.md"
