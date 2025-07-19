#!/bin/bash
# GitHub Repository Setup Script for XAI Lung Segmentation Analysis

echo "🚀 XAI Lung Segmentation Analysis - GitHub Setup"
echo "==============================================="

# GitHub CLI path
GH_CLI="./gh_2.76.0_linux_amd64/bin/gh"

echo "📋 Repository Details:"
echo "  - Name: xai-lung-segmentation"
echo "  - Owner: Mo7aisen"
echo "  - Description: 🫁 Explainable AI for Medical Image Segmentation with Interactive Analysis Dashboard"
echo ""

# Check if already authenticated
if $GH_CLI auth status 2>/dev/null; then
    echo "✅ Already authenticated with GitHub"
else
    echo "🔐 Authentication required..."
    echo "Please run: $GH_CLI auth login"
    echo "Or use: $GH_CLI auth login --with-token < token.txt"
    echo ""
    read -p "Press Enter after authentication is complete..."
fi

# Create repository
echo "📦 Creating GitHub repository..."
$GH_CLI repo create xai-lung-segmentation \
    --public \
    --description "🫁 Explainable AI for Medical Image Segmentation with Interactive Analysis Dashboard" \
    --clone=false

if [ $? -eq 0 ]; then
    echo "✅ Repository created successfully!"
    
    # Add remote origin
    echo "🔗 Adding remote origin..."
    git remote add origin https://github.com/Mo7aisen/xai-lung-segmentation.git 2>/dev/null || \
    git remote set-url origin https://github.com/Mo7aisen/xai-lung-segmentation.git
    
    # Push to GitHub
    echo "📤 Pushing code to GitHub..."
    git branch -M main
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Repository setup complete!"
        echo "🌐 Repository URL: https://github.com/Mo7aisen/xai-lung-segmentation"
        echo ""
        echo "📁 Repository contains:"
        echo "  ✅ Professional project structure"
        echo "  ✅ Complete XAI pipeline code"  
        echo "  ✅ Interactive Streamlit dashboard"
        echo "  ✅ Comprehensive documentation"
        echo "  ✅ CI/CD pipeline setup"
        echo "  ✅ Test framework"
        echo ""
        echo "🚀 Next steps:"
        echo "  1. Visit your repository: https://github.com/Mo7aisen/xai-lung-segmentation"
        echo "  2. Star your own repository! ⭐"
        echo "  3. Share with collaborators"
        echo "  4. Start developing!"
    else
        echo "❌ Failed to push to GitHub"
        echo "Please check your authentication and try again"
    fi
else
    echo "❌ Failed to create repository"
    echo "Please check your authentication and try again"
fi