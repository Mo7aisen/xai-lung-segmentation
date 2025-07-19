#!/bin/bash
# GitHub Authentication Helper

GH_CLI="./gh_2.76.0_linux_amd64/bin/gh"

echo "🔐 GitHub Authentication Options"
echo "==============================="
echo ""
echo "Option 1: Personal Access Token (Recommended)"
echo "  1. Go to: https://github.com/settings/tokens"
echo "  2. Generate new token with 'repo' scope"
echo "  3. Copy the token"
echo "  4. Run: echo 'YOUR_TOKEN' | $GH_CLI auth login --with-token"
echo ""
echo "Option 2: Web Browser"
echo "  1. Run: $GH_CLI auth login --web"
echo "  2. Follow the instructions"
echo ""
echo "Option 3: Username/Password (if enabled)"
echo "  1. Run: $GH_CLI auth login"
echo "  2. Select GitHub.com"
echo "  3. Enter credentials"
echo ""

read -p "Which option would you like to use? (1/2/3): " choice

case $choice in
    1)
        echo "📝 Please enter your Personal Access Token:"
        read -s token
        echo "$token" | $GH_CLI auth login --with-token
        ;;
    2)
        echo "🌐 Opening web authentication..."
        $GH_CLI auth login --web
        ;;
    3)
        echo "🔑 Starting interactive authentication..."
        $GH_CLI auth login
        ;;
    *)
        echo "❌ Invalid option. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Testing authentication..."
if $GH_CLI auth status; then
    echo "🎉 Authentication successful!"
    echo "Now run: ./github-setup.sh"
else
    echo "❌ Authentication failed. Please try again."
fi