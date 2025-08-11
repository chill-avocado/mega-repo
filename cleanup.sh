#!/bin/bash

# Script to remove files unrelated to the actual underlying code

# Remove documentation files
find . -name "README.md" -o -name "*.rst" -o -name "*.txt" -not -path "*/\.*" -not -path "*/src/*" -not -path "*/tests/*" -not -name "requirements.txt" -not -name "REQUIREMENTS.txt" -delete
find . -type d -name "docs" -exec rm -rf {} \; 2>/dev/null || true

# Remove license files
find . -name "LICENSE*" -o -name "COPYING*" -delete

# Remove Git-related files
find . -name ".gitignore" -o -name ".gitattributes" -o -name ".gitmodules" -delete
find . -type d -name ".github" -exec rm -rf {} \; 2>/dev/null || true

# Remove editor configuration files
find . -name ".editorconfig" -o -name ".vscode" -o -name ".idea" -delete

# Remove contribution guidelines
find . -name "CONTRIBUTING*" -o -name "CODE_OF_CONDUCT*" -delete

# Remove CI/CD configuration files
find . -name ".travis.yml" -o -name ".circleci" -o -name "appveyor.yml" -o -name "codecov.yml" -o -name ".azure" -delete

# Remove images and media files
find . -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" -o -name "*.svg" -o -name "*.ico" -delete

# Remove large binary files
find . -name "*.pdf" -o -name "*.ttf" -o -name "*.woff" -o -name "*.woff2" -o -name "*.eot" -o -name "*.mp4" -o -name "*.mp3" -o -name "*.wav" -delete
find . -name "*.bin" -o -name "*.h5" -o -name "*.onnx" -o -name "*.pt" -o -name "*.pth" -o -name "*.weights" -delete
find . -name "*.zip" -o -name "*.tar.gz" -o -name "*.tar" -o -name "*.rar" -delete

# Remove empty directories
find . -type d -empty -delete

echo "Cleanup completed!"