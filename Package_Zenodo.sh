#!/bin/bash
echo "Packaging Research Bundle..."

# Clean up results folder retaining ONLY the images
find Results -type f ! -name '*.png' -delete

# Create the archive bundling the requested items and mandatory system files
zip -r SEDE_Optimizer_v2_Research_Bundle.zip \
    src/ \
    Dockerfile \
    requirements.txt \
    Results/*.png \
    runner.py \
    SEDE.py \
    setup.py \
    pyproject.toml

echo "✅ Bundle Created: SEDE_Optimizer_v2_Research_Bundle.zip"
