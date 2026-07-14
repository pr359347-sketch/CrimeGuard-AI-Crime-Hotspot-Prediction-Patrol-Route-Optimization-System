#!/usr/bin/env pwsh
# PowerShell helper to finalize the reorganization: commit & push
Write-Host "Finalizing reorganization: creating branch, committing, and pushing..."

git checkout -b reorganize-structure
git add -A
try {
    git commit -m "Reorganize: move files into 01.app..06.images and canonicalize 05.src/src; remove duplicates" | Out-Null
    Write-Host "Committed changes."
} catch {
    Write-Host "No commit performed (possibly nothing to commit).";
}

Write-Host "Pushing branch to origin (may prompt for credentials)..."
git push -u origin reorganize-structure

Write-Host "Done. If the push failed, run the above commands manually or check your credentials." 
