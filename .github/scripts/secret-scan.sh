#!/usr/bin/env bash
# PowerTrader AI — Secret / credential leak scanner
# Exits 1 if any suspicious patterns are found in tracked files.
# Run: bash .github/scripts/secret-scan.sh
# CI:  called by docker compose up --exit-code-from tester (test profile)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

FAIL=0

# Files that should NEVER be committed
SECRET_FILES=(
    "r_key.txt"
    "r_secret.txt"
    ".env"
)

echo "=== PowerTrader AI — Secret Scan ==="
echo ""

# 1. Check for committed secret files
echo "[1/3] Checking for committed secret files..."
for f in "${SECRET_FILES[@]}"; do
    if git ls-files --error-unmatch "$f" 2>/dev/null; then
        echo -e "${RED}  FAIL: $f is tracked by git — remove it immediately!${NC}"
        FAIL=1
    else
        echo "  OK: $f not in git"
    fi
done

# 2. Scan committed Python/config files for hardcoded credential patterns
echo ""
echo "[2/3] Scanning for hardcoded credential patterns in tracked files..."

PATTERNS=(
    # Robinhood key patterns
    'r_key\.txt.*=.*[A-Za-z0-9]'
    # Base64 private key blob (min 40 chars of base64)
    '[A-Za-z0-9+/]{40,}={0,2}'
    # Telegram bot token format
    '[0-9]{8,10}:[A-Za-z0-9_-]{35}'
    # Generic "password = " assignments
    'password\s*=\s*["\x27][^"\x27]{4,}'
    # API key assignments with non-empty values
    'api_key\s*=\s*["\x27][A-Za-z0-9]{8,}'
    # AWS-style keys
    'AKIA[0-9A-Z]{16}'
)

EXCLUDE_DIRS=(".venv" "backtest_cache" "backtest_results" "optimizer_results" "hub_data")
EXCLUDE_FLAGS=""
for d in "${EXCLUDE_DIRS[@]}"; do
    EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude-dir=$d"
done

FOUND_PATTERN=0
for pat in "${PATTERNS[@]}"; do
    # Only scan tracked Python, JSON, YAML, txt files
    matches=$(git ls-files '*.py' '*.json' '*.yml' '*.yaml' '*.txt' '*.sh' | \
        xargs grep -lP "$pat" 2>/dev/null || true)
    if [ -n "$matches" ]; then
        # Filter out known-safe files (examples, docs)
        safe_files="requirements.txt CHANGELOG.md README.md .env.example ccxt_config.json.example"
        for m in $matches; do
            skip=0
            for s in $safe_files; do
                if [ "$m" = "$s" ]; then skip=1; break; fi
            done
            if [ $skip -eq 0 ]; then
                echo -e "${RED}  WARN: Pattern '$pat' found in $m${NC}"
                FOUND_PATTERN=1
            fi
        done
    fi
done

if [ $FOUND_PATTERN -eq 0 ]; then
    echo "  OK: no hardcoded credential patterns detected"
fi

# 3. Verify .gitignore covers secrets
echo ""
echo "[3/3] Verifying .gitignore covers secret files..."
GITIGNORE=".gitignore"
REQUIRED_IGNORES=("r_key.txt" "r_secret.txt" ".env")
for entry in "${REQUIRED_IGNORES[@]}"; do
    if grep -qF "$entry" "$GITIGNORE" 2>/dev/null; then
        echo "  OK: $entry in .gitignore"
    else
        echo -e "${RED}  FAIL: $entry not in .gitignore!${NC}"
        FAIL=1
    fi
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}=== Secret scan PASSED ===${NC}"
    exit 0
else
    echo -e "${RED}=== Secret scan FAILED — see above ===${NC}"
    exit 1
fi
