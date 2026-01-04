#!/bin/bash

# ============================================================
#   RL-LNS Services Configuration Script
#   
#   Run this script BEFORE starting any training tasks.
#   It will configure: wandb, Gurobi license, HuggingFace mirror
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}   $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "  $1"
}

# ============================================================
# Configuration file path
# ============================================================
CONFIG_FILE="$HOME/.rl_lns_config"
ENV_FILE=".env"

print_header "RL-LNS Services Configuration"

echo ""
echo "This script will help you configure the following services:"
echo "  1. Weights & Biases (wandb) - Experiment tracking"
echo "  2. Gurobi - MILP solver license"
echo "  3. HuggingFace - Model download settings"
echo ""

# ============================================================
# 1. WandB Configuration
# ============================================================
print_header "1. Weights & Biases (wandb)"

echo "WandB is used for experiment tracking and visualization."
echo ""

# Check if wandb is already logged in
if command -v wandb &> /dev/null; then
    CURRENT_WANDB=$(wandb status 2>/dev/null | grep -o "logged in" || echo "")
    if [ -n "$CURRENT_WANDB" ]; then
        print_success "wandb is already logged in"
        read -p "Do you want to reconfigure? (y/N): " reconfigure_wandb
        if [[ ! "$reconfigure_wandb" =~ ^[Yy]$ ]]; then
            echo "Skipping wandb configuration."
            SKIP_WANDB=true
        fi
    fi
fi

if [ "$SKIP_WANDB" != "true" ]; then
    echo ""
    echo "Get your API key from: https://wandb.ai/authorize"
    echo ""
    read -p "Enter your WandB API key (press Enter to skip): " WANDB_API_KEY
    
    if [ -n "$WANDB_API_KEY" ]; then
        # Set environment variable
        export WANDB_API_KEY="$WANDB_API_KEY"
        
        # Try to login
        if command -v wandb &> /dev/null; then
            echo "$WANDB_API_KEY" | wandb login --relogin 2>/dev/null && \
                print_success "wandb login successful" || \
                print_warning "wandb login command failed, but API key is set"
        else
            print_warning "wandb CLI not found. API key will be exported as environment variable."
        fi
        
        # Save to config file
        echo "WANDB_API_KEY=$WANDB_API_KEY" >> "$CONFIG_FILE.tmp"
        print_success "WandB API key configured"
    else
        print_warning "WandB API key skipped"
    fi
fi

# ============================================================
# 2. Gurobi License Configuration
# ============================================================
print_header "2. Gurobi License"

echo "Gurobi is the MILP solver used in this project."
echo ""

# Check if Gurobi license already exists
GUROBI_DEFAULT_LIC="$HOME/gurobi.lic"
if [ -f "$GRB_LICENSE_FILE" ] || [ -f "$GUROBI_DEFAULT_LIC" ]; then
    print_success "Gurobi license file found"
    read -p "Do you want to reconfigure? (y/N): " reconfigure_gurobi
    if [[ ! "$reconfigure_gurobi" =~ ^[Yy]$ ]]; then
        echo "Skipping Gurobi configuration."
        SKIP_GUROBI=true
    fi
fi

if [ "$SKIP_GUROBI" != "true" ]; then
    echo ""
    echo "Options for Gurobi license:"
    echo "  [1] Enter license key (for academic/commercial licenses)"
    echo "  [2] Specify license file path (if you already have gurobi.lic)"
    echo "  [3] Skip"
    echo ""
    read -p "Choose an option (1/2/3): " gurobi_option
    
    case $gurobi_option in
        1)
            echo ""
            echo "Get your license key from: https://www.gurobi.com/downloads/licenses/"
            echo "For academic users: https://www.gurobi.com/academia/academic-program-and-licenses/"
            echo ""
            read -p "Enter your Gurobi license key: " GUROBI_KEY
            
            if [ -n "$GUROBI_KEY" ]; then
                # Run grbgetkey if available
                if command -v grbgetkey &> /dev/null; then
                    echo "Running grbgetkey to activate license..."
                    grbgetkey "$GUROBI_KEY" && \
                        print_success "Gurobi license activated" || \
                        print_error "grbgetkey failed. You may need to run it manually."
                else
                    print_warning "grbgetkey not found. Please run manually after installing Gurobi:"
                    print_info "grbgetkey $GUROBI_KEY"
                fi
            fi
            ;;
        2)
            read -p "Enter the path to your gurobi.lic file: " GUROBI_LIC_PATH
            
            if [ -f "$GUROBI_LIC_PATH" ]; then
                export GRB_LICENSE_FILE="$GUROBI_LIC_PATH"
                echo "GRB_LICENSE_FILE=$GUROBI_LIC_PATH" >> "$CONFIG_FILE.tmp"
                print_success "Gurobi license file path configured"
            else
                print_error "License file not found: $GUROBI_LIC_PATH"
            fi
            ;;
        3)
            print_warning "Gurobi configuration skipped"
            ;;
        *)
            print_warning "Invalid option. Gurobi configuration skipped"
            ;;
    esac
fi

# ============================================================
# 3. HuggingFace Configuration
# ============================================================
print_header "3. HuggingFace Settings"

echo "HuggingFace is used to download pre-trained models (e.g., Qwen)."
echo ""

# HuggingFace Token (optional, for gated models)
echo "HuggingFace token is optional but required for some gated models."
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token (press Enter to skip): " HF_TOKEN

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "HF_TOKEN=$HF_TOKEN" >> "$CONFIG_FILE.tmp"
    
    # Also set HUGGING_FACE_HUB_TOKEN for older versions
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" >> "$CONFIG_FILE.tmp"
    
    # Try to login via CLI
    if command -v huggingface-cli &> /dev/null; then
        echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN" 2>/dev/null && \
            print_success "HuggingFace login successful" || \
            print_warning "huggingface-cli login failed, but token is set as environment variable"
    fi
    print_success "HuggingFace token configured"
else
    print_warning "HuggingFace token skipped"
fi

# HuggingFace Mirror (for China users)
echo ""
echo "For users in China, a mirror endpoint can speed up model downloads."
read -p "Use HuggingFace mirror (hf-mirror.com)? (y/N): " use_hf_mirror

if [[ "$use_hf_mirror" =~ ^[Yy]$ ]]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "HF_ENDPOINT=https://hf-mirror.com" >> "$CONFIG_FILE.tmp"
    print_success "HuggingFace mirror configured"
fi

# ============================================================
# 4. LLM API Configuration
# ============================================================
print_header "4. LLM API Settings"

echo "Configure LLM API for heuristic evolution (EOH)."
echo ""

read -p "Enter LLM API Endpoint (e.g., https://api.deepseek.com): " LLM_API_ENDPOINT
if [ -n "$LLM_API_ENDPOINT" ]; then
    echo "LLM_API_ENDPOINT=$LLM_API_ENDPOINT" >> "$CONFIG_FILE.tmp"
fi

read -p "Enter LLM API Key: " LLM_API_KEY
if [ -n "$LLM_API_KEY" ]; then
    echo "LLM_API_KEY=$LLM_API_KEY" >> "$CONFIG_FILE.tmp"
fi

read -p "Enter LLM Model Name (e.g., deepseek-coder): " LLM_MODEL
if [ -n "$LLM_MODEL" ]; then
    echo "LLM_MODEL=$LLM_MODEL" >> "$CONFIG_FILE.tmp"
fi

print_success "LLM API settings configured"

# ============================================================
# 5. Save Configuration
# ============================================================
print_header "5. Saving Configuration"

# Create .env file in project directory
if [ -f "$CONFIG_FILE.tmp" ]; then
    # Also create local .env file
    cat "$CONFIG_FILE.tmp" > "$ENV_FILE"
    
    # Move to permanent config
    mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    
    print_success "Configuration saved to:"
    print_info "$CONFIG_FILE (global)"
    print_info "$ENV_FILE (project local)"
    
    echo ""
    echo "To load these settings in future sessions, add to your shell profile:"
    echo ""
    echo "  # RL-LNS configuration"
    echo "  [ -f \"$CONFIG_FILE\" ] && export \$(cat \"$CONFIG_FILE\" | xargs)"
    echo ""
else
    print_warning "No configuration was saved."
fi

# ============================================================
# 6. Generate Slurm environment snippet
# ============================================================
print_header "6. Slurm Script Snippet"

echo "Add the following to your Slurm scripts to load the configuration:"
echo ""
echo "------- Copy below this line -------"
cat << 'EOF'
# Load RL-LNS service configuration
if [ -f "$HOME/.rl_lns_config" ]; then
    export $(cat "$HOME/.rl_lns_config" | xargs)
fi
EOF
echo "------- Copy above this line -------"

# ============================================================
# 7. Verification
# ============================================================
print_header "7. Verification"

echo "Checking configured services..."
echo ""

# Check wandb
if [ -n "$WANDB_API_KEY" ] || command -v wandb &> /dev/null && wandb status 2>/dev/null | grep -q "logged in"; then
    print_success "WandB: Configured"
else
    print_warning "WandB: Not configured"
fi

# Check Gurobi
if [ -n "$GRB_LICENSE_FILE" ] && [ -f "$GRB_LICENSE_FILE" ]; then
    print_success "Gurobi: License file configured ($GRB_LICENSE_FILE)"
elif [ -f "$HOME/gurobi.lic" ]; then
    print_success "Gurobi: License file found at $HOME/gurobi.lic"
else
    print_warning "Gurobi: License not configured (may still work if activated system-wide)"
fi

# Check HuggingFace
if [ -n "$HF_TOKEN" ]; then
    print_success "HuggingFace: Token configured"
else
    print_warning "HuggingFace: Token not configured (public models will still work)"
fi

if [ -n "$HF_ENDPOINT" ]; then
    print_success "HuggingFace Mirror: $HF_ENDPOINT"
fi

# ============================================================
# Done
# ============================================================
print_header "Configuration Complete!"

echo ""
echo "You can now start training. The following environment variables are set:"
echo ""
[ -n "$WANDB_API_KEY" ] && echo "  WANDB_API_KEY=****$(echo $WANDB_API_KEY | tail -c 5)"
[ -n "$GRB_LICENSE_FILE" ] && echo "  GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
[ -n "$HF_TOKEN" ] && echo "  HF_TOKEN=****$(echo $HF_TOKEN | tail -c 5)"
[ -n "$HF_ENDPOINT" ] && echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo ""
echo "To apply these settings in the current shell, run:"
echo ""
echo "  source $CONFIG_FILE"
echo ""
echo "Or start a new terminal session."
echo ""
