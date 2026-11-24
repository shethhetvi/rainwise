#!/usr/bin/env bash
# ==============================================================================
# RAINWISE AI — 1-CLICK DEMO LAUNCHER SCRIPT
# ==============================================================================
# Run this script at the time of your demo / viva presentation:
#   ./run_demo.sh
# ==============================================================================

# Move to the script's directory
cd "$(dirname "$0")"

echo "========================================================================"
echo "🌊  RAINWISE AI — MASTER COMMAND CENTER DEMO LAUNCHER"
echo "========================================================================"
echo "📍 Working Directory: $(pwd)"
echo "🐍 Python Environment: /Applications/miniconda3/bin/python3"

# 1. Clean up any stale Streamlit instances running on port 8501
echo "🔍 Checking port 8501..."
PID_8501=$(lsof -ti:8501 2>/dev/null)
if [ -n "$PID_8501" ]; then
    echo "⚠️  Port 8501 in use by PID $PID_8501. Terminating previous process..."
    kill -9 $PID_8501 2>/dev/null
    sleep 1
fi

# 2. Set environment paths
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MPLCONFIGDIR="/tmp"
export HOME="/tmp"

# 3. Detect Streamlit binary
if [ -f "/Applications/miniconda3/bin/streamlit" ]; then
    STREAMLIT_BIN="/Applications/miniconda3/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
    STREAMLIT_BIN="$(command -v streamlit)"
else
    STREAMLIT_BIN="python3 -m streamlit"
fi

echo "🚀 Starting Master Command Center on http://localhost:8501 ..."
echo "------------------------------------------------------------------------"
echo "💡 PROFESSOR DEMO TIPS:"
echo "   • Tab 1: Select '27-Aug-2024 (Vadodara)' or '08-Aug-2006 (Surat)'"
echo "   • Tab 3: Click 'COMPUTE AXIOMATIC ATTRIBUTIONS' (Captum XAI)"
echo "   • Tab 4: Show PINN 0.0% violations vs Standard DNN 29.2% violations"
echo "   • Tab 5: Select 'Tapi Basin' -> Move Ukai Dam to 900,000 cusecs (Surat 2006)"
echo "   • Tab 6: Click 'RUN SPATIO-TEMPORAL RADAR NOWCAST' (ConvLSTM Nowcasting)"
echo "------------------------------------------------------------------------"

# 4. Launch Streamlit
exec "$STREAMLIT_BIN" run app.py --server.port 8501
