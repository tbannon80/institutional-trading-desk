"""
app.py - Streamlit User Interface Entrypoint
Strictly locked to BTCUSDT and SILVERUSDT institutional SMC scope.
Delegates directly to trading_dashboard.py.
"""
import os
import runpy

dashboard_path = os.path.join(os.path.dirname(__file__), "trading_dashboard.py")
runpy.run_path(dashboard_path, run_name="__main__")
