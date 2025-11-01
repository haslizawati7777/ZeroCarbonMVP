@echo off
cd /d C:\Users\HP\ZeroCarbonMVP
call .venv\Scripts\activate
python -m streamlit run app\streamlit_app.py --server.address localhost --server.port 8507
echo.
echo Press any key to close...
pause >nul
