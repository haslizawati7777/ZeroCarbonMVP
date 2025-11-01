@echo off
cd /d C:\Users\HP\ZeroCarbonMVP
call .venv\Scripts\activate
start /min cmd /c "python -m streamlit run app\streamlit_app.py --server.address localhost --server.port 8507"
