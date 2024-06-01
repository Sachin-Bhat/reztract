# reztract

This is a simple streamlit app that allows you to extract text from a resume in pdf format and then ask questions of it using Llama 3.

To run the app, you need to have a valid Azure account and then follow the steps in the app.

To install the dependencies use:

```
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.lock
```

To run the app type the following:
```
streamlit run src/reztract/app.py --server.runOnSave true
``` 
