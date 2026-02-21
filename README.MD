## Canadian Drone Rag System

Designed to help aspiring pilots not break laws :3. Datasets come from canadian sources (Transport canada, Government of Canada open data portal). 

AutoImmunePipeline is designed to prevent data poisoning, including jurisdictional poisoning (data from other regions creeping into answer)

QA questions:
    “What are the requirements for advanced drone certification in Canada?”
    “Can I fly at night?”
    “Minimum age for pilot certificate?”
    “Maximum legal altitude in Canada?”

How to run on local machine:

1. Create venv (optional but highly recomended)
```
ensure you are in the desired directory
python -m venv venv
```
2. Activate venv (venv\scripts\activate)
```
venv\scripts\activate
```
3. Install libraries
```
pip install -r requirements.txt
```
4. Add your Gemini API key to a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```
5. Run flask app
```
flask run 
```