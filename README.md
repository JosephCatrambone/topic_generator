# Topic Generator

## Summary:
Provides and endpoint which converts a text string into a series of keywords for the given document.

#### Example usage (Python):
```python
import requests

resp = requests.post(
    "https://127.0.0.1:5000/api/v1/", 
    json={
        "text": "Anthropomorphized ship hull stressed by current events."
    }
)

print(resp.content)
>>> "anthropomorphized ship, current events"
```

#### Example usage (bash):
```bash
$ curl http://127.0.0.1:5000/api/v1/ -X POST -H "Content-Type:application/json" --data '{"text":"Greetings from the other side."}'
{"input_text":"Greetings from the other side.","keywords":["well designs for t"]}
```

## ML Development:

To train a new version of the model...
 - Engage your virtual environment
 - `cd` into the project root (/home/you/topic_generator/, for example)
 - `pip install -r requirements.txt`
 - `python ./modeling/train.py`.

Training data should be a CSV with three columns: text, title, and keywords.  It should be placed in the data directory.

Checkpoints will be created in the modeling directory, along with the finished model.
run_logs will be created in the root directory.

To quickly and interactively play with the trained model, from an ipython notebook one can type `from modeling import predict` and invoke predict with an array of strings.

## API Development:

Run `python ./app.py` from the application root to get a local development server up and running.

## Deployment:

Run `docker build` to generate a self-contained