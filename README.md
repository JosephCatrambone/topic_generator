# Topic Generator

## Summary:
Provides and endpoint which converts a text string into a series of keywords for the given document.

## Usage:
Example usage:
```python
import requests

resp = requests.get(
    "https://example.com/api/v1/keywords", 
    payload={
        "text": "Anthropomorphized ship hull stressed by current events."
    }
)

print(resp.content)
>>> "anthropomorphized ship, current events"
```

## Development:

## Deployment: