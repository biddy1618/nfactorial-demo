from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pathlib import Path


class TextData(BaseModel):
    text: str

model_name = "yeshpanovrustem/xlm-roberta-large-ner-kazakh"
save_directory = Path("./models/kazakh_ner")  # pathlib.Path used here

# Check if the model and tokenizer are already downloaded
if not save_directory.is_dir() or not any(save_directory.iterdir()):
    # Create the directory if it does not exist
    save_directory.mkdir(parents=True, exist_ok=True)

    # Download the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Save the model and tokenizer
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
else:
    # Load the model and tokenizer from the local directory
    tokenizer = AutoTokenizer.from_pretrained(str(save_directory))
    model = AutoModelForTokenClassification.from_pretrained(str(save_directory))

nlp = pipeline("ner", model = model, tokenizer = tokenizer)

tag_list = ['GPE', 'LOCATION', 'PROJECT']

def get_tags(ner_results):
    ners = []
    for ner in ner_results:
        if any(tag in ner['entity'] for tag in tag_list):
            ners.append(ner)
    tags = []
    prev_ner = ners[0]
    tag = prev_ner['word']
    for ner in ners[1:]:
        if prev_ner['end'] >= ner['start']-1:
            tag += ner['word']
        else:
            tags.append(tag)
            tag = ner['word']
        prev_ner = ner

    tags.append(tag)
    tags = [tag.replace('▁', ' ').strip() for tag in tags]
    return tags

app = FastAPI()

@app.post("/get_tags/", response_model=List[str])
async def read_text(data: TextData):
    example = data.text
    ner_results = nlp(example)
    tags = get_tags(ner_results)
    return tags
