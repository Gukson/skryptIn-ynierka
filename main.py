import os
import json
import re
from typing import List

from tqdm import tqdm
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") #load api key from .env file

class Person(BaseModel):
    persona: str = Field(description="type of the character")
    description: str = Field(description="description of the character")

def load_api_key() -> str:
    load_dotenv()
    return os.getenv("DEEPSEEK_API_KEY")

def load_config() -> dict:
    """Wczytywanie konfiguracji z pliku JSON."""
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        return json.load(f)

def load_prompts(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def load_json_data(file_path: str) -> List[dict]:
    """Wczytuje dane JSON z pliku."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def init_llm(api_key: str) -> ChatDeepSeek:
    return ChatDeepSeek(model="deepseek-chat", api_key=api_key)

def extract_json_list(text: str) -> str:
    """Wyciąga listę JSON z tekstu."""
    match = re.search(r'\[[\s\S]*?\]', text)
    return match.group(0) if match else "[]"

def parse_personas(response_text: str) -> List[dict]:
    """Analizuje tekst odpowiedzi w celu wyodrębnienia person jako listy słowników."""
    parser = JsonOutputParser(pydantic_object=Person)
    json_array = extract_json_list(response_text)
    lines = json_array.strip().split("\n")[1:-1]

    def enrich_with_id(i, line):
        item = parser.invoke(line)
        item["id"] = i
        return item

    return [enrich_with_id(i, line) for i, line in enumerate(lines)]


def generate_personas(prompt: str, llm: ChatDeepSeek) -> List[dict]:
    response = llm.invoke(prompt)
    return parse_personas(response.content)


def save_to_file(data: List[dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_person(person, description_field="description"):
    """Zwraca reprezentację ciągu znaków persony do tworzenia monitu."""
    return f'{person["persona"]}: {person[description_field]}'

def extend_descriptions(people, llm: ChatDeepSeek, prompt) -> List[dict]:
    """Rozszerza opisy person przy użyciu LLM."""
    prompt_template = PromptTemplate.from_template(prompt)
    parser = JsonOutputParser()
    chain = prompt_template | llm | parser
    for i, person in tqdm(enumerate(people), total=len(people), desc="Extending descriptions"):
        suffix = people[i+1:] if i+1 < len(people) else []
        personas = people[:i] + suffix
        all_descriptions = "\n".join(get_person(p) for p in personas)
        current_description = get_person(person)
        response = chain.invoke({"personas": all_descriptions, "current": current_description})
        person["long_description"] = response["description"]
    return people

def generate_topics(people, llm: ChatDeepSeek, prompt: str) -> List[dict]:
    descriptions = "\n".join(get_person(p) for p in people)
    response = llm.invoke(prompt.format(personas=descriptions))
    match = re.search(r"```json\n([\s\S]+?)```", response.content)
    if not match:
        raise ValueError("Nie znaleziono bloku kodu z listą JSON.")

    json_block = match.group(1).strip()

    data = json.loads(json_block)
    return data

def main():
    """Główna funkcja programu."""
    api_key = load_api_key()
    prompts = load_prompts("prompts.json")
    llm = init_llm(api_key)
    config = load_config()

    # Jeśli ustawione na True – generuj persony od zera, inaczej wczytaj z pliku
    if config["personas"]:
        print("Tworzenie person...")
        prompt = prompts["initial_prompt"]
        personas = generate_personas(prompt, llm)
        save_to_file(personas, "data/people.json")
    else:
        personas = load_json_data("data/people.json")


    # Jeśli extended_descriptions jest ustawione na True, rozszerz opisy person, inaczej wczytaj z pliku
    if config["extended_descriptions"]:
        print("Rozszerzanie opisów...")
        description_prompt = prompts["description_prompt"]
        extended_personas = extend_descriptions(personas, llm, description_prompt)
        save_to_file(extended_personas, "data/people_extended.json")
    else:
        extended_personas = load_json_data("data/people_extended.json")

    # Jeśli topics jest ustawione na True, generuj tematy, inaczej wczytaj z pliku
    if config["topics"]:
        print("Generowanie tematów...")
        topic_prompt = prompts["topics_prompt"]
        topics = generate_topics(extended_personas, llm, topic_prompt)
        save_to_file(topics, "data/topics.json")
    else:
        topics = load_json_data("data/topics.json")

if __name__ == '__main__':
    main()