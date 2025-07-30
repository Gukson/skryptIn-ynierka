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
api_key = os.getenv("DEEPSEEK_API_KEY") #wczytanie klucza API z pliku .env

class Person(BaseModel):
    """Model reprezentujący personę."""
    persona: str = Field(description="type of the character")
    description: str = Field(description="description of the character")

def load_api_key() -> str:
    """Wczytuje klucz API z pliku .env."""
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
    """Wczytuje prompty z pliku JSON."""
    with open(file_path, 'r') as file:
        return json.load(file)

def load_json_data(file_path: str) -> dict:
    """Wczytuje dane JSON z pliku."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def init_llm(api_key: str) -> ChatDeepSeek:
    """Inicjalizuje model LLM DeepSeek Chat."""
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
    """Generuje persony przy użyciu LLM."""
    response = llm.invoke(prompt)
    return parse_personas(response.content)


def save_to_file(data, filename: str):
    """Zapisuje dane do pliku JSON."""
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
    """Generuje tematy na podstawie person przy użyciu LLM."""
    descriptions = "\n".join(get_person(p) for p in people)
    response = llm.invoke(prompt.format(personas=descriptions))
    match = re.search(r"```json\n([\s\S]+?)```", response.content)
    if not match:
        raise ValueError("Nie znaleziono bloku kodu z listą JSON.")

    json_block = match.group(1).strip()

    data = json.loads(json_block)
    return data

def generate_questions(people, topics, llm: ChatDeepSeek, prompt: str) -> dict:
    """Generuje pytania na podstawie tematów i person przy użyciu LLM."""
    all_questions = {}
    index = 0
    descriptions = "\n".join(get_person(p) for p in people)
    template = PromptTemplate.from_template(prompt)
    chain = template | llm

    for topic_id, topic in tqdm(enumerate(topics), total=len(topics), desc="Generating questions"):
        response = chain.invoke({
            "personas": descriptions,
            "topic": topic["topic"],
            "description": topic["description"]
        })

        match = re.search(r"```json\n([\s\S]+?)```", response.content)
        if not match:
            raise ValueError("Nie znaleziono bloku kodu z listą JSON.")

        json_block = match.group(1).strip()
        questions = json.loads(json_block)

        for q in questions:
            all_questions[index] = {
                "id": index,  # <–– dodajemy ID wewnątrz
                "topic": topic_id,
                "question": q["question"]
            }
            index += 1

    return all_questions

def generate_answers(prompt, personas, questions, llm):
    """Generuje odpowiedzi na pytania przy użyciu LLM."""
    descriptions = "\n".join(get_person(p) for p in personas)
    template = PromptTemplate.from_template(prompt)
    chain = template | llm
    for question_id, item in tqdm(questions.items(), total=len(questions), desc="Generating answers"):
        response = chain.invoke({"personas": descriptions, "question": item["question"]})
        questions[question_id]["answers"] = response.content
    return questions

def parse_answers(answers) -> dict:
    """Parsuje odpowiedzi z formatu JSON do słownika."""
    parser = JsonOutputParser()
    for id_, item in tqdm(answers.items()):
        raw_output = item["answers"]
        try:
            out = parser.invoke(raw_output)
            answers[id_]["answer1"] = out["answer1"]
            answers[id_]["answer2"] = out["answer2"]
            del answers[id_]["answers"]
        except Exception:
            output = re.findall(r'"answer\d+":\s*"(.*?)"(?:,?\s*|\s*$)', raw_output)
            if len(output) != 2:
                print(id_)
                print(raw_output)
                print(output)
            for i, out in enumerate(output):
                answers[id_][f"answer{i + 1}"] = out
    return answers

def generate_preferences(people, answers, prompt, llm: ChatDeepSeek) -> List[dict]:
    """Generuje preferencje na podstawie person i odpowiedzi przy użyciu LLM."""
    prompt = PromptTemplate.from_template(prompt)
    chain = prompt | llm
    responses = []
    for _, item in tqdm(answers.items(), total=len(answers), desc="Generating preferences"):
        for person in people:
            curr_item = {}
            description = get_person(person)
            response = chain.invoke(
                {"persona": description, "question": item["question"], "answer1": item["answer1"],
                 "answer2": item["answer2"]})
            curr_item["person"] = person
            curr_item["question"] = item
            curr_item["response"] = response.content
            responses.append(curr_item)
    return responses

def maybe_generate_personas(prompts, config, llm):
    """Generuje persony, jeśli jest to wymagane przez konfigurację."""
    if config["personas"]:
        print("Tworzenie person...")
        personas = generate_personas(prompts["initial_prompt"], llm)
        save_to_file(personas, "data/people.json")
    else:
        personas = load_json_data("data/people.json")
        if not personas:
            raise ValueError("Nie znaleziono pliku z personami. Upewnij się, że konfiguracja jest poprawna lub wygeneruj persony.")
    return personas

def maybe_generate_extended_descriptions(prompts, config, personas, llm):
    """Rozszerza opisy person, jeśli jest to wymagane przez konfigurację."""
    if config["extended_descriptions"]:
        print("Rozszerzanie opisów...")
        description_prompt = prompts["description_prompt"]
        extended_personas = extend_descriptions(personas, llm, description_prompt)
        save_to_file(extended_personas, "data/people_extended.json")
    else:
        extended_personas = load_json_data("data/people_extended.json")
        if not extended_personas:
            raise ValueError("Nie znaleziono pliku z rozszerzonymi opisami person. Upewnij się, że konfiguracja jest poprawna lub wygeneruj rozszerzone opisy.")

    return extended_personas

def maybe_generate_topics(prompts, config, extended_personas, llm):
    """Generuje tematy, jeśli jest to wymagane przez konfigurację."""
    if config["topics"]:
        print("Generowanie tematów...")
        topic_prompt = prompts["topics_prompt"]
        topics = generate_topics(extended_personas, llm, topic_prompt)
        save_to_file(topics, "data/topics.json")
    else:
        topics = load_json_data("data/topics.json")
        if not topics:
            raise ValueError("Nie znaleziono pliku z tematami. Upewnij się, że konfiguracja jest poprawna lub wygeneruj tematy.")
    return topics

def maybe_generate_questions(prompts, config, extended_personas, topics, llm):
    """Generuje pytania, jeśli jest to wymagane przez konfigurację."""
    if config["questions"]:
        print("Generowanie pytań...")
        question_prompt = prompts["questions_prompt"]
        questions = generate_questions(extended_personas, topics, llm, question_prompt)
        save_to_file(questions, "data/questions.json")
    else:
        questions = load_json_data("data/questions.json")
        if not questions:
            raise ValueError("Nie znaleziono pliku z pytaniami. Upewnij się, że konfiguracja jest poprawna lub wygeneruj pytania.")
    return questions

def maybe_generate_answers(prompts, personas, config, questions, llm):
    """Generuje odpowiedzi na pytania, jeśli jest to wymagane przez konfigurację."""
    if config["answers"]:
        answer_prompt = prompts["answers_prompt"]
        answers = generate_answers(answer_prompt, personas, questions, llm)
        parsed_answers = parse_answers(answers)
        save_to_file(parsed_answers, "data/answears.json")
    else:
        parsed_answers = load_json_data("data/answears.json")
        if not questions:
            raise ValueError("Nie znaleziono pliku z odpowiedziami. Upewnij się, że konfiguracja jest poprawna lub wygeneruj odpowiedzi.")
    return parsed_answers

def maybe_generate_preferences(personas, answers, prompts, config, llm):
    """Generuje preferencje, jeśli jest to wymagane przez konfigurację."""
    if config["preferences"]:
        print("Generowanie preferencji...")
        preference_prompt = prompts["preferences_prompt"]
        answers = load_json_data("data/answears.json")
        preferences = generate_preferences(personas, answers, preference_prompt, llm)
        save_to_file(preferences, "data/preferences.json")
    else:
        preferences = load_json_data("data/preferences.json")
        if not preferences:
            raise ValueError("Nie znaleziono pliku z preferencjami. Upewnij się, że konfiguracja jest poprawna lub wygeneruj preferencje.")
    return preferences

def main():
    """Główna funkcja programu."""
    api_key = load_api_key()
    prompts = load_prompts("prompts.json")
    llm = init_llm(api_key)
    config = load_config()

    personas = maybe_generate_personas(prompts, config, llm)
    extend_personas = maybe_generate_extended_descriptions(prompts, config, personas, llm)
    topics = maybe_generate_topics(prompts, config, extend_personas, llm)
    questions = maybe_generate_questions(prompts, config, extend_personas, topics, llm)
    answers = maybe_generate_answers(prompts, personas, config, questions, llm)
    preferences = maybe_generate_preferences(extend_personas, answers, prompts, config, llm)

if __name__ == '__main__':
    main()