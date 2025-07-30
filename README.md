# 🧠 Persona Preference Generator

Ten projekt automatycznie generuje persony, tematy dyskusji, pytania, odpowiedzi i preferencje między odpowiedziami przy użyciu modelu językowego DeepSeek. Na końcu dane są konwertowane do formatu CSV.

## 📁 Struktura projektu

```
.
├── config.json                # Plik konfiguracyjny (ustala co generować)
├── prompts.json              # Zbiór promptów dla LLM
├── data/
│   ├── people.json           # Wygenerowane persony
│   ├── people_extended.json  # Persony z rozszerzonym opisem
│   ├── topics.json           # Tematy rozmów
│   ├── questions.json        # Pytania do tematów
│   ├── answers.json          # Odpowiedzi do pytań
│   ├── preferences.json      # Preferencje person do odpowiedzi
│   └── final_data.csv        # Zbiorczy plik CSV
├── .env                      # Plik z kluczem API do DeepSeek
└── main.py                   # Główna aplikacja
```

## 🚀 Jak to działa?

1. **Persony** – generowane z promptu startowego (`initial_prompt`)
2. **Opisy** – rozszerzane na podstawie kontekstu innych person
3. **Tematy** – generowane na podstawie opisu person
4. **Pytania** – przypisane do tematów
5. **Odpowiedzi** – dwie alternatywne odpowiedzi na każde pytanie
6. **Preferencje** – każda persona wybiera, którą z dwóch odpowiedzi woli
7. **CSV** – końcowe dane zapisywane są w formacie CSV

## ⚙️ Wymagania

- Python 3.10+
- Klucz API do modelu DeepSeek

Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

## 🧪 Użycie

1. Utwórz plik `.env` i dodaj swój klucz API:
   ```
   DEEPSEEK_API_KEY=twoj_klucz_api
   ```

2. Skonfiguruj `config.json`, np.:
   ```json
   {
     "personas": true,
     "extended_descriptions": true,
     "topics": true,
     "questions": true,
     "answers": true,
     "preferences": true
   }
   ```

3. Uruchom skrypt:
   ```bash
   python main.py
   ```

4. Wynik znajdziesz w:
   ```
   data/final_data.csv
   ```

## 🧩 Zastosowania

- Analiza preferencji postaci
- Symulacje społecznych interakcji
- Generowanie danych syntetycznych
- Trening i testowanie modeli NLP

## 📝 Autor

Projekt przygotowany jako część pracy inżynierskiej.  
W razie pytań – zapraszam do kontaktu!