# test.py
# Basic smoke test for llm_to_toon.from_llm_text
from llm_to_toon import from_llm_text


def main():
    schema = {"name": "string", "age": "int"}
    llm_text = "Name: Juan Age: 30"
    try:
        toon_text = from_llm_text(llm_text, schema)
        print("TOON Result:", toon_text)
    except Exception as ex:
        print("Error during from_llm_text execution:", ex)


if __name__ == "__main__":
    main()
