# test.py
# Basic test script for llm_to_json.from_llm_text
from llm_to_json import from_llm_text

def main():
    schema = {"name": "string", "age": "int"}
    llm_text = "Name: Juan Age: 30"
    try:
        result = from_llm_text(llm_text, schema)
        print("Result:", result)
    except Exception as ex:
        print("Error during from_llm_text execution:", ex)

if __name__ == "__main__":
    main()