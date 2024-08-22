import json
from datetime import datetime
from langchain.schema.messages import HumanMessage, AIMessage


def save_chat_history_json(chat_history, file_path):
    """
       Saves the chat history to a JSON file.

       This function takes a list of chat messages, converts each message to a dictionary
       representation, and then saves the resulting list of dictionaries as a JSON file
       at the specified file path.

       Args:
           chat_history (list): A list of chat message objects, each containing details
           like message content, type, timestamp, etc.
           file_path (str): The path where the JSON file will be saved.

       Side Effects:
           - Writes the chat history to a JSON file at the specified file path.
    """
    with open(file_path, "w") as f:
        json_data = [message.dict() for message in chat_history]
        json.dump(json_data, f)


def load_chat_history_json(file_path):
    """
        Loads chat history from a JSON file and reconstructs the chat messages.

        This function reads a JSON file containing chat history, parses it, and
        reconstructs the chat messages as a list of `HumanMessage` and `AIMessage`
        objects based on their type.

        Args:
            file_path (str): The path to the JSON file containing the chat history.

        Returns:
            list: A list of chat message objects (`HumanMessage` or `AIMessage`),
            reconstructed from the JSON data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            JSONDecodeError: If the file contains invalid JSON.
    """
    with open(file_path, "r") as f:
        json_data = json.load(f)
        messages = [HumanMessage(**message) if message["type"] == "human" else AIMessage(**message) for message in
                    json_data]
        return messages


def get_timestamp():
    """
        Generates a timestamp of the current date and time.

        This function returns the current date and time formatted as a string
        in the format "DD-MM-YYYY, HH:MM:SS".

        Returns:
            str: A string representing the current timestamp in the format "DD-MM-YYYY, HH:MM:SS".
    """
    return datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
