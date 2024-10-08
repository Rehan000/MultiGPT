from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from prompt_templates import memory_prompt_template
import chromadb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def create_llm(model_path=config['model_path']['small'], model_type=config['model_type'], model_config=config['model_config']):
    """
        Creates and initializes a language model (LLM) using the specified configuration.

        This function initializes a language model (LLM) using the `CTransformers` library.
        The model is configured based on the provided or default `model_path`, `model_type`,
        and `model_config` parameters.

        Args:
            model_path (str, optional): The file path to the model. Defaults to the
                value of `config['model_path']['small']`.
            model_type (str, optional): The type of model to be used. Defaults to
                the value of `config['model_type']`.
            model_config (dict, optional): A dictionary of configuration options
                for the model. Defaults to the value of `config['model_config']`.

        Returns:
            CTransformers: An instance of the language model initialized with the specified settings.
    """
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm


def create_embeddings(embeddings_path=config['embeddings_path']):
    """
       Creates an embedding model using the specified embeddings path.

       This function initializes an embedding model using the `HuggingFaceInstructEmbeddings`
       class, with the model loaded from the specified file path.

       Args:
           embeddings_path (str, optional): The file path to the pre-trained embeddings model.
               Defaults to the value of `config['embeddings_path']`.

       Returns:
           HuggingFaceInstructEmbeddings: An instance of the embeddings model initialized
           with the specified path.
    """
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)


def create_chat_memory(chat_history):
    """
        Creates a conversational memory buffer with a limited window of recent messages.

        This function initializes a `ConversationBufferWindowMemory` object, which
        stores and manages chat history with a sliding window of the most recent
        messages. The memory stores up to `k` recent messages, providing context
        for ongoing conversations.

        Args:
            chat_history (ChatMessageHistory): An instance of chat history that contains
                the conversation's past messages.

        Returns:
            ConversationBufferWindowMemory: An object that maintains a sliding window
            of up to 3 recent messages in the conversation.
    """
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)


def create_prompt_from_template(template):
    """
        Creates a prompt template for generating prompts based on a provided template string.

        This function initializes a `PromptTemplate` object using the provided template string.
        The `PromptTemplate` is used to generate prompts by filling in placeholders
        defined within the template.

        Args:
            template (str): A string containing the template with placeholders for
                dynamic content.

        Returns:
            PromptTemplate: An instance of `PromptTemplate` created from the provided template string.
    """
    return PromptTemplate.from_template(template)


def create_llm_chain(llm, chat_prompt, memory):
    """
        Creates an LLM chain for handling conversational interactions.

        This function initializes an `LLMChain` object, which is used to manage a conversation
        by integrating a language model (LLM), a prompt template, and a memory component.
        The LLM chain facilitates generating responses based on the provided prompt
        and maintaining context through memory.

        Args:
            llm: The language model to be used in the chain.
            chat_prompt (PromptTemplate): The prompt template that defines how the input
                should be formatted before being passed to the LLM.
            memory (ConversationBufferMemory or ConversationBufferWindowMemory):
                The memory component that keeps track of the conversation history.

        Returns:
            LLMChain: An instance of `LLMChain` configured with the specified LLM, prompt, and memory.
    """
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)


def load_normal_chain(chat_history):
    """
        Loads a normal conversational chain using the provided chat history.

        This function initializes a `chatChain` object with the provided chat history,
        setting up a conversational chain for managing interactions based on the past
        conversation.

        Args:
            chat_history (ChatMessageHistory): An object containing the history of
                messages exchanged in the conversation.

        Returns:
            chatChain: An instance of `chatChain` initialized with the provided chat history.
    """
    return chatChain(chat_history)


def load_vectordb(embeddings):
    """
        Loads or initializes a persistent vector database using embeddings.

        This function sets up a persistent vector database using ChromaDB with the provided
        embeddings function. It connects to a ChromaDB instance and loads a collection
        named "pdfs" for storing and retrieving vectorized data.

        Args:
            embeddings: An embedding function or model used to convert data into vector format
                        for storage in the vector database.

        Returns:
            Chroma: An instance of the Chroma class configured with the persistent database client
                    and specified embedding function.
    """
    persistent_client = chromadb.PersistentClient("chroma_db")

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings,
    )

    return langchain_chroma

def load_pdf_chat_chain(chat_history):
    """
        Loads a PDF-specific conversational chain using the provided chat history.

        This function initializes a `pdfChatChain` object with the provided chat history,
        setting up a conversational chain tailored for interactions involving PDF content.

        Args:
            chat_history (ChatMessageHistory): An object containing the history of messages
                exchanged in the conversation.

        Returns:
            pdfChatChain: An instance of `pdfChatChain` initialized with the provided chat history.
    """
    return pdfChatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
    """
        Loads a retrieval-based question-answering chain using the provided language model, memory, and vector database.

        This function initializes a `RetrievalQA` object that combines a language model (LLM) with memory and a retriever
        built from a vector database. The retrieval-based QA chain allows for question answering by fetching relevant
        information from the vector database using the retriever.

        Args:
            llm: The language model to be used for generating answers.
            memory: A memory component that stores the conversation history for contextual understanding.
            vector_db: A vector database object used to retrieve relevant information based on the input query.

        Returns:
            RetrievalQA: An instance of `RetrievalQA` configured with the specified language model, memory, and retriever.
    """
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever=vector_db.as_retriever())

class pdfChatChain:
    """
        A class to manage a PDF-specific conversational AI chain using a language model, vector database, and memory.

        The `pdfChatChain` class is designed to handle and manage conversations involving PDF content using
        a language model (LLM), a retriever for fetching relevant information from a vector database,
        and memory to retain the context of the conversation.

        Attributes:
            memory (ConversationBufferWindowMemory): A memory object that stores a sliding window of recent
                messages in the conversation.
            vector_db (Chroma): A vector database used to store and retrieve relevant information for answering queries.
            llm_chain (RetrievalQA): A retrieval-based question-answering chain that combines the language model,
                memory, and retriever.

        Methods:
            __init__(chat_history):
                Initializes the `pdfChatChain` object by setting up the memory, vector database, and retrieval-based QA chain.

            run(user_input):
                Processes user input through the retrieval-based QA chain and generates a response based on the conversation history.
    """
    def __init__(self, chat_history):
        """
            Initializes the `pdfChatChain` instance.

            Args:
                chat_history (ChatMessageHistory): An object containing the history of messages
                    exchanged in the conversation.

            Initializes:
                - `self.memory`: Creates a conversation memory using the provided chat history.
                - `self.vector_db`: Loads a vector database using embeddings for retrieving relevant information.
                - `self.llm_chain`: Creates a retrieval-based QA chain with the initialized language model, memory, and vector database.
        """
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectordb(create_embeddings())
        llm = create_llm()
        # chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = load_retrieval_chain(llm,  self.memory, self.vector_db)

    def run(self, user_input):
        """
            Runs the conversational AI chain with the provided user input.

            Args:
                user_input (str): The user's input or query.

            Returns:
                str: The generated response from the language model based on the input and conversation history.
        """
        return self.llm_chain.run(query=user_input, history=self.memory.chat_memory.messages, stop=["Human:"])

class chatChain:
    """
        A class to manage a conversational AI chain using a language model, prompt template, and memory.

        The `chatChain` class is designed to handle and manage a conversation using an LLM (Language Model)
        with a prompt template and memory to retain the context of the conversation.

        Attributes:
            memory (ConversationBufferWindowMemory): A memory object that stores a sliding window of recent
                messages in the conversation.
            llm_chain (LLMChain): An LLM chain that manages the interaction between the language model,
                prompt, and memory.

        Methods:
            __init__(chat_history):
                Initializes the `chatChain` object by setting up the memory, language model, and prompt template.

            run(user_input):
                Processes user input through the LLM chain and generates a response based on the conversation history.
    """
    def __init__(self, chat_history):
        """
            Initializes the `chatChain` instance.

            Args:
                chat_history (ChatMessageHistory): An object containing the history of messages
                    exchanged in the conversation.

            Initializes:
                - `self.memory`: Creates a conversation memory using the provided chat history.
                - `self.llm_chain`: Creates an LLM chain with the initialized language model, prompt, and memory.
        """
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        """
            Runs the conversational AI chain with the provided user input.

            Args:
                user_input (str): The user's input or query.

            Returns:
                str: The generated response from the language model based on the input and conversation history.
        """
        return self.llm_chain.run(human_input=user_input, history=self.memory.chat_memory.messages, stop="Human:")
