from langchain_ollama import OllamaLLM
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# No need to load environment variables unless you're using them for something else

# Initialize the Ollama LLM
llm = OllamaLLM(model="llama3.2")  # or llama2, codellama, etc.

# Test the setup
response = llm.invoke("Hello! Are you working?")
print(response)