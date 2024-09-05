from gradio_client import Client

SIMILARITY_THRESHOLD = 0.65
MAX_NEW_TOKENS = 2048
MAX_ADDITIONAL_NEW_TOKENS = 1024
DO_SAMPLE = True
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.3

client = Client("https://elo.ai-forge.ch")
# Handle preparation of state variables
client.predict(api_name="/loading")

def generation(prompt: str):
    """Main generation API call."""
    job = client.submit(prompt, SIMILARITY_THRESHOLD, MAX_NEW_TOKENS, DO_SAMPLE, TOP_K, TOP_P, TEMPERATURE,
                        api_name="/rag_generation")
    for _, conversation, pdf_link in job:
        yield conversation, f"https://elo.ai-forge.ch/file={pdf_link}"


def retry_generation():
    """Option to retry the last turn of the conversation."""
    job = client.submit(SIMILARITY_THRESHOLD, MAX_NEW_TOKENS, DO_SAMPLE, TOP_K, TOP_P, TEMPERATURE,
                        api_name="/retry_rag_generation")
    for conversation, pdf_link in job:
        yield conversation, f"https://elo.ai-forge.ch/file={pdf_link}"


def continue_generation():
    """Option to continue the last turn of the conversation if it ended too early."""
    job = client.submit(MAX_ADDITIONAL_NEW_TOKENS, DO_SAMPLE, TOP_K, TOP_P, TEMPERATURE,
                        api_name="/continue_rag_generation")
    for conversation in job:
        yield conversation

# client.reset_session()