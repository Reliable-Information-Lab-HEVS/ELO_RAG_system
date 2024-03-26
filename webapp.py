import os
import argparse
from collections import defaultdict

import gradio as gr
import textwiz
from textwiz.templates import GenericConversation
from textwiz.webapp import generator

from helpers import embedding_loader
from templates.template import DEFAULT_SYSTEM_PROMPT
from generation import rag_augmented_generation, retry_rag_augmented_generation

# Disable analytics (can be set to anything except True really, we set it to False for readability)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Chat model
CHAT_MODEL = textwiz.HFCausalModel('zephyr-7B-beta', gpu_rank=0)
# Embedding model
EMBEDDING_MODEL = textwiz.HFEmbeddingModel('SFR-Embedding-Mistral', gpu_rank=1)

# Load embeddings
EMBEDDINGS, EMBEDDINGS_TEXT, EMBEDDINGS_PAGES = embedding_loader.load_embedding('favre')


def get_empty_conversation() -> GenericConversation:
    """Return an empty conversation given the currect model and optional chat template"""
    conv = CHAT_MODEL.get_empty_conversation()
    conv.set_system_prompt(DEFAULT_SYSTEM_PROMPT)
    return conv

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = defaultdict(get_empty_conversation)

# Need to define one logger per user
LOGGERS = defaultdict(gr.CSVLogger)


def rag_generation(conversation: GenericConversation, prompt: str, max_new_tokens: int, do_sample: bool,
                    top_k: int, top_p: float, temperature: float) -> generator[tuple[str, GenericConversation, list[list]]]:
    
    yield from rag_augmented_generation(chat_model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL, db_embeddings=EMBEDDINGS,
                                        db_texts=EMBEDDINGS_TEXT, db_pages=EMBEDDINGS_PAGES, user_query=prompt,
                                        conv=conversation, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                        top_k=top_k, top_p=top_p, temperature=temperature)


def continue_generation(conversation: GenericConversation, additional_max_new_tokens: int, do_sample: bool,
                        top_k: int, top_p: float, temperature: float) -> generator[tuple[GenericConversation, list[list]]]:
    
    yield from textwiz.webapp.continue_generation(model=CHAT_MODEL, conversation=conversation, additional_max_new_tokens=additional_max_new_tokens,
                                                  do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                                  use_seed=False, seed=0)


def retry_rag_generation(conversation: GenericConversation, max_new_tokens: int, do_sample: bool,
                         top_k: int, top_p: float, temperature: float) -> generator[tuple[GenericConversation, list[list]]]:
    
    yield from retry_rag_augmented_generation(chat_model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL, db_embeddings=EMBEDDINGS,
                                              db_texts=EMBEDDINGS_TEXT, db_pages=EMBEDDINGS_PAGES, conv=conversation,
                                              max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                              temperature=temperature)


def clear_chatbot(username: str) -> tuple[GenericConversation, str, list[list]]:
    """Erase the conversation history and reinitialize the elements.

    Parameters
    ----------
    username : str
        The username of the current session if any.

    Returns
    -------
    tuple[GenericConversation, str, list[list]]
        Corresponds to the tuple of components (conversation, output, conv_id)
    """

    # Create new conv object (we need a new unique id)
    conversation = get_empty_conversation()
    if username != '':
        CACHED_CONVERSATIONS[username] = conversation

    return conversation, conversation.to_gradio_format(), conversation.id



def loading(request: gr.Request) -> tuple[GenericConversation, list[list], str, str, dict]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, list[list], str, str, dict]
        Corresponds to the tuple of components (conversation, output, conv_id, username, max_new_tokens)
    """

    # Retrieve username
    if request is not None:
        try:
            username = request.username
        except:
            username = ''
    
    if username is None:
        username = ''
    
    # Get current registered conversation (the defaultdict will provide and register a new empty one if not 
    # already present)
    if username != '':
        actual_conv = CACHED_CONVERSATIONS[username]
        if LOG:
            LOGGERS[username].setup(inputs_to_callback, flagging_dir=f'chatbot_logs/{username}')

    # In this case we do not know the username so we don't store the conversation in cache
    else:
        actual_conv = get_empty_conversation()
        if LOG:
            LOGGERS[username].setup(inputs_to_callback, flagging_dir='chatbot_logs/UNKNOWN')

    conv_id = actual_conv.id
    
    return actual_conv, gr.update(value=actual_conv.to_gradio_format(), label='MathBot'), conv_id, username, gr.update(maximum=CHAT_MODEL.get_context_size())


# Logging functions. We need to define 3 different as we cannot pass the `flag_option` params from inside the demo
def logging_generation(*args):
    """Logging function. Simply flag everything back to the logger."""
    if LOG:
        LOGGERS[args[0]].flag(args, flag_option='generation')

def logging_continuation(*args):
    """Logging function. Simply flag everything back to the logger."""
    if LOG:
        LOGGERS[args[0]].flag(args, flag_option='continuation')

def logging_retry(*args):
    """Logging function. Simply flag everything back to the logger."""
    if LOG:
        LOGGERS[args[0]].flag(args, flag_option='retry')

    

# Define general elements of the UI (generation parameters)
max_new_tokens = gr.Slider(32, 4096, value=2048, step=32, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(16, 1028, value=256, step=16, label='Max additional new tokens',
                           info='New tokens to generate with "Continue last answer".')
do_sample = gr.Checkbox(value=False, label='Sampling', info=('Whether to incorporate randomness in generation. '
                                                            'If not selected, perform greedy search.'))
top_k = gr.Slider(0, 200, value=50, step=5, label='Top-k',
               info='How many tokens with max probability to consider. 0 to deactivate.')
top_p = gr.Slider(0, 1, value=0.90, step=0.01, label='Top-p',
              info='Probability density threshold for new tokens. 1 to deactivate.')
temperature = gr.Slider(0, 1, value=0.4, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')

# Define elements of the chatbot Tab
prompt = gr.Textbox(placeholder='Write your prompt here.', label='Prompt')
output = gr.Chatbot(label='Conversation', height=500)
generate_button = gr.Button('▶️ Submit', variant='primary')
continue_button = gr.Button('🔂 Continue', variant='primary')
retry_button = gr.Button('🔄 Retry', variant='primary')
clear_button = gr.Button('🗑 Clear')

# Initial value does not matter -> will be set correctly at loading time
conversation = gr.State(GenericConversation('</s>'))
# Define NON-VISIBLE elements: they are only used to keep track of variables and save them to the callback (States
# cannot be used in callbacks).
username = gr.Textbox('', label='Username', visible=False)
conv_id = gr.Textbox('', label='Conversation id', visible=False)

# Define the inputs for the main inference
inputs_to_chatbot = [conversation, prompt, max_new_tokens, do_sample, top_k, top_p, temperature]
inputs_to_chatbot_continuation = [conversation, max_additional_new_tokens, do_sample, top_k, top_p, temperature]
inputs_to_chatbot_retry = [conversation, max_new_tokens, do_sample, top_k, top_p, temperature]

# Define inputs for the logging callbacks
inputs_to_callback = [username, output, conv_id, max_new_tokens, max_additional_new_tokens, do_sample,
                      top_k, top_p, temperature]

# Some prompt examples
prompt_examples = [
    "Please write a function to multiply 2 numbers `a` and `b` in Python.",
    "Hello, what's your name?",
    "What's the meaning of life?",
    "How can I write a Python function to generate the nth Fibonacci number?",
    ("Here is my data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' :"
     " [6.1, 5.9, 6.0, 6.1]}. Can you provide Python code to plot a bar graph showing the height of each person?"),
]


demo = gr.Blocks(title='RAG example')

with demo:

    # state variables
    conversation.render()
    username.render()
    conv_id.render()

    # Main UI
    output.render()
    prompt.render()

    with gr.Row():
        generate_button.render()
        continue_button.render()
        retry_button.render()
        clear_button.render()
            
    # Accordion for generation parameters
    with gr.Accordion("Text generation parameters", open=False):
        do_sample.render()
        with gr.Group():
            max_new_tokens.render()
            max_additional_new_tokens.render()
        with gr.Group():
            top_k.render()
            top_p.render()
            temperature.render()

    gr.Markdown("### Prompt Examples")
    gr.Examples(prompt_examples, inputs=prompt)



    # Perform chat generation when clicking the button or pressing enter
    generate_event1 = gr.on(triggers=[generate_button.click, prompt.submit], fn=rag_generation, inputs=inputs_to_chatbot,
                            outputs=[prompt, conversation, output], concurrency_id='generation')
    # Add automatic callback on success
    generate_event1.success(logging_generation, inputs=inputs_to_callback, preprocess=False,
                            queue=False, concurrency_limit=None)
    
    # Continue generation when clicking the button
    generate_event2 = continue_button.click(continue_generation, inputs=inputs_to_chatbot_continuation,
                                            outputs=[conversation, output], concurrency_id='generation')
    # Add automatic callback on success
    generate_event2.success(logging_continuation, inputs=inputs_to_callback, preprocess=False,
                            queue=False, concurrency_limit=None)
    
    # Continue generation when clicking the button
    generate_event3 = retry_button.click(retry_rag_generation, inputs=inputs_to_chatbot_retry,
                                         outputs=[conversation, output], concurrency_id='generation')
    # Add automatic callback on success
    generate_event3.success(logging_retry, inputs=inputs_to_callback, preprocess=False,
                            queue=False, concurrency_limit=None)
    
    # Clear the prompt and output boxes when clicking the button
    clear_button.click(clear_chatbot, inputs=[username], outputs=[conversation, output, conv_id],
                       queue=False, concurrency_limit=None)

    # Change visibility of generation parameters if we perform greedy search
    do_sample.input(lambda value: [gr.update(visible=value) for _ in range(3)], inputs=do_sample,
                    outputs=[top_k, top_p, temperature], queue=False, concurrency_limit=None)
    
    # Correctly display the model and quantization currently on memory if we refresh the page (instead of default
    # value for the elements) and correctly reset the chat output
    loading_events = demo.load(loading, outputs=[conversation, output, conv_id, username, max_new_tokens],
                               queue=False, concurrency_limit=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Playground')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Number of threads that can run for generation (using the GPUs).')
    parser.add_argument('--log', action='store_true',
                        help='If given, will automatically log all interactions.')
    parser.add_argument('--port', type=int, default=7878,
                        help='On which port to deploy the webapp.')
    
    args = parser.parse_args()
    concurrency = args.concurrency
    LOG = args.log
    port = args.port

    print(f'Analytics: {demo.analytics_enabled}')

    demo.queue(default_concurrency_limit=concurrency).launch(server_name='127.0.0.1', server_port=port,
                                                             favicon_path='https://ai-forge.ch/favicon.ico')