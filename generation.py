import re

import torch
import gradio as gr
import textwiz
from textwiz.templates import GenericConversation
from textwiz.webapp import generator, chat_generation, continue_generation

from templates import template
from helpers import utils


def rag_augmented_generation(chat_model: textwiz.HFCausalModel, embedding_model: textwiz.HFEmbeddingModel,
                             db_embeddings: torch.Tensor, db_texts: list[str], db_pages: list[dict],
                             user_query: str, conv: GenericConversation, chatbot_output: list[list], similarity_threshold: float,
                             max_new_tokens: int, do_sample: bool, top_k: int, top_p: float, temperature: float,
                             **kwargs) -> generator[tuple[str, GenericConversation, list[list]]]:
    
    formatted_query = template.formulate_query_for_embedding(user_query.strip())
    query_embedding = embedding_model(formatted_query)
    # Convert to tensor for efficient search on GPU
    query_embedding = torch.tensor(query_embedding, device=db_embeddings.device, requires_grad=False)

    # Use exact search with torch since size is small (if more data, switch to indexing system such as FAISS)
    scores = torch.nn.functional.cosine_similarity(query_embedding, db_embeddings)

    # Find k biggest scores (here we only use k=1)
    similarity, indices = torch.topk(scores, k=1)

    # If we don't use RAG, do not show pdf element
    pdf_element = gr.update(visible=False)

    # If unrelated to the embeddings we have, just pass on the query
    if similarity < similarity_threshold:
        chat_model_input = user_query
    # If related, find corresponding book entry and format prompt
    else:
        # Find corresponding text and book pages
        knowledge = db_texts[indices.item()]
        page_mapping = db_pages[indices.item()]
        book = list(page_mapping.keys())[0]
        pdf_path = utils.create_temporary_pdf(book, page_mapping[book])
        pdf_element = gr.update(value=pdf_path, visible=True)

        # Create model input
        chat_model_input = template.DEFAULT_RAG_PROMPT.format(query=user_query.strip(), knowledge=knowledge.strip())

    chatbot_output.append([user_query, None])

    # Yield tokens for gradio
    for input, conv, chatbot in chat_generation(model=chat_model, conversation=conv, prompt=chat_model_input, max_new_tokens=max_new_tokens,
                                                do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature, use_seed=False,
                                                seed=0, **kwargs):
        chatbot_output[-1][1] = chatbot[-1][1]
        yield input, conv, chatbot_output, chatbot_output, pdf_element
    


def retry_rag_augmented_generation(chat_model: textwiz.HFCausalModel, embedding_model: textwiz.HFEmbeddingModel,
                                   db_embeddings: torch.Tensor, db_texts: list[str], db_pages: list[dict],
                                   conversation: GenericConversation, chatbot_output: list[list], similarity_threshold: float,
                                   max_new_tokens: int, do_sample: bool, top_k: int, top_p: float, temperature: float, **kwargs):

    if len(conversation) == 0:
        gr.Warning(f'You cannot retry generation on an empty conversation.')
        yield conversation, conversation.to_gradio_format()
        return
    if conversation.model_history_text[-1] is None:
        gr.Warning('You cannot retry generation on an empty last turn')
        yield conversation, conversation.to_gradio_format()
        return

    # Remove last turn
    last_prompt = conversation.user_history_text[-1]
    _ = conversation.user_history_text.pop(-1)
    _ = conversation.model_history_text.pop(-1)
    _ = chatbot_output.pop(-1)

    # Extract last user_query from formatted prompt
    if '######## QUESTION ########' in last_prompt:
        match = re.search(r'######## QUESTION ########\n(.*)\n######## QUESTION ########', last_prompt, re.DOTALL)
        user_query = match.group(1)
    else:
        user_query = last_prompt

    # Yield from chat_generation, but remove first value
    for _, conv, output, output1, pdf in rag_augmented_generation(chat_model=chat_model, embedding_model=embedding_model, db_embeddings=db_embeddings,
                                                          db_texts=db_texts, db_pages=db_pages, user_query=user_query,
                                                          conv=conversation, chatbot_output=chatbot_output, similarity_threshold=similarity_threshold, max_new_tokens=max_new_tokens,
                                                          do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature, **kwargs):
        yield conv, output, output1, pdf



def continuation(chat_model: textwiz.HFCausalModel, conversation: GenericConversation, chatbot_output: list[list],
                 additional_max_new_tokens: int, do_sample: bool, top_k: int, top_p: float,
                 temperature: float, **kwargs):
    
    for conv, output in continue_generation(chat_model, conversation, additional_max_new_tokens, do_sample,
                                            top_k, top_p, temperature, False, 0, **kwargs):
        chatbot_output[-1][1] = output[-1][1]
        yield conv, chatbot_output, chatbot_output