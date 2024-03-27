

# Not clear if system prompt should be in English or French even for conversations in French
DEFAULT_SYSTEM_PROMPT = (
    "You are MathBot, a virtual assistant designed to find relevant informations and provide them to the user."
    "\n\nAlways assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, "
    "prejudiced, or negative content. Ensure replies promote fairness and positivity. Always answer by going straight "
    "to the point, and do not repeat yourself in the conversation."
)


DEFAULT_RAG_PROMPT = """Voici ma question :

######## QUESTION ########
{query}
######## QUESTION ########

Pour t'aider à répondre à cette question, tu as accès aux informations suivantes:

######## INFORMATIONS ########
{knowledge}
######## INFORMATIONS ########

Maintenant, réponds à ma question en t'aidant des informations données plus haut. Si les informations ne te \
permettent pas de répondre correctement à la question, ne les utilise pas."""


# Not clear if those kind of instructions will yield better results in English or French, even for 
# queries only in French
DEFAULT_TASK_DESCRIPTION = 'Given a user query, retrieve relevant passages that answer the query.'
# DEFAULT_TASK_DESCRIPTION = ("A partir de la demande de l'utilisateur, retrouve les passages permettant de "
#                             "répondre à cette demande.")


def formulate_query_for_embedding(query: str, task_description: str = DEFAULT_TASK_DESCRIPTION) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'