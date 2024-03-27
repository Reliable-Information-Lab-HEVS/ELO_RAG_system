

# Not clear if system prompt should be in English or French even for conversations in French
# DEFAULT_SYSTEM_PROMPT = (
#     "You are MathBot, a virtual assistant designed to find relevant informations and provide them to the user. Given "
#     "user queries, you answer those queries by looking up relevant informations in the knowledge base provided to you."
#     "\n\nAlways assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, "
#     "prejudiced, or negative content. Ensure replies promote fairness and positivity. Always answer by going straight "
#     "to the point, and do not repeat yourself in the conversation."
# )

# DEFAULT_SYSTEM_PROMPT = (
#     "You are MathBot, a virtual assistant designed to find relevant informations and provide them to the user."
#     "\n\nAlways assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, "
#     "prejudiced, or negative content. Ensure replies promote fairness and positivity. Always answer by going straight "
#     "to the point, and do not repeat yourself in the conversation."
# )


DEFAULT_SYSTEM_PROMPT = (
    "Tu es MathBot, un assistant virtuel spécialisé dans les mathématiques reconnu dans le monde entier. Tu as été créé pour "
    "donner des conseils et répondre aux questions de l'utilisateur sur des concepts mathématiques. Tu es également expert pour "
    "retrouver des informations pertinentes dans des documents et les transmettre à l'utilisateur.\n\nAssiste toujours avec le "
    "plus grand soin, en faisant preuve de respect et en disant la vérité. Réponds avec la plus grande utilité, mais tout en assurant "
    "la sécurité. Evite tout contenu nuisible, non éthique, préjudiciable ou négatif. Veille à ce que tes réponses soient positives "
    "et favorisent l'équité. Réponds toujours en allant droit au but, et ne te répète jamais. Parle uniquement en français."
)


# DEFAULT_RAG_PROMPT = """Voici une question que l'utilisateur veut te poser :

# ######## QUESTION ########
# {query}
# ######## QUESTION ########

# Pour t'aider à répondre à cette question, tu as accès au document suivant :

# ######## DOCUMENT ########
# {knowledge}
# ######## DOCUMENT ########

# Maintenant, réponds à l'utilisateur en t'aidant du document donné plus haut. Si les informations ne te \
# permettent pas de répondre correctement à la question, ne les utilise pas."""


DEFAULT_RAG_PROMPT = """\
######## QUESTION ########
{query}
######## QUESTION ########

######## DOCUMENT ########
{knowledge}
######## DOCUMENT ########

En t'aidant du document donné, réponds à la question. Si le document ne te \
permets pas de répondre correctement à la question, ne l'utilise pas."""



# Not clear if those kind of instructions will yield better results in English or French, even for 
# queries only in French
DEFAULT_TASK_DESCRIPTION = 'Given a user query, retrieve relevant passages that answer the query.'
# DEFAULT_TASK_DESCRIPTION = ("A partir de la demande de l'utilisateur, retrouve les passages permettant de "
#                             "répondre à cette demande.")


def formulate_query_for_embedding(query: str, task_description: str = DEFAULT_TASK_DESCRIPTION) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'




FEW_SHOT_EXAMPLES = (
"""######## QUESTION ########
Aide moi à comprendre les fonctions affines.
######## QUESTION ########

######## DOCUMENT ########
∆xoù∆xest l’accroissement
selon l’axe des xet∆yest l’accroissement selon l’axe des y.
Pour trouver la pente d’une droite, on choisit sur cette droite 2 points (x1;y1)et(x2;y2), puis
on calcule :
a=∆y
∆x=y2−y1
x2−x1
xy
x1 x2y1y2
∆x∆y
Exemple 7.2
Calculer la pente de la droite représentée par le graphe ci-après :
x0y
A
B
Solution
Le point Aa comme coordonnées (0; 3)et le point Bles coordonnées (2; 1). Ainsi :
a=1−3
2−0=−2
2=−1
Remarque
La pente d’une droite s’exprime souvent en pour-cent comme c’est le cas des panneaux rou-
tiers :
Dans le cas d’une pente de 10%, cela signiﬁe que sur une distance horizontale de 100, on
monte d’une hauteur de 10. Attention, une pente de 100%, n’est pas une pente verticale
mais une pente de 45◦.
7.1.3 Droite passant par deux points
Pour déterminer l’équation y=ax+bou la fonction f(x) =ax+bd’une droite passant par
deux points A(x1;y1)etB(x2;y2), on commence par calculer la pente entre ces deux points :
a=y2−y1
x2−x1
puis
b=y1−ax1ou b=y2−ax2
Exemple 7.3
Déterminer l’équation de la droite y=ax+bpassant par les points (1; 5)et(3; 9)
Solution
La pente est donnée par :
a=9−5
3−1=4
2=2
L ’ordonnée à l’origine bpar :
b=5−2×1=3ou b=9−2×3=3
Ainsi l’équation de la droite s’écrit : y=2x+3
7.1.4 Fonction afﬁne par morceaux
Une fonction est dite aﬃne par morceaux si elle est déﬁnie sur plusieurs intervalles disjoints
par des fonctions aﬃnes.
Exemple 7.4
La fonction fdéﬁnie ci-dessous est une fonction aﬃne par morceaux :
f(x) =−x+8,six≥3
2x−1,six<3
f(x)
1
xy
3
La fonction valeur absolue est une représentation très courante de la fonction aﬃne déﬁnie
par morceaux comme suit :
1
xy
######## DOCUMENT ########

En t'aidant du document donné, réponds à la question. Si le document ne te permets pas de répondre correctement à la question, ne l'utilise pas.""",

"""Est-ce qu'il fait beau actuellement ?""",

"""######## QUESTION ########
Qu'est-ce qu'une moyenne harmonique ?
######## QUESTION ########

######## DOCUMENT ########
G=3∏
i=1xfi
i=20,8×50,15×100,05≃2, 48
On constate que la moyenne géométrique est plus basse que la moyenne arithmétique.
Exemple 24.8
Une marchandise de 200 frs a augmenté de 60% la première année et a diminué de
10% l’année suivante. Quelle a été l’augmentation annuelle moyenne ?
Solution
On applique la moyenne géométrique des coeﬃcients multiplicateurs :
G=p
1, 6·0, 9=1, 2
Ce qui correspond à une augmentation annuelle moyenne de 20%.
Moyenne harmonique
Lamoyenne harmonique (H) est utilisée pour les calculs de vitesse moyenne, de taux de
change moyens, c’est-à-dire dans les situations où les valeurs observées sont elles-mêmes ob-
tenues en calculant un rapport.
Pour Nvaleurs individuelles ou regroupées en kclasses, la moyenne harmonique peut se
calculer par l’une ou l’autre des formules suivantes :
H=N
N∑
i=11
xi=N
k∑
i=1ni
xi=1
k∑
i=1fi
xi
Exemple 24.9
Calculer la moyenne harmonique des valeurs individuelles 1 2 3 6 :
Solution
H=4
1
1+1
2+1
3+1
6=4
2=2
Exemple 24.10
Un cycliste a parcouru le tronçon ABde la course à la vitesse de 50 km /h et le tronçon
BCà 20 km /h. Quelle a été sa vitesse moyenne sur tout le parcours ?
30 km 60 kmA
BC
Solution
Les informations peuvent être résumées comme suit :
Vitesse ( xi) 50 20 T otal
Km parcourus ( ni)30 60 90
Vitesse moyenne : H=90
30
50+60
20=90
3, 6=25km/h.
Exercices d’application de la section 24.1
1
[Taux de croissance ]Calculer de tête le taux de croissance des valeurs suivantes :
(a)V0=100etV1=120
(b)V0=120etV1=132
(c)V0=50etV1=65(d)V0=100etV1=80
(e)V0=80etV1=40
(f)V0=80etV1=72
######## DOCUMENT ########

En t'aidant du document donné, réponds à la question. Si le document ne te permets pas de répondre correctement à la question, ne l'utilise pas."""
)


FEW_SHOT_ANSWERS = (
"""Une fonction affine est une fonction qui possède une pente constante sur son graphe, et dont l'équation prend \
la forme y = ax + b. Pour déterminer l'équation d'une droite passant par deux points, il faut calculer la pente entre ces \
deux points et ensuite trouver l'ordonnée à l'origine. La pente est donnée par la formule a = (y2 - y1) / (x2 - x1), et \
l'ordonnée à l'origine b est calculée en remplaçant x par 0 dans l'une des coordonnées des points connus. Une fonction \
affine par morceaux est une fonction définie sur plusieurs intervalles disjoints par des fonctions affines. Cela permet de \
représenter des fonctions complexes de manière plus simple et plus intuitive. L'exemple donné dans le document illustre \
cette notion avec la fonction valeur absolue, qui est une fonction affine par morceaux définie sur les intervalles négatifs \
et positifs.""",

"""Je ne peux pas voir ou entendre le monde physique. J'existe uniquement dans le monde virtuel. Je ne peux pas répondre \
à votre question sur le temps présent et le temps météorologique. Je vous suggère de consulter une source de météorologie \
ou d'utiliser un service de prévision météorologique pour obtenir des informations sur le temps actuel.""",

"""La moyenne harmonique est utilisée pour calculer la vitesse moyenne ou les taux de change moyens lorsque \
les valeurs observées sont obtenues en calculant un rapport. Elle peut se calculer à l'aide de trois formules \
différentes, en fonction des valeurs individuelles ou regroupées en classes. \

Pour illustrer, dans l'exemple 24.10 on utilise la moyenne harmonique afin de déterminer la vitesse moyenne \
(sur la totalité du trajet) d'un cycliste qui a parcouru un tronçon AB à 50 km/h et un tronçon BC à 20 km/h."""
)