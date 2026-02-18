Notre objectif est de créer un agent qui apprend à jouer au dilemme du prisonnier selon le cas standard. Notre agent utilise un algorithme Q learning pour apprendre (avec un taux d'exploration epsilon qui décroit au fil du temps). Dans la première partie du code (4premiers) notre agent joue contre 4 différents types d'adversaires :

1- AlwaysCooperate : cet adversaire coopère toujours avec nous 2- AlwaysDefect : Cet adversaire nous trahis toujours 3- Tit-for-tat: Cet adversaire joue au tour N l'action que le joueur a joué au tour N-1 4- Random : Cet adversaire joue aléatoirement entre coopérer et trahir avec une probabilité de 0,5 pour chaque cas

Les résultats sont donnés par un graphique qui apparaît lorsqu'on run le code 4premiers. Les différentes stratégies apprises sont stockées dans une mémoire nommée trained_agent_Qtable

Ensuite, nous avons testé un dernier cas de figure dans lequel notre agent joue contre une copie de lui même. Cependant, notre agent met à jour sa mémoire tout les 50 matchs, mais pas son adversaire. On observe que notre agent prend le dessus sur son adversaire au fur et à mesure du temps. Ce cde est dans la partie nommé Duel
