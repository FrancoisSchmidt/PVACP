import socket
import select
from Codes.Simulation.sailboat import *

hote = '192.168.43.183'
hote = ''
port = 12800

connexion_principale = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # Création de la socket
connexion_principale.bind((hote, port))
connexion_principale.listen(5)                                          # En écoute
print("Le serveur écoute à présent sur le port {}".format(port))

serveur_lance = True
clients_connectes = []  # Liste de clients connectés (sera limitée à 1 dans notre cas

# Initialisation de constantes
dt = 0.1
awind, ψ = 50, pi/2
listex, listey = [], []
x = array([[0, 0, -3, 3, 0]]).T  # x=(x,y,θ,v,w)
ax=init_figure(-100,100,-60,60)


client = False  # Pour l'instant pas de client connecté
clients_a_lire = []


while client == False:      # Tant qu'il n'y a pas de client
    connexions_demandees, wlist, xlist = select.select([connexion_principale],      # On est en écoute
                                                       [], [], 0.025)
    for connexion in connexions_demandees:
        connexion_avec_client, infos_connexion = connexion.accept()     # Si on peut on se connecte
        # On ajoute le socket connecté à la liste des clients
        clients_connectes.append(connexion_avec_client)
        client = True

while len(clients_a_lire)==0:
    clients_a_lire, wlist, xlist = select.select(clients_connectes,
                                                 [], [], 0.025)



commande = 0
while serveur_lance:    #On travaille uniquement avec le client qui a réussi à se connecter
    for client in clients_a_lire:
        # Client est de type socket
        msg_recu = client.recv(1024)

        # Peut planter si le message contient des caractères spéciaux
        msg_recu = int(msg_recu.decode('utf8')[-1])
        print("Reçu :", msg_recu)
        if msg_recu != "fin":

            clear(ax)
            update_ax(x,ax, commande)

            a = array([[x[0][0]], [x[1][0]]])
            b = array([[x[0][0] + cos(x[2][0])],
                        [x[1][0] + sin(x[2][0])]])

            # En fonction du message reçu, on modifie la commande à donner au voilier
            if msg_recu == 2:
                commande = 1
            elif msg_recu == 1:
                commande = -1
            elif msg_recu == 3:
                commande = 0
            print("Commande :", commande)


            listex.append(a[0, 0]), listex.append(b[0, 0])
            listey.append(a[1, 0]), listey.append(b[1, 0])
            plot(listex, listey, 'blue')  # afficher la trace complète au fur et à mesure (tous les segments)

            u = control(x, a, b, commande)      # Donne la commande à suivre au voilier

            xdot, δs = f(x, u)
            x = x + dt * xdot
            draw_sailboat(x, δs, u[0, 0], ψ, awind)


print("Fermeture des connexions")
for client in clients_connectes:
    client.close()

connexion_principale.close()