import socket
import select
from Codes.Simulation.sailboat import *

hote = '192.168.43.183'
hote = ''
port = 12800

connexion_principale = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connexion_principale.bind((hote, port))
connexion_principale.listen(5)
print("Le serveur écoute à présent sur le port {}".format(port))

serveur_lance = True
clients_connectes = []

x = array([[0, 0, -3, 3, 0]]).T  # x=(x,y,θ,v,w)

ax = init_figure(-60, 60, -60, 60)


client = False
clients_a_lire = []


while client == False:
    connexions_demandees, wlist, xlist = select.select([connexion_principale],
                                                       [], [], 0.025)
    for connexion in connexions_demandees:
        connexion_avec_client, infos_connexion = connexion.accept()
        # On ajoute le socket connecté à la liste des clients
        clients_connectes.append(connexion_avec_client)
        client = True

while len(clients_a_lire)==0:
    clients_a_lire, wlist, xlist = select.select(clients_connectes,
                                                 [], [], 0.025)




while serveur_lance:
    for client in clients_a_lire:
        # Client est de type socket
        msg_recu = client.recv(1024)

        # Peut planter si le message contient des caractères spéciaux
        msg_recu = int(msg_recu.decode('utf8'))
        print("Reçu {}".format(msg_recu))
        print("Reçu :", msg_recu)
        # client.send(b"5 / 5")
        if msg_recu != "fin":
            #
            # clear(ax)
            #
            a = array([[x[0][0]], [x[1][0]]])
            b = array([[x[0][0] + cos(x[2][0])],
                        [x[1][0] + sin(x[2][0])]])

            if msg_recu == 2:
                commande = 2
            elif msg_recu == 1:
                commande = 1
            else:
                commande = 0
            print("Commande :", commande)

            #print(msg_recu.decode())
            #commande = int(msg_recu.decode())
            #print("Commande :", commande)
            #
            listex.append(a[0, 0]), listex.append(b[0, 0])
            listey.append(a[1, 0]), listey.append(b[1, 0])
            # plot([a[0,0],b[0,0]],[a[1,0],b[1,0]],'blue')  # afficher le segment courant
            plot(listex, listey, 'blue')  # afficher la trace complète au fur et à mesure (tous les segments)
            #
            u = control(x, a, b, commande)
            #u1 = angle de la barre
            #u2 = angle de la voile
            xdot, δs = f(x, u)
            x = x + dt * xdot
            draw_sailboat(x, δs, u[0, 0], ψ, awind)
            draw_arrow(75, 40, ψ, 5 * awind, 'red')

print("Fermeture des connexions")
for client in clients_connectes:
    client.close()

connexion_principale.close()