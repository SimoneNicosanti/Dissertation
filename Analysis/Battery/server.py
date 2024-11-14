import socket

# Parametri del server
HOST = '127.0.0.1'  # Indirizzo locale
PORT = 65432        # Porta di ascolto

# Creazione del socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Server in ascolto su", HOST, ":", PORT)
    
    # Accetta connessioni in arrivo
    while True :
        conn, addr = s.accept()
        with conn:
            print('Connesso a:', addr)
            # Ricezione dati dal client
            while True:
                data = conn.recv(1024)  # Riceve pacchetti di 1024 byte
                if not data:
                    break
            # print("Ricevuto:", data.decode('utf-8'))
