# DEV=name of the interface
# IP1=destination IP1
# IP2=destination IP2
DEV=ens4
IP1=10.0.1.15
# IP2=192.168.122.22

tc qdisc del dev $DEV root || true
tc qdisc add dev $DEV handle 1: root htb

tc class add dev $DEV parent 1: classid 1:15 htb rate 100000Mbps quantum 1500
tc qdisc add dev $DEV parent 1:15 handle 11 netem delay 1000ms 1ms distribution normal
tc filter add dev $DEV parent 1:0 prio 1 protocol ip handle 11 fw flowid 1:15
iptables -A OUTPUT -t mangle -d "$IP1" -j MARK --set-mark 11

# Nota: class_id 16 e handle 12
# tc class add dev $DEV parent 1: classid 1:16 htb rate 100000Mbps quantum 1500
# tc qdisc add dev $DEV parent 1:16 handle 12 netem delay 50ms 1ms distribution normal
# tc filter add dev $DEV parent 1:0 prio 1 protocol ip handle 12 fw flowid 1:16
# iptables -A OUTPUT -t mangle -d "$IP2" -j MARK --set-mark 12
# Se vuoi applicare lo stesso ritardo a un altro IP:
#iptables -A OUTPUT -t mangle -d "$ALTRO_IP" -j MARK --set-mark 12

# Se vuoi definire un'altra regola, copia le righe sopra incrementando di 1
# class_id e handle (17 e 13)


