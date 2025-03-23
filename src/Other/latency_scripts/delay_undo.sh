DEV=ens4
IP1=10.0.1.14
IP2=192.168.122.14

iptables -D OUTPUT -t mangle -d "$IP1" -j MARK --set-mark 11
# iptables -D OUTPUT -t mangle -d "$IP2" -j MARK --set-mark 12
tc qdisc del dev $DEV root
