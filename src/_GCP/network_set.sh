# gcloud compute ssh device --command="sudo python3 delay_script.py --dev ens3 --bandwidth 0.731 --latencies 5 55 --ips 10.0.1.16 10.0.1.17"
# gcloud compute ssh edge --command="sudo python3 delay_script.py --dev ens3 --bandwidth 2.36 --latencies 5 50 --ips 10.0.1.15 10.0.1.17"
# gcloud compute ssh cloud --command="sudo python3 delay_script.py --dev ens4 --bandwidth 100 --latencies 55 50 --ips 10.0.1.15 10.0.1.16"

gcloud compute ssh device --command="sudo python3 delay_script.py --dev ens3 --bandwidth 5.0 --latencies 5 55 --ips 10.0.1.16 10.0.1.17"
gcloud compute ssh edge --command="sudo python3 delay_script.py --dev ens3 --bandwidth 20.0 --latencies 5 50 --ips 10.0.1.15 10.0.1.17"
gcloud compute ssh cloud --command="sudo python3 delay_script.py --dev ens5 --bandwidth 100.0 --latencies 55 50 --ips 10.0.1.15 10.0.1.16"
