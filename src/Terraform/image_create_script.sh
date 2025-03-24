gcloud compute images delete project-image -q

gcloud compute instances create "image-create-instance" \
    --machine-type e2-standard-2 \
    --image-project debian-cloud \
    --image-family debian-11 \
    --zone europe-west12-c \
    --metadata="ssh-keys=google:$(cat ~/.ssh/id_rsa.pub)"

rm ~/.ssh/known_hosts*

gcloud compute ssh google@image-create-instance --zone=europe-west12-c --command="mkdir ~/models"
gcloud compute scp --recurse ../Other/models google@image-create-instance:~/models/

gcloud compute scp ./requirements_script.sh google@image-create-instance:~/requirements_script.sh

gcloud compute ssh google@image-create-instance --zone=europe-west12-c --command="chmod +x ~/requirements_script.sh"

gcloud compute ssh google@image-create-instance --zone=europe-west12-c --command="~/requirements_script.sh"

gcloud compute instances stop image-create-instance

gcloud compute images create project-image \
    --source-disk "image-create-instance" \
    --source-disk-zone europe-west12-c \
    --family project-family

gcloud compute instances delete image-create-instance --zone=europe-west12-c -q