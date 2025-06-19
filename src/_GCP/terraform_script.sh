if [ "$1" -eq 0 ]; then
  terraform -chdir=./Terraform/ apply
else
  terraform -chdir=./Terraform/ destroy
fi
