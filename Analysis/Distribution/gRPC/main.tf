terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.8.0"
    }
  }
}

provider "google" {
  project = "ai-at-edge-442215"
  region  = "europe-southwest1"
  zone    = "europe-southwest1-a"
}

resource "google_compute_instance" "vm_instance" {
  name         = "registry"
  machine_type = "e2-micro"
  zone         = "europe-southwest1-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"
    access_config {
      ## Ephemeral IP Addr
    }
  }

  metadata = {
    ssh-keys = "google:${file("/home/customuser/.ssh/id_rsa.pub")}"
  }

  provisioner "remote-exec" {
    inline = [
      "mkdir /home/google/registry",
      "sudo apt-get install python3-pip -y"
    ]

    connection {
      type        = "ssh"
      user        = "google"                                          # Default user for GCP instances
      private_key = file("/home/customuser/.ssh/id_rsa")              # Path to your private SSH key
      host        = self.network_interface[0].access_config[0].nat_ip # Public IP of the instance
    }
  }

  provisioner "file" {
    source      = "./registry/"
    destination = "/home/google/registry"

    connection {
      type        = "ssh"
      user        = "google"                                          # Default user for GCP instances
      private_key = file("/home/customuser/.ssh/id_rsa")              # Path to your private SSH key
      host        = self.network_interface[0].access_config[0].nat_ip # Public IP of the instance
    }
  }

  provisioner "remote-exec" {
    inline = [
      "cd /home/google/registry",
    ]

    connection {
      type        = "ssh"
      user        = "google"                                          # Default user for GCP instances
      private_key = file("/home/customuser/.ssh/id_rsa")              # Path to your private SSH key
      host        = self.network_interface[0].access_config[0].nat_ip # Public IP of the instance
    }
  }
}