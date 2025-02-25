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
  region  = "us-central1"
  zone    = "us-central1-a"
}

resource "google_compute_instance" "vm_instance" {
  name         = "quantization"
  machine_type = "c4-standard-4"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-pro-cloud/ubuntu-pro-2404-lts-amd64"
    }
  }

  network_interface {
    network = "default"
    access_config {}  # Allocates an ephemeral public IP
  }

  metadata = {
    # Set your SSH key for the default user (google)
    "ssh-keys"              = "google:${file("/home/customuser/.ssh/id_rsa.pub")}"
  }

  provisioner "remote-exec" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install python3-pip -y"
    ]
    connection {
      type        = "ssh"
      user        = "google"                                          # Default user for GCP instances
      private_key = file("/home/customuser/.ssh/id_rsa")              # Path to your private SSH key
      host        = self.network_interface[0].access_config[0].nat_ip   # Public IP of the instance
    }
  }
}
