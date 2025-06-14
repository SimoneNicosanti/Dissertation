terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.8.0"
    }
  }
}


provider "google" {
  project = "ai-at-edge-442215"  # Replace with your GCP project ID
  region  = "europe-west12-c"
}

# Create a VPC Network
resource "google_compute_network" "main_vpc" {
  name                    = "main-vpc"
  auto_create_subnetworks = false
}

# Create a Subnet
resource "google_compute_subnetwork" "main_subnet" {
  name          = "main-subnet"
  network       = google_compute_network.main_vpc.id
  ip_cidr_range = "10.0.1.0/24"
  region        = "europe-west12"
}

# Create a Firewall Rule to allow SSH
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.main_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"] # Allow SSH from anywhere (modify for security)
}

resource "google_compute_firewall" "allow_internal_traffic" {
  name    = "allow-internal-traffic"
  network = google_compute_network.main_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]  # Consente traffico TCP su tutte le porte
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]  # Consente traffico UDP su tutte le porte
  }

  allow {
    protocol = "icmp"  # Permette il traffico ICMP (ping)
  }

  source_ranges = ["10.0.1.0/24"]  # Destinazione nella stessa rete VPC
  direction          = "INGRESS"  # Traffico in uscita
}

resource "google_compute_firewall" "allow_internal_egress" {
  name    = "allow-internal-egress"
  network = google_compute_network.main_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]  # Consente traffico TCP su tutte le porte
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]  # Consente traffico UDP su tutte le porte
  }

  allow {
    protocol = "icmp"  # Permette il traffico ICMP (ping)
  }

  destination_ranges = ["10.0.1.0/24"]  # Destinazione nella stessa rete VPC
  direction          = "EGRESS"  # Traffico in uscita
}




# Define the IPs for each VM
locals {
  instance_ips = ["10.0.1.11", "10.0.1.12", "10.0.1.13", "10.0.1.14", "10.0.1.15", "10.0.1.16", "10.0.1.17"]  # Assign specific IPs
  names = ["registry", "optimizer", "model-manager", "deployer", "device", "server-1", "server-2"] ## Assign specific names
}



# Deploy Multiple Compute Engine Instances
resource "google_compute_instance" "vm_instances" {
  count        = 7
  name         = local.names[count.index]
  machine_type = "e2-standard-2"
  zone         = "europe-west12-c"

  boot_disk {
    initialize_params {
      image = "project-image"
    }
  }

  network_interface {
    network    = google_compute_network.main_vpc.id
    subnetwork = google_compute_subnetwork.main_subnet.id
    network_ip = local.instance_ips[count.index]  # Assign predefined static IP
    access_config {} # Assign public IP
  }

  metadata = {
    # Set your SSH key for the default user (google)
    "ssh-keys"              = "google:${file("/home/customuser/.ssh/id_rsa.pub")}"
  }
}
