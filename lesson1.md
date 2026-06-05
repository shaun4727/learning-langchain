Here is a comprehensive troubleshooting reference table summarizing the environment and Docker configuration hurdles encountered during Lesson 1 on your Ubuntu 24.04 system.

### Lesson 1: Environment & Docker Troubleshooting Reference

| Error Signature | Architectural "Why" (Root Cause) | Resolution Steps & Commands |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'distutils'` | Legacy Docker Compose v1 (a Python-based tool) was being invoked on Ubuntu 24.04, which utilizes **Python 3.12**. The `distutils` module was deprecated in Python 3.10 and completely removed in 3.12, causing a fatal crash. | **Switch to Compose V2 (Go-based plugin):**<br>

<br>1. Remove legacy package:<br>

<br>`sudo apt-get remove docker-compose`<br>

<br>2. Install native plugin:<br>

<br>`sudo apt-get install docker-compose-plugin`<br>

<br>3. Drop the hyphen in your syntax:<br>

<br>`docker compose up --build` |
| `unknown flag: --build` | The Ubuntu distro-maintained package (`docker.io`) was being used. Its command-line parsing layout is fragmented, causing the sub-command flags (`--build`) to fall through to the root `docker` binary context instead of forwarding them to the compose plugin module. | **Migrate to Official Upstream Docker CE Repository:**<br>

<br>1. Purge distro packages:<br>

<br>`sudo apt-get remove -y docker.io docker-doc docker-compose docker-compose-v2 containerd runc`<br>

<br>2. Setup official repository and keys, then run:<br>

<br>`sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin` |
| `WARN[0000] ... the attribute 'version' is obsolete` | Modern Docker Compose V2 utilizes a single, dynamically evolving specification matrix. Explicit schema definitions at the top of the file are redundant and ignored. | **Clean Code Maintenance:**<br>

<br>Open `docker-compose.yml` and delete the legacy line:<br>

<br>`version: '3.8'` |
| `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock` | The Docker daemon socket is root-privileged by default. Your active shell user boundary did not possess read/write access permissions to interact with the domain socket interface. | **Grant Non-Root Socket Privileges:**<br>

<br>1. Append user to docker group string matrix:<br>

<br>`sudo usermod -aG docker $USER`<br>

<br>2. Evaluate and reload group security context in active terminal:<br>

<br>`newgrp docker` |
| `Cannot connect to the Docker daemon... Is the docker daemon running?` | After installing the clean upstream engine, the background daemon execution binary (`dockerd`) and its socket interceptor unit were registered but left uninitialized or unstarted by systemd. | **Initialize and Boot Daemon via systemd:**<br>

<br>1. Force systemd manager configuration reload:<br>

<br>`sudo systemctl daemon-reload`<br>

<br>2. Start and enable both units immediately and persistently across reboots:<br>

<br>`sudo systemctl enable --now docker.socket`<br>

<br>`sudo systemctl enable --now docker` |

Keep this table handy in your local project workspace notes. Whenever you are ready to proceed past the environment setup, let me know, and we will move to **Lesson 2: LangChain Core & Structured Output**.