
# Update and install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key and repo
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER

##############

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

##############

# Test Docker
docker run --rm hello-world

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

#####################

git clone https://github.com/vgoklani/semianalysis_hackathon_2026.git

srun --gpus=1 --pty  bash

tmux source-file /mnt/sharefs/user62/semianalysis_hackathon_2026/.tmux.conf

srun --gpus=1 --container-image=nvcr.io/nvidia/pytorch:26.02-py3 --pty bash

#####################
# claude

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc

# Install Node.js
nvm install 22

# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc

# Run it
claude
