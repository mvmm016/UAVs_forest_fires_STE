# Base: Ubuntu 24.04 LTS
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="uav-fire-ste"
LABEL org.opencontainers.image.description="Projeto de STE de focos de incêndio usando UAVs"
LABEL org.opencontainers.image.authors="Carlos Craveiro <carlos.craveiro@usp.br>" 

# Configurações de ambiente para evitar interações e otimizar Python
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalação de dependências essenciais do sistema 
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget tar unzip curl ca-certificates git \
    python3 python3-venv python3-pip \
    tini \
 && rm -rf /var/lib/apt/lists/*

# Cria o ambiente virtual e prepara o diretório de trabalho
RUN python3 -m venv /venv
WORKDIR /root/work

# Copia o arquivo de requisitos e instala as dependências
# Fazemos isso antes de copiar o resto do código para aproveitar o cache do Docker
COPY requirements.txt .
RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt && \
    /venv/bin/pip install mutpy notebook==7.2.2 ipywidgets==8.1.5

# Gerenciador de processos tini 
ENTRYPOINT ["/usr/bin/tini", "--"]

# Inicialização do Jupyter Notebook 
CMD ["/venv/bin/python", "-m", "notebook", "--ip=0.0.0.0", "--no-browser", \
     "--allow-root", \
     "--NotebookApp.notebook_dir=/root/work"]
