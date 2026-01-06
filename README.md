# RedesAgente

Proyecto de entrenamiento y validación de un agente basado en Deep Q-Learning (DQN) para localizar objetos en imágenes usando una combinación de un VAE perceptual y un VAE discriminativo para generar mapas de calor espaciales y semánticos.

## Descripción

Este proyecto implementa un agente que aprende a localizar clases específicas en imágenes. Utiliza:

- Un VAE perceptual para obtener características visuales.
- Un VAE discriminativo para generar mapas de calor semánticos.
- Una red DQN para entrenar un agente que navega la imagen buscando regiones relevantes.
- Un prior espacial que mejora la precisión con el tiempo.

## Requisitos

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow

## Estructura

- `train()` — Función para entrenar el agente.
- `validate()` — Función para validar el agente en imágenes de validación.
- `LocateEnv` — Entorno que simula la navegación del agente sobre la imagen.
- Modelos:
  - `FeatureVAE` — VAE para extracción de características perceptuales.
  - `DiscriminativeVAE` — VAE para generación de mapas semánticos.
  - `DQN` — Red neuronal para la política de acciones.

## Cómo usar

1. Coloca las imágenes de entrenamiento en la carpeta `trains`, organizadas en subcarpetas por clase.
2. Coloca las imágenes de validación en la carpeta `validations1`.
3. Ejecuta el script principal para entrenar y validar:

```bash
python main.py
