# Prod.AI Análisis y Optimización Musical con IA

<p align="center">
  <img src="https://img.shields.io/badge/Status-Borrador-yellowgreen" alt="Status: Borrador/WIP">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="Licencia: MIT">
  <img src="https://img.shields.io/badge/Tecnología-Deep_Learning-red" alt="Tecnología: Deep Learning">
</p>

## Motivación: El Éxito en la Producción Musical Moderna

En el panorama competitivo de la música actual, la **aceptación en plataformas digitales** y la **cohesión en sets de DJ** dependen de que un *track* se alinee perfectamente con las expectativas sónicas de su género. Los productores a menudo trabajan a ciegas, adivinando si su mezcla o *mastering* encaja.

**Prod.AI** es la herramienta diseñada para terminar con las conjeturas. Es un sistema de análisis musical impulsado por IA que proporciona información objetiva y cuantificable sobre las características energéticas y estructurales de una canción.

**💡 Enfocado en:**
* Asegurar que las canciones suenen **acordes al género** para mejorar la aceptación en plataformas.
* Optimizar el **mastering** para lograr *loudness* y dinámicas que encajen en el *set* de un DJ.
* Proveer una fuente de **inspiración técnica** al analizar la composición de otros *tracks*.

---

##  Características Principales

### 1. Detección de Género Avanzada 
Utilizamos una arquitectura de red neuronal recurrente sofisticada para analizar el audio, incluyendo modelos **LSTM (Long Short-Term Memory)** y **GRU (Gated Recurrent Unit)**. 

* **Detección:** Predice el género musical más probable con alta precisión basándose en *features* espectrales y temporales.

### 2. Análisis Métrico Cuantificable 
Extraemos métricas acústicas clave de la canción (como energía, RMS, y densidad espectral) y las comparamos con los **patrones ideales** del género predicho.

* **Feedback Directo:** El sistema indica si el *track* está **por encima o por debajo** del perfil energético estándar del género, facilitando ajustes precisos de *mixing* o *mastering*.

### 3. Separación de Fuentes (Aislamiento Vocal) 
Una funcionalidad esencial para el análisis detallado:
* **Aislamiento:** Permite **separar las pistas vocales de la instrumental** (música de fondo).
* **Utilidad:** Ideal para analizar la complejidad rítmica de la instrumental, estudiar la producción vocal o aislar pistas para remezclas creativas.

### 4. Visualización de Energía 
Generación de una **tabla de energía** detallada, mostrando cómo se distribuye la potencia de la señal a lo largo del tiempo. Crucial para analizar las dinámicas y la percepción de *loudness*.

---

##  Tecnologías Utilizadas

| Componente | Herramientas Clave | Propósito |
| :--- | :--- | :--- |
| **Deep Learning** | Python, TensorFlow / PyTorch, **LSTM, GRU** | Modelos de detección y análisis de género. |
| **Procesamiento de Audio** | Librosa, Essentia | Extracción de *features* y análisis espectral. |
| **Separación de Fuentes** | Spleeter (o similar) | Aislamiento de pistas (vocales/instrumental). |



---
