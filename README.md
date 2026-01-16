<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=8A2BE2&height=220&section=header&text=IngeniosasTech&fontSize=70&fontAlign=50&fontAlignY=35&desc=Hack4Her%202025%20%7C%20Reto%204&descAlign=50&descAlignY=65&descSize=20" width="100%"/>

  <br/>

  <a href="#">
    <img src="https://img.shields.io/badge/🏆_Award-3rd_Place_Winner-FFD700?style=for-the-badge&labelColor=black"/>
  </a>
  
  <br/> <br/>
  **🥉 Solución de Machine Learning ganadora del 3er Lugar en el Reto 4 - Hack4Her 2025**

  <br/>
  
  <a href="https://github.com/Natty093">
    <img src="https://img.shields.io/badge/Team-IngeniosasTech-ff69b4?style=for-the-badge&logo=github"/>
  </a>
  <img src="https://img.shields.io/badge/AI-Random_Forest-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Prototipo_Funcional-success?style=for-the-badge" />

</div>

---

## 👋 ¡Bienvenidas!
Somos **IngeniosasTech**, equipo ganador del tercer lugar en el **Hack4Her 2025**. 
Este repositorio contiene el código fuente de nuestro modelo predictivo desarrollado para el **Reto 4**: *"Live Loyalty de arca continental"*.

---

## 👩‍💻 Nuestro Equipo
| Integrante | Rol | GitHub |
| :--- | :--- | :--- |
| **Natalie** | AI Developer / Backend / Lider | [@Natty093](https://github.com/Natty093) |
| **Rebeca** | AI Developer / Pitch 
| **Lizeth** | Documentación / Frontend / Pitch
| **Coral** | Documentación / Frontend

---

## 🧠 La Solución: Inteligencia Artificial
Nuestra propuesta aborda la problemática mediante un modelo de **Machine Learning** capaz de predecir Opotunidades para ventas basándose en datos históricos.

### ¿Cómo funciona?
El sistema procesa los datos de entrada a través de un flujo inteligente:
1.  **Preprocesamiento:** Usamos `encoders.pkl` para transformar datos categóricos (texto) en numéricos comprensibles para la IA.
2.  **Predicción:** El corazón del proyecto es `modelo_rf.pkl`, un algoritmo de **Random Forest** entrenado para detectar patrones complejos.
3.  **Resultado:** El script `main.py` genera una predicción precisa para apoyar la toma de decisiones.

---

## 📂 Estructura del Proyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.py` | 🚀 **Punto de entrada:** Script principal que carga el modelo y ejecuta la predicción. |
| `modelo_rf.pkl` | 🧠 **El Cerebro:** Modelo de Random Forest ya entrenado y serializado. |
| `encoders.pkl` | 🔄 **Traductor:** Diccionarios para convertir variables de texto a números. |
| `columnas_modelo.pkl` | 📋 **Esquema:** Lista de características (features) que el modelo espera recibir. |
| `requirements.txt` | 📦 **Dependencias:** Librerías necesarias para correr el entorno. |

---

## 🛠️ Tecnologías Utilizadas

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
</div>

---

## 🚀 Instalación y Uso Local

Para probar nuestro modelo ganador en tu computadora:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/Natty093/](https://github.com/Natty093/)[NOMBRE-DEL-REPO].git
    cd [NOMBRE-DEL-REPO]
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar el modelo:**
    ```bash
    python main.py
    ```

---

<div align="center">
  <br/>
  Hecho con 💜, datos y código por <b>IngeniosasTech</b>.
  <br/>
  <i>¡Tercer Lugar Hack4Her 2025!</i> 🥉
</div>
