# 🧠 DWTFormer-TFG

**Desenvolupament d’un model d’aprenentatge profund basat en DWT i Transformers per a la prevenció de lesions esportives**

Autor: Victor Gallardo Martínez  
Tutor: Xavier Font Aragonés  
Grau en Enginyeria Informàtica – Tecnocampus

---

## 📚 Descripció del projecte

Aquest projecte implementa un model anomenat **DWTFormer**, que combina la **Transformada Wavelet Discreta (DWT)** amb una arquitectura **Transformer** per a la classificació d’imatges mèdiques. L’objectiu principal és validar la viabilitat d’aquest enfocament en la detecció de patrons associats a **lesions esportives**, utilitzant imatges del dataset **PathMNIST** com a cas base.

---

## 🗂️ Estructura del projecte
DWTFormer-TFG/
├── notebooks/ 
├── src/ 
│ ├── data/ # Càrrega i preprocés de dades
│ ├── model/ # Arquitectura del model DWTFormer
│ └── train/ # Entrenament i avaluació
├── model/ # Models entrenats (.pt)
├── annexos/metrics/ # Gràfics i resultats per a la memòria
├── tests/ # Tests automatitzats amb pytest
├── main.py # Script principal d'entrenament i test
├── requirements.txt # Dependències del projecte
└── README.md # Aquest fitxer

## 🚀 Com executar el projecte

1. **Instal·la les dependències**
   pip install -r requirements.txt

2. **Entrena el model**
   python main.py
3. **Executa els tests**
   pytest tests/

4. **Revisa les mètriques**
   annexos/metrics/confusion_matrix.png
   /metrics/roc_curve_multiclass.png