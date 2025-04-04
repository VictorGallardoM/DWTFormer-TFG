# 🧠 DWTFormer: Prevenció de Lesions amb Deep Learning en Imatges Mèdiques

Aquest projecte forma part del meu **Treball de Fi de Grau** en Enginyeria Informàtica. L'objectiu és desenvolupar un model híbrid d'aprenentatge profund anomenat **DWTFormer**, que combina la **Transformada Wavelet Discreta (DWT)** amb una arquitectura **Transformer**, per detectar patrons en imatges mèdiques (del conjunt de dades MedMNIST) que puguin estar relacionats amb lesions esportives.

---

## 📌 Descripció del Projecte

En l’àmbit de la medicina esportiva, la prevenció de lesions és essencial. Aquest projecte proposa un enfocament automatitzat amb intel·ligència artificial per millorar la detecció precoç de factors de risc a partir d’imatges biomèdiques, utilitzant:

- **DWT** per extreure característiques multiescala.
- **Transformer** per capturar relacions espacials complexes.
- **Dataset**: PathMNIST, part del conjunt MedMNIST, adaptat a un context de risc esportiu.

---

## 🧠 Model: DWTFormer

El model es compon de tres fases:

1. **Preprocessament amb DWT**  
   Aplicació de la transformada wavelet a la imatge d’entrada per obtenir subbandes de freqüència (LL, LH, HL, HH).

2. **Divisió en patches i embeddings**  
   Les subbandes es divideixen en blocs vectoritzats i es codifiquen com a embeddings.

3. **Transformer Encoder**  
   Processament dels embeddings amb capes `nn.TransformerEncoder` que modelen dependències espacials i semàntiques.

---

## 🧪 Resultats

- Entrenament i validació amb mètriques com **accuracy**, **matriu de confusió** i corbes d'aprenentatge.
- Comparació amb models base com una **CNN** o un **Vision Transformer (ViT)** pur.
- Avaluació final sobre el conjunt de test: precisió global i anàlisi d'errors.

---

## 📁 Estructura del Repositori

