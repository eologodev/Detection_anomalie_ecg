# ECG Anomaly Detection — Autoencoder (TensorFlow/Keras)


---

## Description

Détection automatique d'anomalies dans des signaux ECG (électrocardiogrammes) à l'aide d'un **autoencodeur** entraîné sur des signaux normaux.

L'idée : un autoencodeur entraîné uniquement sur des signaux sains apprend à les reconstruire fidèlement. Face à un signal anormal, l'erreur de reconstruction (MSE) est significativement plus élevée — ce qui permet de détecter l'anomalie.

---

## Pipeline

```
Dataset ECG (.mat)
      
      
Prétraitement
   Filtre Butterworth (passe-bas, suppression du bruit)
   Convolution + normalisation
      
      
Autoencodeur (Keras/TensorFlow)
   Encodeur  : couches Dense avec activation ReLU
   Décodeur  : reconstruction du signal d'entrée
      
      
Calcul MSE (erreur de reconstruction)
      
      
Seuillage au 95e percentile
      
      
Classification : Normal / Anomalie
```

---

## Technologies

 Outil | Rôle 

 Python | Langage principal |
 TensorFlow / Keras | Construction et entraînement de l'autoencodeur |
 NumPy / SciPy | Prétraitement du signal (filtre Butterworth, convolution) |
 Matplotlib | Visualisation des signaux et des erreurs de reconstruction |
 Jupyter Notebook | Environnement de développement |

---

## Dataset

- Format `.mat` (MATLAB)
- Signaux ECG réels - battements normaux et pathologiques
- Chargé via `scipy.io.loadmat`

---

## Résultats

- Détection basée sur le **95e percentile** de la distribution MSE sur les données d'entraînement
- Les signaux dont la MSE dépasse ce seuil sont classifiés comme anomalies
- Visualisation des signaux normaux vs anomalies avec la courbe d'erreur

---



---

## Structure

```
.
 ECG (1).ipynb       — notebook principal (prétraitement + modèle + résultats)
 ecg_dataset.mat     — dataset ECG
 Rapport.pdf         — rapport complet du projet
```

---

## Concepts clés

- **Autoencodeur** : réseau de neurones qui apprend à compresser puis reconstruire ses entrées
- **MSE (Mean Squared Error)** : mesure l'écart entre le signal original et sa reconstruction
- **Filtre Butterworth** : filtre passe-bas pour supprimer le bruit haute fréquence du signal ECG
- **Seuillage au 95e percentile** : toute MSE supérieure au seuil = anomalie détectée
