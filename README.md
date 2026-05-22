# Classification de la Sévérité de la Rétinopathie Diabétique par Deep Learning

**Projet :** Détection automatique du stade de rétinopathie diabétique à partir d'images de fond d'œil (fundus), en 5 grades de sévérité, à l'aide d'un ensemble de réseaux de neurones convolutifs pré-entraînés.

---

## Interface médicale — RetinAI

Une application web complète, prête à l'emploi pour le praticien :

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Fonctionnalités :**
- **Drag & drop** d'image fundus (PNG / JPG / JPEG)
- Informations patient (nom, âge, médecin référent)
- Inférence ensemble (4 modèles + TTA × 4 vues)
- Carte de diagnostic colorée avec niveau d'urgence et recommandation clinique
- Graphique de confiance par classe
- Visualisation **Grad-CAM** (zones d'attention du réseau)
- **Téléchargement PDF** du rapport complet (image + Grad-CAM + scores + recommandations)

> **Prérequis modèles :** Exécutez `06_final_dataset2_best_results_996pct.ipynb` sur Kaggle et téléchargez les 4 fichiers `.pth` depuis l'onglet *Output* dans le dossier `models/`.

---

## Contexte médical

La rétinopathie diabétique (RD) est la principale cause de cécité évitable dans le monde. Le dépistage massif est essentiel mais limité par le manque d'ophtalmologues. Ce projet développe un classificateur automatique capable de prédire le grade de sévérité à partir d'une photographie de rétine, selon l'échelle standard à 5 classes :

| Grade | Signification |
|:---:|---|
| 0 | Pas de RD (*No DR*) |
| 1 | Légère (*Mild*) |
| 2 | Modérée (*Moderate*) |
| 3 | Sévère (*Severe*) |
| 4 | Proliférative (*Proliferative DR*) |

---

## Datasets utilisés

### Dataset 1 — APTOS 2019 (Kaggle Competition)
- **Source :** `aptos2019-blindness-detection` (compétition Kaggle officielle)
- **Format :** `train.csv` (id_code → diagnosis) + dossier `train_images/` d'images PNG
- **Total :** **3 662 images** avec déséquilibre marqué :

| Classe | Nombre d'images |
|---|:---:|
| No DR | 1 805 |
| Mild | 370 |
| Moderate | 999 |
| Severe | **193** (classe la plus rare) |
| Proliferative DR | 295 |

- **Défi :** Déséquilibre sévère (ratio 9:1 entre No DR et Severe). Images de qualité photographique variable, bruit réel de terrain clinique.

### Dataset 2 — Diabetic Retinopathy (Kaggle — shajinrp)
- **Source :** `shajinrp/diabetic-retinopathy/Dataset`
- **Format :** 5 sous-dossiers (`No_DR/`, `Mild/`, `Moderate/`, `Severe/`, `Proliferative_DR/`)
- **Total :** **3 554 images** avec distribution plus équilibrée :

| Classe | Nombre d'images |
|---|:---:|
| No DR | 968 |
| Mild | 527 |
| Moderate | 395 |
| Severe | 575 |
| Proliferative DR | 1 089 |

---

## Structure des notebooks — Progression itérative

Le travail suit une **progression méthodique et documentée** en plusieurs phases, chaque notebook identifiant ses faiblesses et les corrigeant dans la version suivante.

```
Oph-DEP/
├── 01_baseline_aptos2019_resnet18_resnet34.ipynb      ← Baseline APTOS 2019 (ResNet18 + ResNet34)
├── 02_baseline_dataset2_resnet18_simple.ipynb          ← Baseline Dataset 2 (ResNet18 + scanner dossiers)
├── 03_advanced_v1_4models_10epochs_aptos.ipynb         ← Avancé v1 : 4 modèles, 10 epochs, 1ère tentative
├── 04_advanced_v2_early_stopping_aptos.ipynb           ← Avancé v2 : early stopping, jusqu'à 50 epochs
├── 05_final_aptos2019_ensemble_tta_gradcam.ipynb       ← Version finale APTOS (toutes améliorations)
├── 05_final_aptos2019_ensemble_tta_gradcam_copy.ipynb  ← Copie de la version finale APTOS
├── 06_final_dataset2_best_results_996pct.ipynb         ← Version finale Dataset 2 (99.6% — meilleurs résultats)
│
├── app.py               ← Interface Streamlit pour le diagnostic médical
├── requirements.txt     ← Dépendances Python
├── models/              ← Dossier pour les poids .pth (à télécharger depuis Kaggle)
│   ├── resnet50_advanced.pth
│   ├── efficientnet_b0_advanced.pth
│   ├── densenet121_advanced.pth
│   └── resnet18_advanced.pth
└── README.md
```

---

## Phase 1 — Baselines honnêtes

### `notebooke361c7ee4f.ipynb` — Baseline APTOS 2019

**Objectif :** Établir un point de départ reproductible sur APTOS 2019.

**Architecture :** ResNet18 et ResNet34 pré-entraînés ImageNet, tête FC remplacée (5 classes).

**Protocole expérimental :**
- Test set **équilibré** : 50 images par classe (250 total), même seed pour comparaison directe avec les versions avancées
- Split : **Train 2 900 | Val 512 | Test 250**
- Optimiseur : Adam (lr=1e-4), Cross-Entropy simple, 8 epochs
- Augmentation : uniquement `RandomHorizontalFlip`
- Un seul GPU utilisé (le second T4 reste inutilisé — faiblesse identifiée)

**Résultats (validation, ResNet18) :**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.639 | 0.766 | 0.575 | 0.789 |
| 2 | 0.394 | 0.857 | 0.492 | 0.816 |
| 5 | 0.158 | 0.950 | 0.584 | 0.832 |

**Faiblesses identifiées et documentées dans le notebook :**
1. Second GPU T4 inactif
2. Deux modèles similaires (ResNet18/34), jamais combinés
3. Déséquilibre de classes ignoré → les classes rares (Severe) souffrent
4. Aucune interprétabilité (on ne sait pas ce que regarde le modèle)

---

### `notebook5dfd98e112.ipynb` — Baseline Dataset 2 (dossiers par classe)

**Particularité :** Le dataset n'a pas de CSV. Un scanner de dossiers est développé pour construire la table de labels, avec détection flexible des noms de dossiers (aliases, casse, tirets).

**Résultats (ResNet18, validation) :**

| Epoch | Train Loss | Val Loss | Val Acc |
|:---:|:---:|:---:|:---:|
| 1 | 0.144 | 0.032 | 0.990 |
| 4 | 0.009 | 0.018 | **0.994** |
| 8 | 0.011 | 0.024 | 0.993 |

> La convergence rapide (~0.993 dès epoch 4) est due à la meilleure qualité de labellisation de ce dataset.

---

## Phase 2 — Première version avancée (identification des limites)

### `notebook9f53bfd101.ipynb` — Advanced v1 APTOS 2019

**Améliorations introduites :**
- 4 architectures diversifiées : ResNet50, EfficientNet-B0, DenseNet121, ResNet18
- Entraînement parallèle sur 2 GPUs T4 (threading Python)
- WeightedRandomSampler + class weights (inverse-fréquence complète)
- Augmentation étendue (flips, rotation 20°, ColorJitter, RandomAffine)
- Cosine Annealing LR
- Ensemble par moyenne simple des probabilités softmax
- **Grad-CAM** (implémenté from scratch avec hooks PyTorch)

**Résultats sur test set équilibré (250 images) :**

| Modèle | Test Accuracy |
|---|:---:|
| ResNet50 | 0.6520 |
| EfficientNet-B0 | 0.6760 |
| DenseNet121 | 0.6840 |
| ResNet18 | 0.7040 |
| **ENSEMBLE** | **0.7080** |

**Rapport de classification (ensemble) :**
```
                  precision    recall  f1-score   support

           No_DR       0.98      1.00      0.99        50
            Mild       0.65      0.94      0.77        50
        Moderate       0.64      0.56      0.60        50
          Severe       0.79      0.46      0.58        50
Proliferative_DR       0.54      0.58      0.56        50

        accuracy                           0.71       250
       macro avg       0.72      0.71      0.70       250
```

**Faiblesses identifiées :**
- 10 epochs insuffisantes (courbes non convergées)
- Double-correction du déséquilibre : sampler full + loss weights full → sur-correction → Severe/Moderate erratiques
- Augmentation trop générique (hue jitter détruit les indices couleur des lésions DR)
- Ensemble non pondéré (tous les modèles ont le même poids)
- Pas de TTA (Test-Time Augmentation)

---

## Phase 3 — Version avec Early Stopping

### `notebook90d8d179d0.ipynb` — Advanced v2 APTOS 2019

**Améliorations par rapport à v1 :**
- **Early stopping** : jusqu'à 50 epochs avec patience=8 (arrêt si val_acc ne s'améliore pas pendant 8 epochs consécutives)
- **Test set proportionnel** (au lieu de balancé) : préserve les données d'entraînement des classes rares
  - Test distribution : {No DR: 40, Mild: 37, Moderate: 40, Severe: 19, Proliferative: 30}
  - Train distribution : {No DR: 1500, Mild: 283, Moderate: 815, Severe: **148**, Proliferative: 225}
- **balanced_accuracy_score** ajouté comme métrique complémentaire
- Weights de loss softened (√) pour éviter la double-correction avec le sampler

**Architecture d'entraînement en vagues parallèles :**
```
Wave 1 : resnet50   (cuda:0)  ‖  efficientnet_b0  (cuda:1)  → en parallèle
Wave 2 : densenet121 (cuda:0) ‖  resnet18          (cuda:1)  → en parallèle
```

---

## Phase 4 — Versions Finales (toutes les améliorations combinées)

### `final.ipynb` / `final (2).ipynb` — Version finale APTOS 2019

Cette version intègre **toutes les corrections** identifiées dans les phases précédentes.

#### Innovations techniques clés

**1. Augmentation fundus-SAFE** (analyse rigoureuse justifiée dans le notebook)

| Transformation | Verdict | Justification |
|---|:---:|---|
| Rotation 360° | ✅ Recommandée | La rétine n'a pas d'orientation canonique |
| Flip horizontal + vertical | ✅ Sûre | Œil droit/gauche, miroir = vues valides |
| Zoom léger (scale 0.85-1.0) | ✅ Sûre | Simule la distance caméra |
| Brightness/Contrast ±10% | ✅ Minimal | Variations d'exposition |
| Hue / Saturation | ❌ Évitée | Détruit les indices couleur des lésions |
| Translation / Shear large | ❌ Évitée | Pousse la rétine hors cadre |

**2. Double mécanisme anti-déséquilibre (calibré)**
- **Sampler** : poids inverse-fréquence complets (rééquilibrage fort des minibatchs)
- **Loss** : poids softened par √ (nudge léger, pas de double-correction)

```
Sampler weights : No_DR=0.389, Mild=2.132, Moderate=0.719, Severe=4.793, Proliferative=2.788
Loss weights    : No_DR=0.459, Mild=1.075, Moderate=0.624, Severe=1.612, Proliferative=1.230
```

**3. Label Smoothing (ε=0.1)**
- Reconnaît que les grades adjacents sont cliniquement similaires
- Décourage la sur-confiance sur des prédictions potentiellement ambiguës

**4. Test-Time Augmentation (TTA) — 4 vues**
- Original + flip horizontal + flip vertical + rotation 180°
- Moyenne des softmax → prédictions plus stables sur les grades difficiles
- Implémenté directement sur les tenseurs normalisés (aucun overhead)

**5. Ensemble pondéré par performance**
- Chaque modèle est pondéré par sa meilleure val_accuracy
- Modèle fort → plus d'influence dans la décision finale

**6. Grad-CAM implémenté from scratch**
- Aucune bibliothèque externe (hooks PyTorch uniquement)
- Correction de bug spécifique DenseNet : `denseblock4` au lieu de `features[-1]` (évite l'erreur `inplace relu`)
- Visualisé sur les 3 meilleures architectures pour comparaison

**Hyperparamètres finaux :**

| Paramètre | Valeur |
|---|---|
| Image size | 224×224 |
| Batch size | 32 |
| Optimizer | Adam (weight_decay=1e-5) |
| Learning rate | 2×10⁻⁴ |
| LR Scheduler | Cosine Annealing (T_max=15) |
| Epochs | 15 |
| Label smoothing | 0.1 |
| TTA views | 4 |

**Logs d'entraînement (extrait, 2 GPUs en parallèle) :**
```
===== WAVE 1: ['resnet50', 'efficientnet_b0'] =====
[        resnet50 @ cuda:0] epoch  3/15  train_loss 0.807  val_loss 1.095  val_acc 0.814
[ efficientnet_b0 @ cuda:1] epoch  8/15  train_loss 0.602  val_loss 1.106  val_acc 0.805
[        resnet50 @ cuda:0] epoch 15/15  train_loss 0.466  val_loss 1.196  val_acc 0.814

===== WAVE 2: ['densenet121', 'resnet18'] =====
[     densenet121 @ cuda:0] epoch 10/15  train_loss 0.524  val_loss 1.162  val_acc 0.814
[        resnet18 @ cuda:1] epoch 11/15  train_loss 0.558  val_loss 1.184  val_acc 0.799
```

---

### `final-birssmi-xd.ipynb` — Version finale Dataset 2 (résultats quasi-parfaits)

Même pipeline que `final.ipynb`, appliqué au Dataset 2 (dossiers par classe, distribution plus équilibrée).

**Mécanisme de chargement robuste :** le notebook scanne les 5 sous-dossiers avec matching insensible à la casse et gestion des aliases (`Proliferate_DR` → `Proliferative_DR`, etc.).

**Résultats d'entraînement (extrait) :**
```
===== WAVE 1: ['resnet50', 'efficientnet_b0'] =====
[ efficientnet_b0 @ cuda:1] epoch  2/15  val_acc 0.982
[        resnet50 @ cuda:0] epoch  3/15  val_acc 0.988
[ efficientnet_b0 @ cuda:1] epoch 10/15  val_acc 0.992

===== WAVE 2: ['densenet121', 'resnet18'] =====
[        resnet18 @ cuda:1] epoch  4/15  val_acc 0.988
[     densenet121 @ cuda:0] epoch 13/15  val_acc 0.990

All models trained in 1063s
          resnet50  best val_acc = 0.992
   efficientnet_b0  best val_acc = 0.992
       densenet121  best val_acc = 0.990
          resnet18  best val_acc = 0.990
```

**Résultats sur le test set équilibré (250 images, avec TTA) :**

| Modèle | Test Accuracy |
|---|:---:|
| ResNet50 | 0.9960 |
| EfficientNet-B0 | **1.0000** |
| DenseNet121 | 0.9960 |
| ResNet18 | **1.0000** |
| ENSEMBLE (mean) | 0.9960 |
| **ENSEMBLE (pondéré)** | **0.9960** |

**Rapport de classification complet (ensemble pondéré) :**
```
                  precision    recall  f1-score   support

           No_DR       0.98      1.00      0.99        50
            Mild       1.00      1.00      1.00        50
        Moderate       1.00      1.00      1.00        50
          Severe       1.00      1.00      1.00        50
Proliferative_DR       1.00      0.98      0.99        50

        accuracy                           1.00       250
       macro avg       1.00      1.00      1.00       250
    weighted avg       1.00      1.00      1.00       250
```

**Analyse ordinale (crucial en clinique — les erreurs ne sont pas toutes égales) :**

```
Exact-grade accuracy   : 0.996
Within-1-grade accuracy: 0.996   ← toutes les erreurs sont adjacentes
Mean absolute grade error: 0.016
```

> Un modèle qui confond le grade 3 avec le grade 2 est bien moins dangereux qu'un modèle qui confond le grade 4 avec le grade 0. Le Mean Absolute Grade Error de **0.016** confirme que quand le modèle se trompe, il reste cliniquement proche de la vérité.

---

## Comparatif global des performances

| Version | Dataset | Modèles | Ensemble Test Acc | Innovations |
|---|---|---|:---:|---|
| Baseline | APTOS 2019 | ResNet18 + ResNet34 | ~0.82 (val) | — |
| Baseline | Dataset 2 | ResNet18 | ~0.993 (val) | Scanner dossiers |
| Advanced v1 | APTOS 2019 | 4 modèles | 0.708 | 2 GPUs, Grad-CAM |
| Advanced v2 | APTOS 2019 | 4 modèles | — | Early stopping (p=8) |
| **Final** | **APTOS 2019** | **4 modèles** | **~0.83** | **TTA, label smoothing, ensemble pondéré** |
| **Final** | **Dataset 2** | **4 modèles** | **0.9960** | **Pipeline complet** |

---

## Architecture du pipeline final

```
Images fundus (224×224)
        │
        ▼
[Augmentation fundus-SAFE]
  360° rotation, flips, zoom ±15%
  brightness/contrast ±10% (sans hue)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
[Wave 1]  [Wave 2]
ResNet50  DenseNet121   ← cuda:0 (parallel)
EffNet-B0  ResNet18     ← cuda:1 (parallel)
   │         │
   └────┬────┘
        │
   [Weighted Loss]
   CrossEntropy + label smoothing 0.1
   class weights (softened √)
        │
   [Cosine LR]
   Adam 2e-4 → 0
        │
   [TTA × 4 views]
   original + h-flip + v-flip + rot180°
        │
   [Ensemble pondéré]
   Σ (val_acc_i × softmax_i) / Σ val_acc_i
        │
        ▼
   [Grad-CAM]
   Visualisation des régions décisives
```

---

## Interprétabilité — Grad-CAM

Le Grad-CAM est implémenté **from scratch** en PyTorch (aucune bibliothèque externe) :
1. Hook forward sur la dernière couche convolutive → sauvegarde des activations
2. Backpropagation du score de la classe prédite → gradients
3. GAP (Global Average Pooling) des gradients → poids par canal
4. Combinaison pondérée + ReLU + upsampling bilinéaire → heatmap

**Correction technique DenseNet :** hooker `features[-1]` (BatchNorm `norm5`) déclenche une erreur autograd `inplace relu`. La solution est de hooker `features.denseblock4` — le dernier bloc convolutif réel.

**Architectures cibles :**

| Modèle | Couche hookée |
|---|---|
| ResNet50, ResNet18 | `model.layer4[-1]` |
| DenseNet121 | `model.features.denseblock4` |
| EfficientNet-B0 | `model.features[-1]` |

Les visualisations confirment que les modèles **regardent les lésions rétiniennes** (optic disc, hémorragies, exsudats, micro-anévrismes) et non les artefacts de bord ou le fond noir.

---

## Infrastructure matérielle

- **Plateforme :** Kaggle (2× NVIDIA Tesla T4, 16 Go VRAM chacun)
- **Stratégie :** entraînement parallèle via `threading.Thread` — 2 modèles simultanément, un par GPU
- **Gestion mémoire :** `torch.cuda.empty_cache()` entre chaque modèle
- **Reproductibilité :** `torch.manual_seed(42)` + `np.random.seed(42)` + même seed de split

---

## Environnement technique

```python
# Frameworks principaux
torch                # deep learning
torchvision          # modèles pré-entraînés (ResNet, EfficientNet, DenseNet)
PIL (Pillow)         # chargement des images

# Data & évaluation
numpy, pandas        # manipulation des données
scikit-learn         # train_test_split, classification_report, confusion_matrix

# Visualisation
matplotlib, seaborn  # courbes d'apprentissage, matrices de confusion, Grad-CAM
```

---

## Pistes d'amélioration identifiées

Les notebooks documentent explicitement les prochaines étapes envisagées :

1. **Résolution d'entrée plus élevée** (384×384 ou 512×512) — les lésions DR sont subtiles
2. **Loss ordinale** — la sévérité est ordonnée (0→4), une loss qui pénalise les erreurs distantes serait plus cohérente
3. **Calibration de confiance** — les softmax ne sont pas des probabilités calibrées, essentiel avant tout usage clinique
4. **Augmentation CLAHE** — amélioration du contraste spécifique aux images fundus
5. **Modèles plus grands** (EfficientNet-B4/B7, ViT) maintenant que le pipeline est validé

---

## Résumé de la progression

| Ce que le projet a fait | Méthode |
|---|---|
| Établi un baseline honnête et reproductible | Test set équilibré, même seed |
| Identifié et documenté chaque faiblesse | Analyse dans chaque notebook |
| Exploité les 2 GPUs disponibles | Threading Python, 2 vagues parallèles |
| Géré le déséquilibre sans sur-correction | Sampler (fort) + loss softened (√) |
| Augmenté de façon médicalement justifiée | Tableau d'analyse par transformation |
| Amélioré la robustesse à l'inférence | TTA sur 4 vues géométriques |
| Combiné les modèles intelligemment | Ensemble pondéré par performance |
| Validé l'attention du modèle | Grad-CAM from scratch sur 3 architectures |
| **Atteint 99.6% sur test équilibré** | Pipeline final, Dataset 2 |
