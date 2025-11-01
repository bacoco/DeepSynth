# Déploiement sur Mac Studio

## Changement effectué

✅ **Batch size réduit de 5000 → 100 échantillons**

**Avant:**
- Upload toutes les 5000 échantillons (6-9 heures d'attente)
- Aucune visibilité sur HuggingFace pendant des heures

**Après:**
- Upload tous les 100 échantillons (5-10 minutes)
- Progression visible en temps réel sur HuggingFace

## Déploiement via Git

### 1. Pousser le commit

```bash
# Sur votre machine locale
git push origin main
```

### 2. Sur le Mac Studio

```bash
# Arrêter le processus en cours
pkill -f generate_qa_dataset

# Aller dans le répertoire du projet
cd /path/to/DeepSynth

# Récupérer les changements
git pull origin main

# Vérifier que le changement est bien là
grep "default=100" generate_qa_dataset.py
# Devrait afficher: default=100,

# Nettoyer les anciens fichiers (optionnel)
rm -rf work/ qa_samples/ *.log

# Relancer la génération
export PYTHONPATH=./src
python3 generate_qa_dataset.py 2>&1 | tee production_qa.log
```

### 3. Monitoring

**Vérifier l'état local:**
```bash
# Voir les derniers logs
tail -f production_qa.log

# Voir la progression locale
ls -lh work/samples/*.pkl | wc -l
du -sh work/samples/

# Vérifier le processus
ps aux | grep generate_qa_dataset | grep -v grep
```

**Vérifier HuggingFace:**
```bash
# Utiliser le script de monitoring
export PYTHONPATH=./src
python3 check_hf_dataset.py
```

## Timeline attendue

Avec le nouveau batch size de 100:

- **T+0:** Démarrage de la génération MS MARCO
- **T+5-10 min:** Premier upload de 100 échantillons visible sur HuggingFace
- **T+10-20 min:** Deuxième batch uploadé
- **Tous les 5-10 min:** Nouveaux uploads réguliers

Total estimé pour MS MARCO (~140k échantillons):
- ~140k × 3-5 sec = 7-10 heures
- Uploads: 1400 batches de 100
- Visibilité constante sur HuggingFace

## Options avancées

### Ajuster le batch size

Si vous voulez encore plus de fréquence:
```bash
# Upload tous les 50 échantillons (toutes les 2-5 minutes)
python3 generate_qa_dataset.py --batch-size 50

# Ou retourner à l'ancien comportement
python3 generate_qa_dataset.py --batch-size 5000
```

### Test rapide avant production

```bash
# Tester avec 500 échantillons pour vérifier
python3 generate_qa_dataset.py --test --max-samples 500 2>&1 | tee test_500.log

# Devrait montrer 5 uploads (500 ÷ 100 = 5)
```

## Troubleshooting

**Si rien n'apparaît sur HuggingFace:**

1. Vérifier les credentials:
```bash
export PYTHONPATH=./src
python3 -c "from deepsynth.config import Config; c = Config.from_env(); print(f'User: {c.hf_username}, Token: {c.hf_token[:10]}...')"
```

2. Vérifier les logs:
```bash
tail -100 production_qa.log | grep -i error
```

3. Vérifier le réseau:
```bash
curl -I https://huggingface.co/
```

**Si le processus semble bloqué:**

Les conversions Natural Questions prennent 2-5 minutes pour initialiser (287 shards).
C'est normal de voir "⏳ Initializing dataset connection" pendant quelques minutes.

## Scripts utiles

**check_hf_dataset.py** - Vérifier l'état sur HuggingFace
```bash
export PYTHONPATH=./src
python3 check_hf_dataset.py
```

**monitor_remote.sh** - Monitoring complet local + HF
```bash
./monitor_remote.sh
```

## Commit

```
commit 7c9e041cf762cc61fe461246dd97e47d0e03ab40
fix: reduce batch size from 5000 to 100 for faster HuggingFace upload visibility
```
