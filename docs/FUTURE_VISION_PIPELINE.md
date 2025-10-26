# Vision-First Extensions for DeepSynth

Ce document récapitule des axes d'évolution imaginés autour du pipeline DeepSynth basé sur la conversion texte→image et le fine-tuning d'un décodeur OCR. L'objectif est d'explorer des usages où la conservation du flux visuel apporte un avantage fonctionnel, que ce soit pour réduire les coûts d'inférence, améliorer la robustesse ou ouvrir de nouveaux produits.

## RAG visuel avec bases vectorielles compactes
- **Principe** : au lieu d'archiver les textes sources, rasteriser les documents, encoder les images via l'encodeur vision figé et sauvegarder uniquement les embeddings de patchs ou de tokens visuels. Pour la récupération, on recharge les vecteurs, puis on réutilise le décodeur pour reconstruire un texte ou une réponse.
- **Bénéfices attendus** : taille de base réduite (embeddings float16 compressés), homogénéité avec la pile OCR existante, capacité à traiter des éléments où la mise en page est riche (tables, formulaires, signatures) sans repasser par une couche de prétraitement textuel.
- **Points de vigilance** : nécessité d'un index vectoriel adapté (FAISS/HNSW), définition d'un schéma de chunking visuel, calibration des métriques de similarité lorsque les documents sont bruités.

## Génération d'archives conformes
- Produire des résumés légaux ou des rapports de conformité directement à partir d'images de contrats scannés, en appliquant des règles de masquage automatique.

## Base de connaissances multimodale
- Agréger des brochures, fiches techniques ou slides en images, associer les embeddings visuels à des métadonnées (langue, produit) et générer des réponses marketing ou support sans conversion préalable en texte brut.

## Workflows formulaires → API
- Coupler la sortie textuelle du décodeur avec un parseur JSON pour alimenter des systèmes d'information depuis des formulaires papier ou manuscrits, en conservant les captures originales pour audit.

## Indexation événementielle
- Pour des corpus presse scannés, générer des timelines (dates, acteurs) en sortie, tout en conservant les embeddings pour des requêtes ultérieures sur le contexte visuel.

## Synthèse multimodale pour accessibilité
- Transformer des affiches ou supports pédagogiques en descriptions narratives adaptées aux lecteurs d'écran, avec un mode d'inférence batché exploitant les embeddings visuels partagés.

## Étapes ultérieures suggérées
1. **Prototyper l'extraction d'embeddings** en instrumentant l'encodeur vision pour exposer les sorties intermédiaires (patch embeddings, CLS) et évaluer la compression.
2. **Valider la pertinence retrieval** sur un corpus interne (documents longs, formulaires) en comparant rappel/precision entre la recherche texte et la recherche sur embeddings visuels.
3. **Adapter le décodeur** pour supporter la génération conditionnelle à partir d'un lot d'embeddings (concaténation contextuelle) et mesurer la latence.
4. **Industrialiser le stockage** (FAISS, Milvus, Weaviate) avec chiffrement et politiques de rétention pour respecter les contraintes compliance.
5. **Documenter les coûts** (CPU pour rasterisation, GPU pour indexation, stockage) afin de déterminer les cas où la solution visuelle devient plus compétitive qu'une pipeline NLP classique.

Ce plan vise à capitaliser sur l'architecture existante tout en ouvrant la voie à des cas d'usage où la représentation visuelle structurée est un avantage différenciant.
