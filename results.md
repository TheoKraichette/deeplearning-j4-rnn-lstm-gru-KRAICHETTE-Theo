# Comparaison LSTM vs GRU — Airline Passengers

Même dataset et même configuration (64 units, Dropout 0.2, `batch_size=1`, EarlyStopping sur `val_loss` patience=10, 100 epochs max, seed 42). Seule l'architecture change.

| Métrique | LSTM | GRU |
|----------|------|-----|
| Paramètres | 16 961 | 12 929 |
| Epochs (EarlyStopping) | 41 | 98 |
| Temps total (s) | 18.1 | 53.0 |
| Temps / epoch (s) | 0.44 | 0.54 |
| Val loss finale | 0.0061 | 0.0031 |
| RMSE test (passagers) | 35.1 | 33.4 |

## Lecture

- **GRU plus léger** : 12 929 paramètres contre 16 961 (~24 % de moins), car il a 2 portes au lieu de 3.
- **RMSE du même ordre** : 33.4 (GRU) vs 35.1 (LSTM) — écart négligeable (< 2 passagers), attendu sur un dataset aussi petit.
- **Epochs** : le GRU a amélioré sa `val_loss` plus longtemps (98 vs 41 epochs), d'où un temps total plus élevé — pas parce qu'il est plus lent, mais parce qu'il s'est entraîné plus longtemps avant l'arrêt.
- **Temps / epoch** : quasi identique ; en `batch_size=1` sur CPU, le temps est dominé par l'overhead, pas par le nombre de paramètres.
