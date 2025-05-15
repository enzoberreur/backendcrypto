import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Callable
import streamlit as st
from backtester import Backtester
from strategies import (
    MovingAverageCrossover,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
    SupertrendStrategy
)

def multi_objective_optimization(
    backtester: Backtester,
    strategy_generator: Callable,
    param_combinations: List[Dict],
    metrics: Dict[str, float] = {'sharpe_ratio': 1.0},
    constraints: Dict[str, Tuple[float, float]] = {},
) -> Dict[str, Any]:
    """
    Fonction d'optimisation multi-objectif améliorée qui permet d'optimiser selon plusieurs métriques pondérées.
    
    Args:
        backtester: Instance du backtester
        strategy_generator: Fonction qui génère une stratégie à partir des paramètres
        param_combinations: Liste de dictionnaires, chacun contenant une combinaison de paramètres
        metrics: Dictionnaire des métriques à optimiser avec leurs poids (positifs pour maximiser, négatifs pour minimiser)
        constraints: Contraintes min/max sur certaines métriques (ex: {'win_rate': (0.4, None)} pour win_rate > 40%)
        
    Returns:
        Dict avec les meilleurs paramètres et les résultats
    """
    results = []
    best_score = -float('inf')
    best_params = None
    best_result = None
    
    # Barre de progression Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Log des métriques et contraintes utilisées
    st.write(f"Métriques optimisées: {metrics}")
    st.write(f"Contraintes appliquées: {constraints}")
    st.write(f"Nombre de combinaisons à tester: {len(param_combinations)}")
    
    # Stockage des valeurs min/max pour normalisation
    metric_ranges = {metric: {'min': float('inf'), 'max': -float('inf')} for metric in metrics}
    
    # Debug: vérifier si les métriques sont bien définies
    if not metrics:
        st.error("Aucune métrique définie pour l'optimisation multi-objectif!")
        return {
            'best_params': None,
            'best_result': None,
            'best_score': 0,
            'results_df': pd.DataFrame(),
            'metric_ranges': {}
        }
    
    # Premier passage pour collecter les statistiques sur les métriques (sur tous les paramètres)
    # Cela nous permettra d'avoir une meilleure normalisation
    st.write("Collecte des statistiques sur les métriques (1ère passe)...")
    
    # Échantillon plus représentatif (25% des combinaisons, min 5, max 50)
    sample_size = max(5, min(50, int(len(param_combinations) * 0.25)))
    sampled_params = param_combinations[:sample_size]
    
    for i, params in enumerate(sampled_params):
        strategy = strategy_generator(**params)
        result = backtester.run(strategy)
        
        # Mettre à jour la barre de progression pour la première passe
        progress_bar.progress(i / len(sampled_params) * 0.2)  # 20% de la barre pour la phase d'initialisation
        
        for metric_name in metrics:
            # Chercher d'abord dans les métriques avancées, puis dans les métriques de base
            value = None
            if metric_name in result['advanced_metrics']:
                value = result['advanced_metrics'][metric_name]
            elif metric_name in result['basic_metrics']:
                value = result['basic_metrics'][metric_name]
                
            if value is not None:
                metric_ranges[metric_name]['min'] = min(metric_ranges[metric_name]['min'], value)
                metric_ranges[metric_name]['max'] = max(metric_ranges[metric_name]['max'], value)
    
    # Log des plages de valeurs des métriques pour l'échantillon initial
    st.write(f"Plages de valeurs des métriques (échantillon initial): {metric_ranges}")
    
    # Vérifier si les plages sont valides (min < max)
    for metric_name, range_values in metric_ranges.items():
        if range_values['min'] >= range_values['max']:
            st.warning(f"La métrique {metric_name} a des valeurs identiques (min={range_values['min']}, max={range_values['max']}). La normalisation sera basée sur la valeur absolue.")
            # Ajuster pour éviter la division par zéro dans la normalisation
            if range_values['min'] == range_values['max'] and range_values['min'] != 0:
                # Si toutes les valeurs sont identiques et non nulles, utiliser la valeur absolue
                metric_ranges[metric_name]['min'] = 0
                metric_ranges[metric_name]['max'] = abs(range_values['min']) * 2
            elif range_values['min'] == range_values['max'] and range_values['min'] == 0:
                # Si toutes les valeurs sont zéro, définir une plage arbitraire
                metric_ranges[metric_name]['min'] = -1
                metric_ranges[metric_name]['max'] = 1
    
    # Pour chaque combinaison de paramètres
    for i, params in enumerate(param_combinations):
        # Mettre à jour la barre de progression (commence à 20% pour cette phase)
        progress_percent = 0.2 + (i / len(param_combinations) * 0.8)
        progress_bar.progress(progress_percent)
        status_text.text(f"Optimisation multi-objectif: test de la combinaison {i+1}/{len(param_combinations)}")
        
        # Créer la stratégie avec ces paramètres
        strategy = strategy_generator(**params)
        
        # Exécuter le backtester
        result = backtester.run(strategy)
        
        # Vérifier les contraintes
        constraints_satisfied = True
        constraints_log = {}
        
        for metric_name, (min_val, max_val) in constraints.items():
            value = None
            # Chercher la métrique dans les résultats
            if metric_name in result['basic_metrics']:
                value = result['basic_metrics'][metric_name]
            elif metric_name in result['advanced_metrics']:
                value = result['advanced_metrics'][metric_name]
            
            # Si la métrique n'est pas trouvée, considérer la contrainte comme non satisfaite
            if value is None:
                st.warning(f"Métrique '{metric_name}' introuvable dans les résultats pour {params}")
                constraints_satisfied = False
                constraints_log[metric_name] = "non trouvée"
                break
                
            # Vérifier si la contrainte est satisfaite
            if min_val is not None and value < min_val:
                constraints_satisfied = False
                constraints_log[metric_name] = f"{value} < {min_val} (min)"
                break
            if max_val is not None and value > max_val:
                constraints_satisfied = False
                constraints_log[metric_name] = f"{value} > {max_val} (max)"
                break
                
            constraints_log[metric_name] = f"{min_val if min_val is not None else '-∞'} <= {value} <= {max_val if max_val is not None else '∞'}"
        
        # Initialiser le résultat avec les paramètres
        result_entry = {**params}
        
        if not constraints_satisfied:
            # Log des contraintes non satisfaites
            if i % 10 == 0:  # Limiter le nombre de logs pour éviter de surcharger l'interface
                st.write(f"Paramètres {params} - Contraintes non satisfaites: {constraints_log}")
            
            # Ajouter un score de 0 pour les contraintes non satisfaites, mais garder la trace dans les résultats
            result_entry['combined_score'] = 0.0
            
            # Ajouter quand même les métriques aux résultats pour la visualisation
            for k, v in result['advanced_metrics'].items():
                result_entry[f"metric_{k}"] = v
            for k, v in result['basic_metrics'].items():
                result_entry[f"basic_{k}"] = v
                
            # Ajouter aux résultats même si les contraintes ne sont pas satisfaites
            results.append(result_entry)
            continue
            
        # Calculer le score multi-objectif normalisé
        score = 0.0
        score_components = {}
        
        for metric_name, weight in metrics.items():
            value = None
            # Chercher la métrique dans les résultats
            if metric_name in result['basic_metrics']:
                value = result['basic_metrics'][metric_name]
            elif metric_name in result['advanced_metrics']:
                value = result['advanced_metrics'][metric_name]
                
            if value is None:
                st.warning(f"Métrique '{metric_name}' non trouvée pour le calcul du score")
                continue
                
            # Mettre à jour les plages min/max pour la normalisation
            metric_ranges[metric_name]['min'] = min(metric_ranges[metric_name]['min'], value)
            metric_ranges[metric_name]['max'] = max(metric_ranges[metric_name]['max'], value)
            
            # Normaliser la valeur si la plage n'est pas nulle
            range_size = metric_ranges[metric_name]['max'] - metric_ranges[metric_name]['min']
            if range_size > 0:
                # Pour les métriques à minimiser (poids négatif), inverser la normalisation
                if weight < 0:
                    normalized_value = 1 - ((value - metric_ranges[metric_name]['min']) / range_size)
                    score_component = normalized_value * abs(weight)
                else:
                    normalized_value = (value - metric_ranges[metric_name]['min']) / range_size
                    score_component = normalized_value * weight
            else:
                # Si toutes les valeurs sont identiques, utiliser directement la valeur
                normalized_value = 1.0  # Utiliser 1.0 si toutes les valeurs sont identiques
                score_component = weight  # Appliquer simplement le poids
                
            score += score_component
            score_components[metric_name] = {
                'value': value,
                'normalized': normalized_value,
                'weight': weight,
                'component': score_component
            }
        
        # Assurer que le score n'est jamais exactement zéro (sauf si toutes les métriques sont à zéro)
        # Cela permet d'éviter les problèmes de visualisation avec des heatmaps pleines de zéros
        if score == 0 and any(comp['value'] != 0 for comp in score_components.values()):
            score = 1e-10  # Valeur très petite mais non nulle
        
        # Si c'est le meilleur score jusqu'à présent
        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
            
            # Log du nouveau meilleur score
            if i % 5 == 0:  # Limiter le nombre de logs
                st.write(f"Nouveau meilleur score ({score:.4f}) avec les paramètres: {params}")
                st.write(f"Composants du score: {score_components}")
        
        # Stocker les résultats avec toutes les métriques et le score combiné
        result_entry['combined_score'] = score
        for k, v in result['advanced_metrics'].items():
            result_entry[f"metric_{k}"] = v
        for k, v in result['basic_metrics'].items():
            result_entry[f"basic_{k}"] = v
        
        results.append(result_entry)
    
    # Finaliser la barre de progression
    progress_bar.progress(100)
    status_text.text("Optimisation multi-objectif terminée!")
    
    # Log des résultats finaux
    if best_params:
        st.write(f"Meilleur score: {best_score:.4f}")
        st.write(f"Meilleurs paramètres: {best_params}")
        
        # Afficher les métriques du meilleur résultat
        st.write("Métriques du meilleur résultat:")
        for metric_name in metrics:
            if metric_name in best_result['advanced_metrics']:
                st.write(f"  - {metric_name}: {best_result['advanced_metrics'][metric_name]:.4f}")
            elif metric_name in best_result['basic_metrics']:
                st.write(f"  - {metric_name}: {best_result['basic_metrics'][metric_name]:.4f}")
    else:
        st.error("Aucune combinaison de paramètres ne satisfait les contraintes!")
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    
    # Vérifier si la colonne combined_score contient autre chose que des zéros
    if 'combined_score' in results_df.columns and (results_df['combined_score'] == 0).all():
        st.warning("Attention: Tous les scores combinés sont à zéro. Vérifiez vos métriques et contraintes.")
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'best_score': best_score,
        'results_df': results_df,
        'metric_ranges': metric_ranges  # Ajouter les plages de métriques pour référence
    }

# Fonctions auxiliaires pour générer des paramètres avec des contraintes plus complexes
def generate_ma_crossover_params(
    short_window_range: Tuple[int, int, int] = (5, 50, 5),
    long_window_range: Tuple[int, int, int] = (20, 200, 10),
    min_gap: int = 10  # Écart minimum entre les fenêtres courte et longue
) -> List[Dict]:
    """
    Génère toutes les combinaisons valides de paramètres pour la stratégie MA Crossover
    """
    short_windows = range(short_window_range[0], short_window_range[1] + 1, short_window_range[2])
    long_windows = range(long_window_range[0], long_window_range[1] + 1, long_window_range[2])
    
    # Créer les combinaisons avec la contrainte que short_window < long_window
    # et qu'il y ait un écart minimum entre les deux
    param_combinations = [
        {'short_window': s, 'long_window': l} 
        for s in short_windows 
        for l in long_windows 
        if s < l and (l - s) >= min_gap
    ]
    
    return param_combinations

def generate_rsi_params(
    period_range: Tuple[int, int, int] = (5, 30, 1),
    oversold_range: Tuple[int, int, int] = (20, 40, 5),
    overbought_range: Tuple[int, int, int] = (60, 80, 5),
    min_band_width: int = 20  # Écart minimum entre les niveaux de surachat et de survente
) -> List[Dict]:
    """
    Génère toutes les combinaisons valides de paramètres pour la stratégie RSI
    """
    periods = range(period_range[0], period_range[1] + 1, period_range[2])
    oversolds = range(oversold_range[0], oversold_range[1] + 1, oversold_range[2])
    overboughts = range(overbought_range[0], overbought_range[1] + 1, overbought_range[2])
    
    # Créer les combinaisons avec des contraintes logiques
    param_combinations = [
        {'rsi_period': p, 'oversold': os, 'overbought': ob} 
        for p in periods 
        for os in oversolds 
        for ob in overboughts 
        if os < ob and (ob - os) >= min_band_width
    ]
    
    return param_combinations

def generate_macd_params(
    fast_period_range: Tuple[int, int, int] = (5, 20, 1),
    slow_period_range: Tuple[int, int, int] = (20, 40, 2),
    signal_period_range: Tuple[int, int, int] = (5, 20, 1),
    min_gap: int = 10  # Écart minimum entre les périodes rapide et lente
) -> List[Dict]:
    """
    Génère toutes les combinaisons valides de paramètres pour la stratégie MACD
    """
    fast_periods = range(fast_period_range[0], fast_period_range[1] + 1, fast_period_range[2])
    slow_periods = range(slow_period_range[0], slow_period_range[1] + 1, slow_period_range[2])
    signal_periods = range(signal_period_range[0], signal_period_range[1] + 1, signal_period_range[2])
    
    # Créer les combinaisons avec des contraintes logiques
    param_combinations = [
        {'fast_period': f, 'slow_period': s, 'signal_period': sig} 
        for f in fast_periods 
        for s in slow_periods 
        for sig in signal_periods 
        if f < s and (s - f) >= min_gap
    ]
    
    return param_combinations

def generate_bollinger_params(
    window_range: Tuple[int, int, int] = (5, 50, 5),
    num_std_range: Tuple[float, float, float] = (1.0, 3.0, 0.5),
) -> List[Dict]:
    """
    Génère toutes les combinaisons valides de paramètres pour la stratégie Bollinger Bands
    """
    windows = range(window_range[0], window_range[1] + 1, window_range[2])
    num_stds = np.arange(num_std_range[0], num_std_range[1] + 0.1, num_std_range[2])
    
    # Créer les combinaisons
    param_combinations = [
        {'window': w, 'num_std': float(ns)} 
        for w in windows 
        for ns in num_stds
    ]
    
    return param_combinations

def generate_supertrend_params(
    period_range: Tuple[int, int, int] = (5, 50, 5),
    multiplier_range: Tuple[float, float, float] = (1.0, 6.0, 0.5),
) -> List[Dict]:
    """
    Génère toutes les combinaisons valides de paramètres pour la stratégie Supertrend
    """
    periods = range(period_range[0], period_range[1] + 1, period_range[2])
    multipliers = np.arange(multiplier_range[0], multiplier_range[1] + 0.1, multiplier_range[2])
    
    # Créer les combinaisons
    param_combinations = [
        {'period': p, 'multiplier': float(m)} 
        for p in periods 
        for m in multipliers
    ]
    
    return param_combinations

# Fonctions de création de stratégies à partir de paramètres
def create_ma_strategy(**params):
    return MovingAverageCrossover(**params)

def create_rsi_strategy(**params):
    return RSIStrategy(**params)

def create_macd_strategy(**params):
    return MACDStrategy(**params)

def create_bollinger_strategy(**params):
    return BollingerBandsStrategy(**params)

def create_supertrend_strategy(**params):
    return SupertrendStrategy(**params)

def optimize_moving_average_crossover(
    backtester: Backtester,
    short_window_range: Tuple[int, int, int] = (5, 50, 5),
    long_window_range: Tuple[int, int, int] = (20, 200, 10),
    metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
    """
    Optimise les paramètres de la stratégie de croisement de moyennes mobiles.
    
    Args:
        backtester: Instance du backtester
        short_window_range: Tuple (min, max, step) pour la fenêtre courte
        long_window_range: Tuple (min, max, step) pour la fenêtre longue
        metric: Métrique à optimiser ('sharpe_ratio', 'sortino_ratio', 'total_pnl', etc.)
    
    Returns:
        Dict avec les meilleurs paramètres et les résultats
    """
    short_windows = range(short_window_range[0], short_window_range[1] + 1, short_window_range[2])
    long_windows = range(long_window_range[0], long_window_range[1] + 1, long_window_range[2])
    
    # Créer les combinaisons de paramètres où short_window < long_window
    param_combinations = [(s, l) for s in short_windows for l in long_windows if s < l]
    
    results = []
    best_value = -float('inf') if metric != 'max_drawdown_percentage' else float('inf')
    best_params = None
    best_result = None
    
    # Barre de progression Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pour chaque combinaison de paramètres
    for i, (short_window, long_window) in enumerate(param_combinations):
        # Mettre à jour la barre de progression
        progress_bar.progress(i / len(param_combinations))
        status_text.text(f"Optimisation en cours: test de la combinaison {i+1}/{len(param_combinations)}")
        
        # Créer la stratégie avec ces paramètres
        strategy = MovingAverageCrossover(
            short_window=short_window,
            long_window=long_window
        )
        
        # Exécuter le backtester
        result = backtester.run(strategy)
        
        # Extraire la métrique à optimiser
        if metric.startswith('advanced_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['advanced_metrics'][metric_key]
        elif metric.startswith('basic_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['basic_metrics'][metric_key]
        else:
            # Par défaut, utiliser le Sharpe ratio
            value = result['advanced_metrics']['sharpe_ratio']
        
        # Si c'est une métrique à minimiser (comme le drawdown)
        if metric == 'max_drawdown_percentage' or metric.endswith('max_drawdown_percentage'):
            is_better = value < best_value
        else:  # Sinon c'est une métrique à maximiser
            is_better = value > best_value
        
        if is_better:
            best_value = value
            best_params = {
                'short_window': short_window,
                'long_window': long_window
            }
            best_result = result
        
        # Stocker les résultats
        results.append({
            'short_window': short_window,
            'long_window': long_window,
            'metric_value': value,
            **{
                f"metric_{k}": v for k, v in result['advanced_metrics'].items()
            },
            **{
                f"basic_{k}": v for k, v in result['basic_metrics'].items()
            }
        })
    
    # Finaliser la barre de progression
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'best_value': best_value,
        'results_df': results_df
    }

def optimize_rsi_strategy(
    backtester: Backtester,
    period_range: Tuple[int, int, int] = (5, 30, 1),
    oversold_range: Tuple[int, int, int] = (20, 40, 5),
    overbought_range: Tuple[int, int, int] = (60, 80, 5),
    metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
    """
    Optimise les paramètres de la stratégie RSI.
    
    Args:
        backtester: Instance du backtester
        period_range: Tuple (min, max, step) pour la période du RSI
        oversold_range: Tuple (min, max, step) pour le niveau de survente
        overbought_range: Tuple (min, max, step) pour le niveau de surachat
        metric: Métrique à optimiser
    
    Returns:
        Dict avec les meilleurs paramètres et les résultats
    """
    periods = range(period_range[0], period_range[1] + 1, period_range[2])
    oversolds = range(oversold_range[0], oversold_range[1] + 1, oversold_range[2])
    overboughts = range(overbought_range[0], overbought_range[1] + 1, overbought_range[2])
    
    # Créer les combinaisons de paramètres où oversold < overbought
    param_combinations = [
        (p, os, ob) for p in periods 
        for os in oversolds 
        for ob in overboughts 
        if os < ob
    ]
    
    results = []
    best_value = -float('inf') if metric != 'max_drawdown_percentage' else float('inf')
    best_params = None
    best_result = None
    
    # Barre de progression Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pour chaque combinaison de paramètres
    for i, (period, oversold, overbought) in enumerate(param_combinations):
        # Mettre à jour la barre de progression
        progress_bar.progress(i / len(param_combinations))
        status_text.text(f"Optimisation en cours: test de la combinaison {i+1}/{len(param_combinations)}")
        
        # Créer la stratégie avec ces paramètres
        strategy = RSIStrategy(
            rsi_period=period,
            oversold=oversold,
            overbought=overbought
        )
        
        # Exécuter le backtester
        result = backtester.run(strategy)
        
        # Extraire la métrique à optimiser
        if metric.startswith('advanced_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['advanced_metrics'][metric_key]
        elif metric.startswith('basic_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['basic_metrics'][metric_key]
        else:
            # Par défaut, utiliser le Sharpe ratio
            value = result['advanced_metrics']['sharpe_ratio']
        
        # Si c'est une métrique à minimiser (comme le drawdown)
        if metric == 'max_drawdown_percentage' or metric.endswith('max_drawdown_percentage'):
            is_better = value < best_value
        else:  # Sinon c'est une métrique à maximiser
            is_better = value > best_value
        
        if is_better:
            best_value = value
            best_params = {
                'rsi_period': period,
                'oversold': oversold,
                'overbought': overbought
            }
            best_result = result
        
        # Stocker les résultats
        results.append({
            'rsi_period': period,
            'oversold': oversold,
            'overbought': overbought,
            'metric_value': value,
            **{
                f"metric_{k}": v for k, v in result['advanced_metrics'].items()
            },
            **{
                f"basic_{k}": v for k, v in result['basic_metrics'].items()
            }
        })
    
    # Finaliser la barre de progression
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'best_value': best_value,
        'results_df': results_df
    }

def optimize_macd_strategy(
    backtester: Backtester,
    fast_period_range: Tuple[int, int, int] = (5, 20, 1),
    slow_period_range: Tuple[int, int, int] = (20, 40, 2),
    signal_period_range: Tuple[int, int, int] = (5, 20, 1),
    metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
    """
    Optimise les paramètres de la stratégie MACD.
    
    Args:
        backtester: Instance du backtester
        fast_period_range: Tuple (min, max, step) pour la période rapide
        slow_period_range: Tuple (min, max, step) pour la période lente
        signal_period_range: Tuple (min, max, step) pour la période du signal
        metric: Métrique à optimiser
    
    Returns:
        Dict avec les meilleurs paramètres et les résultats
    """
    fast_periods = range(fast_period_range[0], fast_period_range[1] + 1, fast_period_range[2])
    slow_periods = range(slow_period_range[0], slow_period_range[1] + 1, slow_period_range[2])
    signal_periods = range(signal_period_range[0], signal_period_range[1] + 1, signal_period_range[2])
    
    # Créer les combinaisons de paramètres où fast_period < slow_period
    param_combinations = [
        (f, s, sig) for f in fast_periods 
        for s in slow_periods 
        for sig in signal_periods 
        if f < s
    ]
    
    results = []
    best_value = -float('inf') if metric != 'max_drawdown_percentage' else float('inf')
    best_params = None
    best_result = None
    
    # Barre de progression Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pour chaque combinaison de paramètres
    for i, (fast_period, slow_period, signal_period) in enumerate(param_combinations):
        # Mettre à jour la barre de progression
        progress_bar.progress(i / len(param_combinations))
        status_text.text(f"Optimisation en cours: test de la combinaison {i+1}/{len(param_combinations)}")
        
        # Créer la stratégie avec ces paramètres
        strategy = MACDStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period
        )
        
        # Exécuter le backtester
        result = backtester.run(strategy)
        
        # Extraire la métrique à optimiser
        if metric.startswith('advanced_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['advanced_metrics'][metric_key]
        elif metric.startswith('basic_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['basic_metrics'][metric_key]
        else:
            # Par défaut, utiliser le Sharpe ratio
            value = result['advanced_metrics']['sharpe_ratio']
        
        # Si c'est une métrique à minimiser (comme le drawdown)
        if metric == 'max_drawdown_percentage' or metric.endswith('max_drawdown_percentage'):
            is_better = value < best_value
        else:  # Sinon c'est une métrique à maximiser
            is_better = value > best_value
        
        if is_better:
            best_value = value
            best_params = {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }
            best_result = result
        
        # Stocker les résultats
        results.append({
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'metric_value': value,
            **{
                f"metric_{k}": v for k, v in result['advanced_metrics'].items()
            },
            **{
                f"basic_{k}": v for k, v in result['basic_metrics'].items()
            }
        })
    
    # Finaliser la barre de progression
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'best_value': best_value,
        'results_df': results_df
    }

def optimize_bollinger_bands_strategy(
    backtester: Backtester,
    window_range: Tuple[int, int, int] = (5, 50, 5),
    num_std_range: Tuple[int, int, float] = (1, 4, 0.5),
    metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
    """
    Optimise les paramètres de la stratégie Bollinger Bands.
    
    Args:
        backtester: Instance du backtester
        window_range: Tuple (min, max, step) pour la fenêtre
        num_std_range: Tuple (min, max, step) pour le nombre d'écarts-types
        metric: Métrique à optimiser
    
    Returns:
        Dict avec les meilleurs paramètres et les résultats
    """
    windows = range(window_range[0], window_range[1] + 1, window_range[2])
    num_stds = np.arange(num_std_range[0], num_std_range[1] + 0.1, num_std_range[2])
    
    # Créer les combinaisons de paramètres
    param_combinations = [(w, ns) for w in windows for ns in num_stds]
    
    results = []
    best_value = -float('inf') if metric != 'max_drawdown_percentage' else float('inf')
    best_params = None
    best_result = None
    
    # Barre de progression Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pour chaque combinaison de paramètres
    for i, (window, num_std) in enumerate(param_combinations):
        # Mettre à jour la barre de progression
        progress_bar.progress(i / len(param_combinations))
        status_text.text(f"Optimisation en cours: test de la combinaison {i+1}/{len(param_combinations)}")
        
        # Créer la stratégie avec ces paramètres
        strategy = BollingerBandsStrategy(
            window=window,
            num_std=num_std
        )
        
        # Exécuter le backtester
        result = backtester.run(strategy)
        
        # Extraire la métrique à optimiser
        if metric.startswith('advanced_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['advanced_metrics'][metric_key]
        elif metric.startswith('basic_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['basic_metrics'][metric_key]
        else:
            # Par défaut, utiliser le Sharpe ratio
            value = result['advanced_metrics']['sharpe_ratio']
        
        # Si c'est une métrique à minimiser (comme le drawdown)
        if metric == 'max_drawdown_percentage' or metric.endswith('max_drawdown_percentage'):
            is_better = value < best_value
        else:  # Sinon c'est une métrique à maximiser
            is_better = value > best_value
        
        if is_better:
            best_value = value
            best_params = {
                'window': window,
                'num_std': num_std
            }
            best_result = result
        
        # Stocker les résultats
        results.append({
            'window': window,
            'num_std': num_std,
            'metric_value': value,
            **{
                f"metric_{k}": v for k, v in result['advanced_metrics'].items()
            },
            **{
                f"basic_{k}": v for k, v in result['basic_metrics'].items()
            }
        })
    
    # Finaliser la barre de progression
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'best_value': best_value,
        'results_df': results_df
    }

def optimize_supertrend_strategy(
    backtester: Backtester,
    period_range: Tuple[int, int, int] = (5, 50, 5),
    multiplier_range: Tuple[int, int, float] = (1, 6, 0.5),
    metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
    """
    Optimise les paramètres de la stratégie Supertrend.
    
    Args:
        backtester: Instance du backtester
        period_range: Tuple (min, max, step) pour la période
        multiplier_range: Tuple (min, max, step) pour le multiplicateur
        metric: Métrique à optimiser
    
    Returns:
        Dict avec les meilleurs paramètres et les résultats
    """
    periods = range(period_range[0], period_range[1] + 1, period_range[2])
    multipliers = np.arange(multiplier_range[0], multiplier_range[1] + 0.1, multiplier_range[2])
    
    # Créer les combinaisons de paramètres
    param_combinations = [(p, m) for p in periods for m in multipliers]
    
    results = []
    best_value = -float('inf') if metric != 'max_drawdown_percentage' else float('inf')
    best_params = None
    best_result = None
    
    # Barre de progression Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pour chaque combinaison de paramètres
    for i, (period, multiplier) in enumerate(param_combinations):
        # Mettre à jour la barre de progression
        progress_bar.progress(i / len(param_combinations))
        status_text.text(f"Optimisation en cours: test de la combinaison {i+1}/{len(param_combinations)}")
        
        # Créer la stratégie avec ces paramètres
        strategy = SupertrendStrategy(
            period=period,
            multiplier=multiplier
        )
        
        # Exécuter le backtester
        result = backtester.run(strategy)
        
        # Extraire la métrique à optimiser
        if metric.startswith('advanced_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['advanced_metrics'][metric_key]
        elif metric.startswith('basic_metrics.'):
            metric_key = metric.split('.')[1]
            value = result['basic_metrics'][metric_key]
        else:
            # Par défaut, utiliser le Sharpe ratio
            value = result['advanced_metrics']['sharpe_ratio']
        
        # Si c'est une métrique à minimiser (comme le drawdown)
        if metric == 'max_drawdown_percentage' or metric.endswith('max_drawdown_percentage'):
            is_better = value < best_value
        else:  # Sinon c'est une métrique à maximiser
            is_better = value > best_value
        
        if is_better:
            best_value = value
            best_params = {
                'period': period,
                'multiplier': multiplier
            }
            best_result = result
        
        # Stocker les résultats
        results.append({
            'period': period,
            'multiplier': multiplier,
            'metric_value': value,
            **{
                f"metric_{k}": v for k, v in result['advanced_metrics'].items()
            },
            **{
                f"basic_{k}": v for k, v in result['basic_metrics'].items()
            }
        })
    
    # Finaliser la barre de progression
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'best_value': best_value,
        'results_df': results_df
    }