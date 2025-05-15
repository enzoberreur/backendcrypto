import pandas as pd
import numpy as np
from scipy import stats


def calculate_basic_metrics(trades_df):
    """
    Calcule les métriques de base à partir d'une liste de trades.
    
    Args:
        trades_df: DataFrame contenant les trades (colonnes: entry_time, exit_time, pnl, etc.)
    
    Returns:
        dict: Dictionnaire contenant les métriques de base
    """
    if len(trades_df) == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "average_trade": 0.0,
            "max_profit": 0.0,
            "max_loss": 0.0,
            "average_duration_hours": 0.0
        }
    
    # Trades gagnants et perdants
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    # Calcul des métriques de base
    total_trades = len(trades_df)
    total_wins = len(winning_trades)
    total_losses = len(losing_trades)
    win_rate = total_wins / total_trades if total_trades > 0 else 0
    
    # PnL et facteur de profit
    total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    # Moyennes
    avg_profit = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    avg_trade = trades_df['pnl'].mean()
    
    # Extremums
    max_profit = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
    
    # Durée moyenne des trades
    avg_duration = trades_df['duration_hours'].mean() if 'duration_hours' in trades_df.columns else 0
    
    return {
        "total_trades": total_trades,
        "winning_trades": total_wins,
        "losing_trades": total_losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pnl": total_profit - total_loss,
        "average_profit": avg_profit,
        "average_loss": avg_loss,
        "average_trade": avg_trade,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "average_duration_hours": avg_duration
    }


def calculate_advanced_metrics(trades_df, initial_capital=10000):
    """
    Calcule des métriques avancées à partir d'une liste de trades.
    
    Args:
        trades_df: DataFrame contenant les trades
        initial_capital: Capital initial pour les calculs de rendement
    
    Returns:
        dict: Dictionnaire contenant les métriques avancées
    """
    if len(trades_df) == 0:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percentage": 0.0,
            "recovery_factor": 0.0,
            "cagr": 0.0,
            "calmar_ratio": 0.0,
            "expectancy": 0.0
        }
    
    # Créer une série temporelle d'équité
    trades_df = trades_df.sort_values('exit_time')
    
    # Calculer les rendements quotidiens si disponibles
    if 'exit_time' in trades_df.columns:
        trades_df['date'] = pd.to_datetime(trades_df['exit_time']).dt.date
        daily_returns = trades_df.groupby('date')['pnl'].sum() / initial_capital
    else:
        daily_returns = pd.Series(trades_df['pnl'].values) / initial_capital
    
    # Capital cumulatif
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    cumulative_capital = initial_capital * (1 + cumulative_returns)
    
    # Drawdown
    peak = cumulative_capital.expanding().max()
    drawdown = (cumulative_capital - peak) / peak
    max_drawdown = abs(drawdown.min())
    max_drawdown_amount = peak.max() - cumulative_capital[drawdown.idxmin()] if len(drawdown) > 0 else 0
    
    # Ratio de Sharpe (supposant un taux sans risque de 0%)
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Ratio de Sortino (rendements négatifs seulement)
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Calculer l'espérance mathématique
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # CAGR (Compound Annual Growth Rate)
    if 'exit_time' in trades_df.columns and len(trades_df) > 1:
        start_date = pd.to_datetime(trades_df['entry_time'].min())
        end_date = pd.to_datetime(trades_df['exit_time'].max())
        years = (end_date - start_date).days / 365.0
        ending_capital = initial_capital + trades_df['pnl'].sum()
        cagr = (ending_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    else:
        cagr = 0
    
    # Calmar ratio
    calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
    
    # Recovery factor
    total_return = trades_df['pnl'].sum() / initial_capital
    recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf') if total_return > 0 else 0
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown_amount,
        "max_drawdown_percentage": max_drawdown * 100,
        "recovery_factor": recovery_factor,
        "cagr": cagr,
        "calmar_ratio": calmar_ratio,
        "expectancy": expectancy
    }


def calculate_trade_statistics(trades_df):
    """
    Calcule des statistiques détaillées sur les trades.
    
    Args:
        trades_df: DataFrame contenant les trades
        
    Returns:
        dict: Dictionnaire contenant les statistiques
    """
    if len(trades_df) == 0:
        return {
            "consecutive_wins_max": 0,
            "consecutive_losses_max": 0,
            "pnl_by_day_of_week": {},
            "pnl_by_hour": {},
            "trade_duration_stats": {},
        }
    
    # Calculer les séquences de gains/pertes consécutifs
    trades_df = trades_df.sort_values('entry_time')
    trades_df['profitable'] = trades_df['pnl'] > 0
    
    # Séquences de gains consécutifs
    win_streaks = []
    current_streak = 0
    
    for profitable in trades_df['profitable']:
        if profitable:
            current_streak += 1
        else:
            if current_streak > 0:
                win_streaks.append(current_streak)
                current_streak = 0
    
    if current_streak > 0:
        win_streaks.append(current_streak)
        
    # Séquences de pertes consécutives
    loss_streaks = []
    current_streak = 0
    
    for profitable in trades_df['profitable']:
        if not profitable:
            current_streak += 1
        else:
            if current_streak > 0:
                loss_streaks.append(current_streak)
                current_streak = 0
                
    if current_streak > 0:
        loss_streaks.append(current_streak)
    
    # PnL par jour de la semaine
    if 'entry_time' in trades_df.columns:
        trades_df['day_of_week'] = pd.to_datetime(trades_df['entry_time']).dt.day_name()
        pnl_by_day = trades_df.groupby('day_of_week')['pnl'].agg(['sum', 'mean', 'count']).to_dict('index')
    else:
        pnl_by_day = {}
        
    # PnL par heure d'entrée
    if 'entry_time' in trades_df.columns:
        trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        pnl_by_hour = trades_df.groupby('hour')['pnl'].agg(['sum', 'mean', 'count']).to_dict('index')
    else:
        pnl_by_hour = {}
        
    # Statistiques sur la durée des trades
    if 'duration_hours' in trades_df.columns:
        duration_stats = {
            'mean': trades_df['duration_hours'].mean(),
            'median': trades_df['duration_hours'].median(),
            'min': trades_df['duration_hours'].min(),
            'max': trades_df['duration_hours'].max(),
            'std': trades_df['duration_hours'].std()
        }
    else:
        duration_stats = {}
    
    return {
        "consecutive_wins_max": max(win_streaks) if win_streaks else 0,
        "consecutive_losses_max": max(loss_streaks) if loss_streaks else 0,
        "pnl_by_day_of_week": pnl_by_day,
        "pnl_by_hour": pnl_by_hour,
        "trade_duration_stats": duration_stats,
    }


def calculate_monthly_returns(trades_df, initial_capital=10000):
    """
    Calcule les rendements mensuels à partir d'une liste de trades.
    
    Args:
        trades_df: DataFrame contenant les trades
        initial_capital: Capital initial
        
    Returns:
        DataFrame: Rendements mensuels
    """
    if len(trades_df) == 0 or 'exit_time' not in trades_df.columns:
        return pd.DataFrame()
    
    # Ajouter le mois/année pour chaque trade
    trades_df = trades_df.copy()
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['month'] = trades_df['exit_date'].dt.month.astype(int)
    trades_df['year'] = trades_df['exit_date'].dt.year.astype(int)
    
    # Grouper par mois/année et calculer le PnL
    monthly_pnl = trades_df.groupby(['year', 'month'])['pnl'].sum()
    
    # Convertir en DataFrame avec index multi-niveaux pour une meilleure lisibilité
    monthly_pnl = monthly_pnl.reset_index()
    monthly_pnl['date'] = monthly_pnl.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)
    monthly_pnl = monthly_pnl.set_index('date')
    
    # Calculer le rendement mensuel en pourcentage
    monthly_pnl['return_%'] = (monthly_pnl['pnl'] / initial_capital) * 100
    
    return monthly_pnl[['pnl', 'return_%']]


def monte_carlo_simulation(trades_df, initial_capital=10000, num_simulations=1000, confidence_level=0.95):
    """
    Effectue des simulations Monte Carlo pour estimer la distribution des résultats possibles.
    
    Args:
        trades_df: DataFrame contenant les trades
        initial_capital: Capital initial
        num_simulations: Nombre de simulations à exécuter
        confidence_level: Niveau de confiance pour les intervalles de confiance
        
    Returns:
        dict: Résultats des simulations, y compris les intervalles de confiance
    """
    if len(trades_df) < 5:  # Nécessite un minimum de trades
        return {
            "expected_final_capital": initial_capital,
            "confidence_interval_lower": initial_capital,
            "confidence_interval_upper": initial_capital,
            "max_drawdown_mean": 0,
            "profit_factor_mean": 0,
            "win_rate_mean": 0,
            "simulation_percentiles": {}
        }
    
    # Convertir les profits/pertes en rendements en pourcentage
    returns = trades_df['pnl'] / initial_capital
    
    # Exécuter les simulations
    simulation_results = []
    drawdowns = []
    profit_factors = []
    win_rates = []
    
    for _ in range(num_simulations):
        # Échantillonner aléatoirement avec remplacement
        sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
        
        # Calculer le chemin d'équité pour cette simulation
        equity_curve = initial_capital * (1 + np.cumsum(sampled_returns))
        simulation_results.append(equity_curve[-1])
        
        # Calculer le drawdown maximum
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        drawdowns.append(abs(min(drawdown)))
        
        # Calculer le facteur de profit
        positive_returns = sampled_returns[sampled_returns > 0]
        negative_returns = sampled_returns[sampled_returns < 0]
        profit_factor = (sum(positive_returns) / abs(sum(negative_returns))) if sum(negative_returns) != 0 else float('inf')
        profit_factors.append(profit_factor)
        
        # Calculer le taux de réussite
        win_rate = len(positive_returns) / len(sampled_returns)
        win_rates.append(win_rate)
    
    # Calculer les statistiques des résultats
    simulation_results = np.array(simulation_results)
    simulation_results.sort()
    
    # Calculer l'intervalle de confiance
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    lower_bound = np.percentile(simulation_results, lower_percentile * 100)
    upper_bound = np.percentile(simulation_results, upper_percentile * 100)
    
    # Calculer différents percentiles
    percentiles = {}
    for perc in [5, 10, 25, 50, 75, 90, 95]:
        percentiles[str(perc)] = np.percentile(simulation_results, perc)
    
    return {
        "expected_final_capital": np.mean(simulation_results),
        "confidence_interval_lower": lower_bound,
        "confidence_interval_upper": upper_bound,
        "max_drawdown_mean": np.mean(drawdowns) * 100,  # en pourcentage
        "profit_factor_mean": np.mean(profit_factors),
        "win_rate_mean": np.mean(win_rates) * 100,  # en pourcentage
        "simulation_percentiles": percentiles
    }