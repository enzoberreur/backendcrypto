import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os

from backtester import Backtester
from strategies import (
    MovingAverageCrossover,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
    SupertrendStrategy,
    create_golden_cross_strategy,
    create_rsi_macd_strategy,
    create_bollinger_rsi_strategy
)
from optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_macd_strategy,
    optimize_bollinger_bands_strategy,
    optimize_supertrend_strategy,
    multi_objective_optimization,
    generate_ma_crossover_params,
    generate_rsi_params,
    generate_macd_params,
    generate_bollinger_params,
    generate_supertrend_params,
    create_ma_strategy,
    create_rsi_strategy,
    create_macd_strategy,
    create_bollinger_strategy,
    create_supertrend_strategy
)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Crypto Trading Backtester",
    page_icon="📈",
    layout="wide"
)

# Fonction pour charger les données historiques
@st.cache_data
def load_historical_data(symbol, timeframe):
    file_path = f"Historique/{symbol}_{timeframe}.csv"
    if not os.path.exists(file_path):
        st.error(f"Le fichier {file_path} n'existe pas!")
        return None
    
    df = pd.read_csv(file_path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    return df

# === FONCTION POUR SUPPORT/RESISTANCE HORIZONTAUX ET CHART CHARTISTE ===
def find_support_resistance(df, window=5, threshold=0.02):
    highs = []
    lows = []
    for i in range(window, len(df) - window):
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
            highs.append((df.index[i], df['high'].iloc[i]))
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
            lows.append((df.index[i], df['low'].iloc[i]))
    resistance_levels = []
    support_levels = []
    for high_date, high_price in highs:
        exists = False
        for i, (_, r_price) in enumerate(resistance_levels):
            if abs(high_price - r_price) / r_price < threshold:
                exists = True
                resistance_levels[i] = (resistance_levels[i][0], (resistance_levels[i][1] + high_price) / 2)
                break
        if not exists:
            resistance_levels.append((high_date, high_price))
    for low_date, low_price in lows:
        exists = False
        for i, (_, s_price) in enumerate(support_levels):
            if abs(low_price - s_price) / s_price < threshold:
                exists = True
                support_levels[i] = (support_levels[i][0], (support_levels[i][1] + low_price) / 2)
                break
        if not exists:
            support_levels.append((low_date, low_price))
    return support_levels, resistance_levels

def find_trend_lines(df, min_points=3, max_lines=3, price_tolerance=0.005):
    highs = [(x, y) for x, y in zip(df.reset_index().index, df['high'])]
    lows = [(x, y) for x, y in zip(df.reset_index().index, df['low'])]
    def line_quality(line_params, points, is_support=True):
        a, b = line_params
        touch_points = []
        violations = 0
        for x, y in points:
            line_y = a * x + b
            error = (y - line_y) / y
            if abs(error) <= price_tolerance:
                touch_points.append((x, y))
            if (is_support and line_y > y) or (not is_support and line_y < y):
                violations += 1
        if len(touch_points) < min_points or violations > 0:
            return -1, None
        if touch_points:
            touch_points.sort(key=lambda p: p[0])
            x_span = touch_points[-1][0] - touch_points[0][0]
            importance = len(touch_points) * (x_span / len(df))
            angle_deg = abs(np.degrees(np.arctan(a)))
            if angle_deg > 45 or angle_deg < 2:
                importance *= 0.5
            return importance, touch_points
        return -1, None
    support_lines = []
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            x1, y1 = lows[i]
            x2, y2 = lows[j]
            if x1 == x2:
                continue
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            importance, touch_points = line_quality((a, b), lows, is_support=True)
            if importance > 0 and touch_points is not None:
                support_lines.append((importance, a, b, touch_points))
    resistance_lines = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            x1, y1 = highs[i]
            x2, y2 = highs[j]
            if x1 == x2:
                continue
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            importance, touch_points = line_quality((a, b), highs, is_support=False)
            if importance > 0 and touch_points is not None:
                resistance_lines.append((importance, a, b, touch_points))
    support_lines.sort(key=lambda x: x[0], reverse=True)
    resistance_lines.sort(key=lambda x: x[0], reverse=True)
    def filter_similar_lines(lines, tolerance=0.01):
        if not lines:
            return []
        filtered_lines = [lines[0]]
        for line in lines[1:]:
            _, a1, b1, _ = line
            is_duplicate = False
            for _, a2, b2, _ in filtered_lines:
                if abs(a1 - a2) < tolerance and abs(b1 - b2) / (abs(b1) + abs(b2) + 1e-10) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_lines.append(line)
        return filtered_lines[:max_lines]
    return (filter_similar_lines(support_lines), filter_similar_lines(resistance_lines))

def plot_chart_analysis(df):
    chart_df = df.copy().iloc[-200:]
    support_lvls, resistance_lvls = find_support_resistance(chart_df, window=5, threshold=0.02)
    trend_supports, trend_resistances = find_trend_lines(chart_df, min_points=3, max_lines=3)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'],
        name='Prix'))
    # Bandes de Bollinger si dispo
    if all(col in chart_df.columns for col in ['upper_band', 'middle_band', 'lower_band']):
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['upper_band'], name="BB Upper", line=dict(color="#616161", width=1)))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['lower_band'], name="BB Lower", line=dict(color="#BDBDBD", width=1), fill='tonexty', fillcolor='rgba(79,195,247,0.08)'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['middle_band'], name="BB Middle", line=dict(color="#90A4AE", width=1, dash="dot")))
    # Supports horizontaux
    for _, price in support_lvls:
        fig.add_shape(type="line", x0=chart_df.index[0], x1=chart_df.index[-1], y0=price, y1=price,
            line=dict(color="#4FC3F7", width=1, dash="dash"))
        fig.add_annotation(x=chart_df.index[-1], y=price, text=f"S: {price:.2f}", showarrow=False, xanchor="right", font=dict(color="#4FC3F7"))
    # Résistances horizontales
    for _, price in resistance_lvls:
        fig.add_shape(type="line", x0=chart_df.index[0], x1=chart_df.index[-1], y0=price, y1=price,
            line=dict(color="#FFB74D", width=1, dash="dash"))
        fig.add_annotation(x=chart_df.index[-1], y=price, text=f"R: {price:.2f}", showarrow=False, xanchor="right", font=dict(color="#FFB74D"))
    # Supports dynamiques (obliques)
    for _, a, b, touch_points in trend_supports:
        start_idx = min(tp[0] for tp in touch_points)
        end_idx = max(tp[0] for tp in touch_points)
        x_start = chart_df.iloc[start_idx].name
        x_end = chart_df.iloc[end_idx].name
        y_start = a * start_idx + b
        y_end = a * end_idx + b
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_end, y1=y_end, line=dict(color="#1976D2", width=2))
        for candle_idx, price in touch_points:
            date = chart_df.iloc[candle_idx].name
            fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers', marker=dict(color='#BA68C8', size=8, symbol='circle-open'), showlegend=False))
    # Résistances dynamiques (obliques)
    for _, a, b, touch_points in trend_resistances:
        start_idx = min(tp[0] for tp in touch_points)
        end_idx = max(tp[0] for tp in touch_points)
        x_start = chart_df.iloc[start_idx].name
        x_end = chart_df.iloc[end_idx].name
        y_start = a * start_idx + b
        y_end = a * end_idx + b
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_end, y1=y_end, line=dict(color="#F57C00", width=2))
        for candle_idx, price in touch_points:
            date = chart_df.iloc[candle_idx].name
            fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers', marker=dict(color='#BA68C8', size=8, symbol='circle-open'), showlegend=False))
    fig.update_layout(
        title="Chart chartiste (200 dernières bougies)",
        height=500,
        template="plotly_dark",
        margin=dict(t=40, b=20)
    )
    return fig

# Fonction pour créer le graphique des prix avec les indicateurs
def create_price_chart(data, strategy_name):
    # Calculer les indicateurs nécessaires si ils n'existent pas déjà
    df = data.copy()
    
    if 'RSI' in strategy_name and 'rsi_14' not in df.columns:
        from ta.momentum import RSIIndicator
        df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # Chandelier principal
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Ajouter les indicateurs en fonction de la stratégie
    if 'MA' in strategy_name or 'Moving Average' in strategy_name:
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    name="SMA 20",
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_50'],
                    name="SMA 50",
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
            
    elif 'RSI' in strategy_name:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi_14'],
                name="RSI",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        # Ajouter les lignes de surachat/survente
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
    elif 'MACD' in strategy_name:
        if 'macd_diff' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['macd_diff'],
                    name="MACD Histogram",
                    marker_color='gray'
                ),
                row=2, col=1
            )
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    name="MACD",
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    name="Signal",
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
        
    elif 'Bollinger' in strategy_name:
        if 'bb_high' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_high'],
                    name="BB Upper",
                    line=dict(color='gray', dash='dash')
                ),
                row=1, col=1
            )
        if 'bb_mid' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_mid'],
                    name="BB Middle",
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        if 'bb_low' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_low'],
                    name="BB Lower",
                    line=dict(color='gray', dash='dash')
                ),
                row=1, col=1
            )
    
    # Mise en page
    fig.update_layout(
        title=f"Prix et Indicateurs - {strategy_name}",
        xaxis_title="Date",
        yaxis_title="Prix",
        height=800
    )
    
    return fig

# Fonction pour créer le graphique de l'equity curve
def create_equity_curve_chart(equity_curve):
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name="Capital",
            line=dict(color='green')
        )
    )
    
    fig.update_layout(
        title="Évolution du Capital",
        xaxis_title="Date",
        yaxis_title="Capital (USDT)",
        height=400
    )
    
    return fig

# Fonction pour créer un graphique des rendements mensuels
def create_monthly_returns_chart(monthly_returns):
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=monthly_returns.index,
            y=monthly_returns['return_%'],
            name="Rendement mensuel",
            marker_color=['red' if x < 0 else 'green' for x in monthly_returns['return_%']]
        )
    )
    
    fig.update_layout(
        title="Rendements Mensuels",
        xaxis_title="Mois",
        yaxis_title="Rendement (%)",
        height=400
    )
    
    return fig

# Initialiser les paramètres de session si ce n'est pas déjà fait
if 'params_optimized' not in st.session_state:
    st.session_state.params_optimized = False
    st.session_state.strategy_params = {}
    st.session_state.optimization_results = None

# Titre principal
st.title("Crypto Trading Backtester")

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("Configuration")
    
    # Sélection de la paire
    symbol = st.selectbox(
        "Paire de trading",
        ["ETHUSDT", "BTCUSDT", "BNBUSDT", "DOGEUSDT", "SOLUSDT"]
    )
    
    # Sélection du timeframe
    timeframe = st.selectbox(
        "Intervalle",
        ["1h", "4h", "1d"]
    )
    
    # Sélection de la stratégie
    strategy_type = st.selectbox(
        "Stratégie",
        [
            "Moving Average Crossover",
            "RSI",
            "MACD",
            "Bollinger Bands",
            "Supertrend",
            "Golden Cross",
            "RSI + MACD",
            "Bollinger + RSI"
        ],
        on_change=lambda: setattr(st.session_state, 'params_optimized', False)
    )
    
    # Paramètres de la stratégie
    # Initialiser ou mettre à jour st.session_state.strategy_params si nécessaire
    if strategy_type not in st.session_state.strategy_params:
        st.session_state.strategy_params[strategy_type] = {}
    
    if strategy_type == "Moving Average Crossover":
        # Récupérer les valeurs optimisées si disponibles
        if st.session_state.params_optimized and 'short_window' in st.session_state.strategy_params[strategy_type]:
            default_short = st.session_state.strategy_params[strategy_type]['short_window']
            default_long = st.session_state.strategy_params[strategy_type]['long_window']
        else:
            default_short, default_long = 20, 50
            
        short_window = st.slider("MA Courte", 5, 50, default_short)
        long_window = st.slider("MA Longue", 20, 200, default_long)
        
        # Mettre à jour la session state
        st.session_state.strategy_params[strategy_type] = {
            'short_window': short_window,
            'long_window': long_window
        }
        
    elif strategy_type == "RSI":
        if st.session_state.params_optimized and 'rsi_period' in st.session_state.strategy_params[strategy_type]:
            default_period = st.session_state.strategy_params[strategy_type]['rsi_period']
            default_oversold = st.session_state.strategy_params[strategy_type]['oversold']
            default_overbought = st.session_state.strategy_params[strategy_type]['overbought']
        else:
            default_period, default_oversold, default_overbought = 14, 30, 70
            
        rsi_period = st.slider("Période RSI", 5, 30, default_period)
        oversold = st.slider("Niveau de survente", 10, 40, default_oversold)
        overbought = st.slider("Niveau de surachat", 60, 90, default_overbought)
        
        # Mettre à jour la session state
        st.session_state.strategy_params[strategy_type] = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought
        }
        
    elif strategy_type == "MACD":
        if st.session_state.params_optimized and 'fast_period' in st.session_state.strategy_params[strategy_type]:
            default_fast = st.session_state.strategy_params[strategy_type]['fast_period']
            default_slow = st.session_state.strategy_params[strategy_type]['slow_period']
            default_signal = st.session_state.strategy_params[strategy_type]['signal_period']
        else:
            default_fast, default_slow, default_signal = 12, 26, 9
            
        fast_period = st.slider("Période rapide", 5, 20, default_fast)
        slow_period = st.slider("Période lente", 20, 40, default_slow)
        signal_period = st.slider("Période du signal", 5, 20, default_signal)
        
        # Mettre à jour la session state
        st.session_state.strategy_params[strategy_type] = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }
        
    elif strategy_type == "Bollinger Bands":
        if st.session_state.params_optimized and 'window' in st.session_state.strategy_params[strategy_type]:
            default_period = st.session_state.strategy_params[strategy_type]['window']
            default_std = st.session_state.strategy_params[strategy_type]['num_std']
        else:
            default_period, default_std = 20, 2.0
            
        bb_period = st.slider("Période", 5, 50, default_period)
        bb_std = st.slider("Nombre d'écarts-types", 1.0, 4.0, float(default_std), step=0.1)
        
        # Mettre à jour la session state
        st.session_state.strategy_params[strategy_type] = {
            'window': bb_period,
            'num_std': bb_std
        }
        
    elif strategy_type == "Supertrend":
        if st.session_state.params_optimized and 'period' in st.session_state.strategy_params[strategy_type]:
            default_period = st.session_state.strategy_params[strategy_type]['period']
            default_mult = st.session_state.strategy_params[strategy_type]['multiplier']
        else:
            default_period, default_mult = 10, 3.0
            
        st_period = st.slider("Période", 5, 50, default_period)
        st_multiplier = st.slider("Multiplicateur", 1.0, 6.0, float(default_mult), step=0.1)
        
        # Mettre à jour la session state
        st.session_state.strategy_params[strategy_type] = {
            'period': st_period,
            'multiplier': st_multiplier
        }
    
    # Ajout d'une section pour l'optimisation des paramètres
    if strategy_type in ["Moving Average Crossover", "RSI", "MACD", "Bollinger Bands", "Supertrend"]:
        st.subheader("Optimisation des paramètres")
        
        # Option pour choisir entre optimisation simple ou multi-objectif
        optimization_mode = st.radio(
            "Mode d'optimisation",
            ["Optimisation simple", "Optimisation multi-objectif"],
            help="L'optimisation multi-objectif permet d'optimiser sur plusieurs métriques simultanément avec des pondérations personnalisées"
        )
        
        # Liste complète des métriques disponibles
        available_metrics = [
            ("Ratio de Sharpe", "sharpe_ratio", "advanced_metrics", True),
            ("Ratio de Sortino", "sortino_ratio", "advanced_metrics", True),
            ("Win Rate", "win_rate", "basic_metrics", True),
            ("Profit Factor", "profit_factor", "basic_metrics", True),
            ("PnL Total", "total_pnl", "basic_metrics", True),
            ("Drawdown Max (%)", "max_drawdown_percentage", "advanced_metrics", False),
            ("CAGR", "cagr", "advanced_metrics", True),
            ("Calmar Ratio", "calmar_ratio", "advanced_metrics", True),
            ("Nombre de trades", "total_trades", "basic_metrics", True),
            ("Expectancy", "expectancy", "advanced_metrics", True)
        ]
        
        if optimization_mode == "Optimisation simple":
            # Sélection d'une seule métrique à optimiser
            optimization_metrics_display = [(m[0], f"{m[2]}.{m[1]}" if m[2] == "basic_metrics" else m[1]) for m in available_metrics]
            
            optimization_metric = st.selectbox(
                "Métrique à optimiser",
                options=[m[0] for m in optimization_metrics_display],
                index=0
            )
            
            # Obtenir la clé réelle de la métrique sélectionnée
            selected_metric_key = [m[1] for m in optimization_metrics_display if m[0] == optimization_metric][0]
            
        else:  # Optimisation multi-objectif
            st.write("Sélectionnez les métriques à optimiser et leurs poids:")
            
            # Créer des colonnes pour afficher les métriques côte à côte
            metric_cols = st.columns(2)
            selected_metrics = {}
            
            for i, metric in enumerate(available_metrics):
                col_idx = i % 2
                with metric_cols[col_idx]:
                    # Pour les métriques à maximiser (comme sharpe), le poids est positif
                    # Pour les métriques à minimiser (comme drawdown), le poids est négatif
                    default_weight = 1.0 if metric[3] else -1.0
                    weight = st.slider(
                        f"{metric[0]} ({'+' if metric[3] else '-'})",
                        min_value=-3.0 if metric[3] else -3.0,
                        max_value=3.0 if metric[3] else 0.0,
                        value=default_weight if st.checkbox(f"Utiliser {metric[0]}", value=False) else 0.0,
                        step=0.1,
                        help=f"{'Maximiser' if metric[3] else 'Minimiser'} cette métrique. Un poids de 0 signifie que la métrique est ignorée."
                    )
                    
                    metric_key = f"{metric[2]}.{metric[1]}" if metric[2] == "basic_metrics" else metric[1]
                    if weight != 0:
                        selected_metrics[metric_key] = weight
            
            if not selected_metrics:
                st.warning("Veuillez sélectionner au moins une métrique à optimiser.")
                selected_metrics = {"sharpe_ratio": 1.0}
            
            # Option pour ajouter des contraintes
            add_constraints = st.checkbox("Ajouter des contraintes", value=False)
            constraints = {}
            
            if add_constraints:
                st.write("Définissez les contraintes min/max pour certaines métriques:")
                
                constraint_cols = st.columns(2)
                for i, metric in enumerate(available_metrics):
                    col_idx = i % 2
                    with constraint_cols[col_idx]:
                        if st.checkbox(f"Contrainte sur {metric[0]}", value=False):
                            min_val = st.number_input(f"Min {metric[0]}", value=None, step=0.1)
                            max_val = st.number_input(f"Max {metric[0]}", value=None, step=0.1)
                            
                            # Vérifier que les contraintes sont valides
                            if (min_val is not None or max_val is not None):
                                metric_key = f"{metric[2]}.{metric[1]}" if metric[2] == "basic_metrics" else metric[1]
                                constraints[metric_key] = (min_val, max_val)
        
        # Paramètres spécifiques pour chaque stratégie
        if strategy_type == "Moving Average Crossover":
            st.subheader("Paramètres d'optimisation")
            short_min = st.number_input("MA Courte (min)", value=5, min_value=2, max_value=50)
            short_max = st.number_input("MA Courte (max)", value=50, min_value=5, max_value=100)
            short_step = st.number_input("MA Courte (pas)", value=5, min_value=1, max_value=10)
            
            long_min = st.number_input("MA Longue (min)", value=20, min_value=10, max_value=100)
            long_max = st.number_input("MA Longue (max)", value=200, min_value=20, max_value=500)
            long_step = st.number_input("MA Longue (pas)", value=10, min_value=1, max_value=20)
            
            min_gap = st.number_input("Écart minimum entre les MAs", value=10, min_value=1, max_value=50)
            
        elif strategy_type == "RSI":
            st.subheader("Paramètres d'optimisation")
            period_min = st.number_input("Période RSI (min)", value=5, min_value=2, max_value=20)
            period_max = st.number_input("Période RSI (max)", value=30, min_value=5, max_value=50)
            period_step = st.number_input("Période RSI (pas)", value=2, min_value=1, max_value=5)
            
            oversold_min = st.number_input("Niveau survente (min)", value=20, min_value=10, max_value=40)
            oversold_max = st.number_input("Niveau survente (max)", value=40, min_value=20, max_value=50)
            oversold_step = st.number_input("Niveau survente (pas)", value=5, min_value=1, max_value=10)
            
            overbought_min = st.number_input("Niveau surachat (min)", value=60, min_value=50, max_value=80)
            overbought_max = st.number_input("Niveau surachat (max)", value=80, min_value=60, max_value=90)
            overbought_step = st.number_input("Niveau surachat (pas)", value=5, min_value=1, max_value=10)
            
            min_band_width = st.number_input("Écart minimum entre niveaux", value=20, min_value=10, max_value=40)
            
        elif strategy_type == "MACD":
            st.subheader("Paramètres d'optimisation")
            fast_min = st.number_input("Période rapide (min)", value=5, min_value=2, max_value=20)
            fast_max = st.number_input("Période rapide (max)", value=20, min_value=5, max_value=30)
            fast_step = st.number_input("Période rapide (pas)", value=1, min_value=1, max_value=5)
            
            slow_min = st.number_input("Période lente (min)", value=20, min_value=10, max_value=40)
            slow_max = st.number_input("Période lente (max)", value=40, min_value=20, max_value=60)
            slow_step = st.number_input("Période lente (pas)", value=2, min_value=1, max_value=5)
            
            signal_min = st.number_input("Période signal (min)", value=5, min_value=2, max_value=15)
            signal_max = st.number_input("Période signal (max)", value=20, min_value=5, max_value=30)
            signal_step = st.number_input("Période signal (pas)", value=1, min_value=1, max_value=5)
            
            min_gap = st.number_input("Écart minimum entre périodes", value=10, min_value=5, max_value=30)
            
        elif strategy_type == "Bollinger Bands":
            st.subheader("Paramètres d'optimisation")
            window_min = st.number_input("Période (min)", value=5, min_value=2, max_value=30)
            window_max = st.number_input("Période (max)", value=50, min_value=10, max_value=100)
            window_step = st.number_input("Période (pas)", value=5, min_value=1, max_value=10)
            
            num_std_min = st.number_input("Nombre d'écarts-types (min)", value=1.0, min_value=0.5, max_value=3.0, step=0.1)
            num_std_max = st.number_input("Nombre d'écarts-types (max)", value=3.0, min_value=1.0, max_value=5.0, step=0.1)
            num_std_step = st.number_input("Nombre d'écarts-types (pas)", value=0.2, min_value=0.1, max_value=1.0, step=0.1)
            
        elif strategy_type == "Supertrend":
            st.subheader("Paramètres d'optimisation")
            period_min = st.number_input("Période (min)", value=5, min_value=2, max_value=30)
            period_max = st.number_input("Période (max)", value=50, min_value=10, max_value=100)
            period_step = st.number_input("Période (pas)", value=5, min_value=1, max_value=10)
            
            mult_min = st.number_input("Multiplicateur (min)", value=1.0, min_value=0.5, max_value=3.0, step=0.1)
            mult_max = st.number_input("Multiplicateur (max)", value=5.0, min_value=1.0, max_value=10.0, step=0.1)
            mult_step = st.number_input("Multiplicateur (pas)", value=0.2, min_value=0.1, max_value=1.0, step=0.1)
        
        # Bouton d'optimisation
        optimize_button = st.button("📊 Optimiser les paramètres")
        
        # Bouton pour réinitialiser les paramètres optimisés
        if st.session_state.params_optimized:
            reset_button = st.button("🔄 Réinitialiser les paramètres")
            if reset_button:
                st.session_state.params_optimized = False
                # Réinitialiser les paramètres pour la stratégie actuelle
                st.session_state.strategy_params[strategy_type] = {}
                st.rerun()
    
    # Paramètres de backtesting
    st.header("Paramètres de Backtesting")
    initial_capital = st.number_input("Capital initial (USDT)", 100, 1000000, 10000)
    position_size = st.slider("Taille de position (%)", 1, 100, 10) / 100
    
    # Paramètres de frais et slippage
    st.subheader("Frais et Slippage")
    fees_help = """
    **Comprendre les frais et le slippage :**
    - **Frais maker** : Appliqués lors de l'ajout de liquidité (ordres limites)
    - **Frais taker** : Appliqués lors du retrait de liquidité (ordres market)
    - **Slippage** : Différence entre prix attendu et prix d'exécution
    """
    st.info(fees_help)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Frais de trading")
        maker_fee = st.slider(
            "Frais maker (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Les frais maker sont généralement plus bas car vous ajoutez de la liquidité au marché"
        ) / 100
        
        taker_fee = st.slider(
            "Frais taker (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Les frais taker sont généralement plus élevés car vous prenez de la liquidité du marché"
        ) / 100
        
    with col2:
        st.markdown("##### Slippage")
        slippage = st.slider(
            "Slippage estimé (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Le slippage représente la différence entre le prix attendu et le prix d'exécution réel. Il augmente avec la volatilité et la taille des ordres."
        ) / 100
        
        st.markdown("""
        <small>
        💡 Plus le marché est volatil ou moins il y a de liquidité, plus le slippage sera important.
        </small>
        """, unsafe_allow_html=True)
    
    # Période de test
    st.header("Période de Test")
    date_range = st.date_input(
        "Période",
        value=(datetime.now() - timedelta(days=365), datetime.now()),
        max_value=datetime.now()
    )

# Charger les données
data = load_historical_data(symbol, timeframe)

if data is not None:
    # Afficher le chart chartiste tout en haut
    st.plotly_chart(plot_chart_analysis(data), use_container_width=True)
    # Filtrer les données selon la période sélectionnée
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    # Créer le backtester et exécuter la stratégie
    backtester = Backtester(
        data=data,
        initial_capital=initial_capital,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        slippage=slippage,
        position_size=position_size
    )
    
    # Si le bouton d'optimisation est pressé, lancer l'optimisation
    if 'optimize_button' in locals() and optimize_button:
        st.header(f"Optimisation de la stratégie {strategy_type} sur {symbol} {timeframe}")
        st.info("L'optimisation peut prendre plusieurs minutes selon le nombre de combinaisons de paramètres...")
        
        # Utiliser la méthode d'optimisation appropriée selon le mode sélectionné
        if 'optimization_mode' in locals() and optimization_mode == "Optimisation multi-objectif":
            # Générer les combinaisons de paramètres selon la stratégie
            if strategy_type == "Moving Average Crossover":
                param_combinations = generate_ma_crossover_params(
                    short_window_range=(short_min, short_max, short_step),
                    long_window_range=(long_min, long_max, long_step),
                    min_gap=min_gap
                )
                strategy_generator = create_ma_strategy
                
            elif strategy_type == "RSI":
                param_combinations = generate_rsi_params(
                    period_range=(period_min, period_max, period_step),
                    oversold_range=(oversold_min, oversold_max, oversold_step),
                    overbought_range=(overbought_min, overbought_max, overbought_step),
                    min_band_width=min_band_width
                )
                strategy_generator = create_rsi_strategy
                
            elif strategy_type == "MACD":
                param_combinations = generate_macd_params(
                    fast_period_range=(fast_min, fast_max, fast_step),
                    slow_period_range=(slow_min, slow_max, slow_step),
                    signal_period_range=(signal_min, signal_max, signal_step),
                    min_gap=min_gap
                )
                strategy_generator = create_macd_strategy
                
            elif strategy_type == "Bollinger Bands":
                param_combinations = generate_bollinger_params(
                    window_range=(window_min, window_max, window_step),
                    num_std_range=(num_std_min, num_std_max, num_std_step)
                )
                strategy_generator = create_bollinger_strategy
                
            elif strategy_type == "Supertrend":
                param_combinations = generate_supertrend_params(
                    period_range=(period_min, period_max, period_step),
                    multiplier_range=(mult_min, mult_max, mult_step)
                )
                strategy_generator = create_supertrend_strategy
            
            # Exécuter l'optimisation multi-objectif
            optimization_results = multi_objective_optimization(
                backtester=backtester,
                strategy_generator=strategy_generator,
                param_combinations=param_combinations,
                metrics=selected_metrics,
                constraints=constraints if 'constraints' in locals() else {}
            )
            
            # La structure du résultat est légèrement différente pour l'optimisation multi-objectif
            if 'best_score' in optimization_results:
                optimization_results['best_value'] = optimization_results['best_score']
            
        else:  # Optimisation simple (une seule métrique)
            # Obtenir la clé réelle de la métrique sélectionnée pour le mode d'optimisation simple
            if 'optimization_metric' in locals() and 'optimization_metrics_display' in locals():
                selected_metric_key = [m[1] for m in optimization_metrics_display if m[0] == optimization_metric][0]
            else:
                selected_metric_key = "sharpe_ratio"  # Valeur par défaut
            
            # Optimisation classique à objectif unique
            if strategy_type == "Moving Average Crossover":
                optimization_results = optimize_moving_average_crossover(
                    backtester=backtester,
                    short_window_range=(short_min, short_max, short_step),
                    long_window_range=(long_min, long_max, long_step),
                    metric=selected_metric_key
                )
                
            elif strategy_type == "RSI":
                optimization_results = optimize_rsi_strategy(
                    backtester=backtester,
                    period_range=(period_min, period_max, period_step),
                    oversold_range=(oversold_min, oversold_max, oversold_step),
                    overbought_range=(overbought_min, overbought_max, overbought_step),
                    metric=selected_metric_key
                )
                
            elif strategy_type == "MACD":
                optimization_results = optimize_macd_strategy(
                    backtester=backtester,
                    fast_period_range=(fast_min, fast_max, fast_step),
                    slow_period_range=(slow_min, slow_max, slow_step),
                    signal_period_range=(signal_min, signal_max, signal_step),
                    metric=selected_metric_key
                )
                
            elif strategy_type == "Bollinger Bands":
                optimization_results = optimize_bollinger_bands_strategy(
                    backtester=backtester,
                    window_range=(window_min, window_max, window_step),
                    num_std_range=(num_std_min, num_std_max, num_std_step),
                    metric=selected_metric_key
                )
                
            elif strategy_type == "Supertrend":
                optimization_results = optimize_supertrend_strategy(
                    backtester=backtester,
                    period_range=(period_min, period_max, period_step),
                    multiplier_range=(mult_min, mult_max, mult_step),
                    metric=selected_metric_key
                )
        
        # Mettre à jour les paramètres avec les meilleurs trouvés et stocker dans session_state
        st.session_state.strategy_params[strategy_type] = optimization_results['best_params']
        
        # Stocker les résultats complets d'optimisation
        st.session_state.optimization_results = optimization_results
        st.session_state.params_optimized = True
        
        # Recharger la page pour que les widgets reflètent les nouveaux paramètres optimaux
        st.rerun()
        
    # Afficher les résultats de l'optimisation précédente si disponibles
    if st.session_state.params_optimized and st.session_state.optimization_results is not None:
        st.header(f"Résultats de l'optimisation précédente")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Meilleurs paramètres trouvés")
            st.json(st.session_state.optimization_results['best_params'])
            
        with col2:
            st.subheader(f"Valeur optimisée")
            st.metric(
                "Valeur obtenue", 
                f"{st.session_state.optimization_results['best_value']:.4f}"
            )
        
        # Afficher le tableau des résultats
        st.subheader("Top 10 des meilleures combinaisons")
        sorted_results = st.session_state.optimization_results['results_df']
        
        # Déterminer la colonne à utiliser pour le tri (metric_value ou combined_score)
        sort_column = 'combined_score' if 'combined_score' in sorted_results.columns else 'metric_value'
        
        # Trier correctement selon le type d'optimisation et la métrique
        if sort_column == 'metric_value':
            # Pour l'optimisation simple
            if 'optimization_metric' in locals() and 'optimization_metrics_display' in locals():
                selected_metric_key = [m[1] for m in optimization_metrics_display if m[0] == optimization_metric][0]
                ascending = selected_metric_key == "max_drawdown_percentage"
            else:
                # Valeur par défaut si optimization_metric n'est pas défini
                ascending = False
            sorted_results = sorted_results.sort_values(by=sort_column, ascending=ascending)
        else:
            # Pour l'optimisation multi-objectif, le score combiné est toujours à maximiser
            sorted_results = sorted_results.sort_values(by=sort_column, ascending=False)
        
        st.dataframe(sorted_results.head(10))
        
        # Visualisation des résultats d'optimisation selon le type de stratégie
        if strategy_type == "Moving Average Crossover" and 'short_window' in sorted_results.columns:
            st.subheader("Visualisation des résultats")
            
            # Déterminer la colonne à utiliser pour la visualisation
            value_column = 'combined_score' if 'combined_score' in sorted_results.columns else 'metric_value'
            
            # Créer un graphique de surface pour voir l'effet des paramètres
            pivot_table = sorted_results.pivot_table(
                values=value_column, 
                index="short_window", 
                columns="long_window", 
                aggfunc="mean"
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='Viridis',
                hoverongaps=False
            ))
            
            value_title = "Score combiné" if value_column == 'combined_score' else "Valeur métrique"
            
            fig.update_layout(
                title=f"Carte de chaleur des {value_title}s",
                xaxis_title="Fenêtre longue",
                yaxis_title="Fenêtre courte",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Visualisation pour la stratégie RSI
        elif strategy_type == "RSI" and 'rsi_period' in sorted_results.columns:
            st.subheader("Visualisation des résultats")
            
            # Déterminer la colonne à utiliser pour la visualisation
            value_column = 'combined_score' if 'combined_score' in sorted_results.columns else 'metric_value'
            
            # Créer un graphique de surface pour voir l'effet des paramètres
            try:
                pivot_table = sorted_results.pivot_table(
                    values=value_column, 
                    index="oversold", 
                    columns="overbought", 
                    aggfunc="mean"
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale='Viridis',
                    hoverongaps=False
                ))
                
                value_title = "Score combiné" if value_column == 'combined_score' else "Valeur métrique"
                
                fig.update_layout(
                    title=f"Carte de chaleur des {value_title}s (niveaux de surachat/survente)",
                    xaxis_title="Niveau de surachat",
                    yaxis_title="Niveau de survente",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer la visualisation pour RSI: {str(e)}")
                
        # Visualisation pour la stratégie MACD
        elif strategy_type == "MACD" and 'fast_period' in sorted_results.columns:
            st.subheader("Visualisation des résultats")
            
            # Déterminer la colonne à utiliser pour la visualisation
            value_column = 'combined_score' if 'combined_score' in sorted_results.columns else 'metric_value'
            
            # Créer un graphique de surface pour voir l'effet des paramètres
            try:
                pivot_table = sorted_results.pivot_table(
                    values=value_column, 
                    index="fast_period", 
                    columns="slow_period", 
                    aggfunc="mean"
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale='Viridis',
                    hoverongaps=False
                ))
                
                value_title = "Score combiné" if value_column == 'combined_score' else "Valeur métrique"
                
                fig.update_layout(
                    title=f"Carte de chaleur des {value_title}s (périodes MACD)",
                    xaxis_title="Période lente",
                    yaxis_title="Période rapide",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer la visualisation pour MACD: {str(e)}")
                
        # Visualisation pour la stratégie Bollinger Bands
        elif strategy_type == "Bollinger Bands" and 'window' in sorted_results.columns:
            st.subheader("Visualisation des résultats")
            
            # Déterminer la colonne à utiliser pour la visualisation
            value_column = 'combined_score' if 'combined_score' in sorted_results.columns else 'metric_value'
            
            # Créer un graphique de surface pour voir l'effet des paramètres
            try:
                pivot_table = sorted_results.pivot_table(
                    values=value_column, 
                    index="window", 
                    columns="num_std", 
                    aggfunc="mean"
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale='Viridis',
                    hoverongaps=False
                ))
                
                value_title = "Score combiné" if value_column == 'combined_score' else "Valeur métrique"
                
                fig.update_layout(
                    title=f"Carte de chaleur des {value_title}s (Bollinger Bands)",
                    xaxis_title="Écarts-types",
                    yaxis_title="Période",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer la visualisation pour Bollinger Bands: {str(e)}")
                
        # Visualisation pour la stratégie Supertrend
        elif strategy_type == "Supertrend" and 'period' in sorted_results.columns:
            st.subheader("Visualisation des résultats")
            
            # Déterminer la colonne à utiliser pour la visualisation
            value_column = 'combined_score' if 'combined_score' in sorted_results.columns else 'metric_value'
            
            # Créer un graphique de surface pour voir l'effet des paramètres
            try:
                pivot_table = sorted_results.pivot_table(
                    values=value_column, 
                    index="period", 
                    columns="multiplier", 
                    aggfunc="mean"
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale='Viridis',
                    hoverongaps=False
                ))
                
                value_title = "Score combiné" if value_column == 'combined_score' else "Valeur métrique"
                
                fig.update_layout(
                    title=f"Carte de chaleur des {value_title}s (Supertrend)",
                    xaxis_title="Multiplicateur",
                    yaxis_title="Période",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer la visualisation pour Supertrend: {str(e)}")
    
    # Mise à jour de la stratégie avec les paramètres actuels (optimisés ou définis manuellement)
    if strategy_type == "Moving Average Crossover":
        strategy = MovingAverageCrossover(
            short_window=short_window,
            long_window=long_window
        )
    elif strategy_type == "RSI":
        strategy = RSIStrategy(rsi_period=rsi_period, oversold=oversold, overbought=overbought)
    elif strategy_type == "MACD":
        strategy = MACDStrategy(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
    elif strategy_type == "Bollinger Bands":
        strategy = BollingerBandsStrategy(window=bb_period, num_std=bb_std)
    elif strategy_type == "Supertrend":
        strategy = SupertrendStrategy(period=st_period, multiplier=st_multiplier)
    elif strategy_type == "Golden Cross":
        strategy = create_golden_cross_strategy()
    elif strategy_type == "RSI + MACD":
        strategy = create_rsi_macd_strategy()
    elif strategy_type == "Bollinger + RSI":
        strategy = create_bollinger_rsi_strategy()
    
    results = backtester.run(strategy)
    
    # Afficher les résultats
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Capital final",
            f"{results['equity_curve'].iloc[-1]:.2f} USDT",
            f"{((results['equity_curve'].iloc[-1] / initial_capital) - 1) * 100:.1f}%"
        )
        
    with col2:
        st.metric(
            "Nombre de trades",
            results['basic_metrics']['total_trades'],
            f"Win rate: {results['basic_metrics']['win_rate']*100:.1f}%"
        )
    
    # Graphique des prix et indicateurs
    st.plotly_chart(create_price_chart(data, strategy_type), use_container_width=True)
    
    # Graphique de l'equity curve
    st.plotly_chart(create_equity_curve_chart(results['equity_curve']), use_container_width=True)
    
    # Métriques détaillées
    st.header("Métriques détaillées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Métriques de base")
        metrics = results['basic_metrics']
        st.write(f"Total trades: {metrics['total_trades']}")
        st.write(f"Trades gagnants: {metrics['winning_trades']}")
        st.write(f"Trades perdants: {metrics['losing_trades']}")
        st.write(f"Win rate: {metrics['win_rate']*100:.1f}%")
        st.write(f"Profit factor: {metrics['profit_factor']:.2f}")
        st.write(f"PnL total: {metrics['total_pnl']:.2f} USDT")
        
    with col2:
        st.subheader("Métriques avancées")
        metrics = results['advanced_metrics']
        st.write(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
        st.write(f"Ratio de Sortino: {metrics['sortino_ratio']:.2f}")
        st.write(f"Drawdown max: {metrics['max_drawdown_percentage']:.1f}%")
        st.write(f"CAGR: {metrics['cagr']*100:.1f}%")
        st.write(f"Ratio de Calmar: {metrics['calmar_ratio']:.2f}")
        st.write(f"Expectancy: {metrics['expectancy']:.2f} USDT")
        
    with col3:
        st.subheader("Statistiques des trades")
        metrics = results['trade_statistics']
        st.write(f"Gains consécutifs max: {int(metrics['consecutive_wins_max'])}")
        st.write(f"Pertes consécutives max: {int(metrics['consecutive_losses_max'])}")
        if 'trade_duration_stats' in metrics and metrics['trade_duration_stats']:
            st.write(f"Durée moyenne: {metrics['trade_duration_stats']['mean']:.1f}h")
            st.write(f"Durée médiane: {metrics['trade_duration_stats']['median']:.1f}h")
    
    # Graphique des rendements mensuels
    st.plotly_chart(create_monthly_returns_chart(results['monthly_returns']), use_container_width=True)
    
    # Tableau des trades
    st.header("Journal des trades")
    if not results['trades'].empty:
        st.dataframe(
            results['trades'].style.format({
                'entry_price': '{:.2f}',
                'exit_price': '{:.2f}',
                'pnl': '{:.2f}',
                'pnl_percentage': '{:.2f}',
                'duration_hours': '{:.2f}'
            })
        )
    else:
        st.write("Aucun trade effectué pendant la période.")
        
else:
    st.error("Impossible de charger les données. Veuillez vérifier que les fichiers historiques sont présents dans le dossier approprié.")