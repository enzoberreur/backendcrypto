from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator

class Strategy(ABC):
    """Classe de base abstraite pour toutes les stratégies"""
    
    def __init__(self, name="Strategy"):
        self.name = name
        self.description = "Stratégie de base"
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les signaux de trading pour les données fournies.
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les colonnes buy_signal et sell_signal ajoutées
        """
        pass

class MovingAverageCrossover(Strategy):
    """Stratégie de croisement de moyennes mobiles"""
    
    def __init__(self, short_window=20, long_window=50):
        super().__init__(name=f"Croisement de MA {short_window}/{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.description = f"Génère des signaux d'achat lorsque la MA{short_window} croise au-dessus de la MA{long_window}, et des signaux de vente lors du croisement inverse."
    
    def prepare_data(self, df):
        df = df.copy()
        short_col = f'sma_{self.short_window}'
        long_col = f'sma_{self.long_window}'
        
        if short_col not in df.columns:
            df[short_col] = df['close'].rolling(window=self.short_window).mean()
        if long_col not in df.columns:
            df[long_col] = df['close'].rolling(window=self.long_window).mean()
        
        return df
    
    def generate_signals(self, df):
        df = self.prepare_data(df.copy())
        
        short_col = f'sma_{self.short_window}'
        long_col = f'sma_{self.long_window}'
        
        df['short_above_long'] = df[short_col] > df[long_col]
        df['buy_signal'] = (df['short_above_long'] != df['short_above_long'].shift(1)) & df['short_above_long']
        df['sell_signal'] = (df['short_above_long'] != df['short_above_long'].shift(1)) & ~df['short_above_long']
        
        return df

class RSIStrategy(Strategy):
    """Stratégie basée sur l'indicateur RSI"""
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__(name=f"RSI {rsi_period}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.description = f"Génère des signaux d'achat quand le RSI est sous {oversold} et des signaux de vente quand il est au-dessus de {overbought}"
    
    def generate_signals(self, df):
        df = df.copy()
        
        rsi = RSIIndicator(close=df['close'], window=self.rsi_period)
        df['rsi'] = rsi.rsi()
        
        # Signal d'achat quand le RSI sort de la zone de survente
        df['buy_signal'] = (df['rsi'] > self.oversold) & (df['rsi'].shift(1) <= self.oversold)
        
        # Signal de vente quand le RSI entre dans la zone de surachat
        df['sell_signal'] = (df['rsi'] < self.overbought) & (df['rsi'].shift(1) >= self.overbought)
        
        return df

class MACDStrategy(Strategy):
    """Stratégie basée sur le MACD"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super().__init__(name=f"MACD {fast_period}/{slow_period}/{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.description = f"Génère des signaux basés sur les croisements du MACD ({fast_period},{slow_period},{signal_period})"
    
    def generate_signals(self, df):
        df = df.copy()
        
        macd = MACD(
            close=df['close'],
            window_slow=self.slow_period,
            window_fast=self.fast_period,
            window_sign=self.signal_period
        )
        
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Signal d'achat quand le MACD croise au-dessus de sa ligne de signal
        df['buy_signal'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        # Signal de vente quand le MACD croise en-dessous de sa ligne de signal
        df['sell_signal'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        return df

class BollingerBandsStrategy(Strategy):
    """Stratégie basée sur les bandes de Bollinger"""
    
    def __init__(self, window=20, num_std=2):
        super().__init__(name=f"Bollinger {window},{num_std}")
        self.window = window
        self.num_std = num_std
        self.description = f"Génère des signaux basés sur les bandes de Bollinger ({window},{num_std})"
    
    def generate_signals(self, df):
        df = df.copy()
        
        bollinger = BollingerBands(
            close=df['close'],
            window=self.window,
            window_dev=self.num_std
        )
        
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        
        # Signal d'achat quand le prix touche la bande inférieure
        df['buy_signal'] = df['close'] <= df['bb_low']
        
        # Signal de vente quand le prix touche la bande supérieure
        df['sell_signal'] = df['close'] >= df['bb_high']
        
        return df

class SupertrendStrategy(Strategy):
    """Stratégie basée sur l'indicateur Supertrend"""
    
    def __init__(self, period=10, multiplier=3):
        super().__init__(name=f"Supertrend {period},{multiplier}")
        self.period = period
        self.multiplier = multiplier
        self.description = f"Génère des signaux basés sur l'indicateur Supertrend ({period},{multiplier})"
    
    def calculate_supertrend(self, df):
        # Calculer l'ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = pd.DataFrame(high - low).abs()
        tr2 = pd.DataFrame(high - close.shift(1)).abs()
        tr3 = pd.DataFrame(low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        # Calculer les bandes
        upperband = (high + low) / 2 + self.multiplier * atr
        lowerband = (high + low) / 2 - self.multiplier * atr
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(self.period, len(df)):
            if close[i] > supertrend[i-1]:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])
                direction[i] = -1
                
        return supertrend, direction
    
    def generate_signals(self, df):
        df = df.copy()
        
        supertrend, direction = self.calculate_supertrend(df)
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        
        # Signal d'achat quand la direction change de -1 à 1
        df['buy_signal'] = (df['supertrend_direction'] == 1) & (df['supertrend_direction'].shift(1) == -1)
        
        # Signal de vente quand la direction change de 1 à -1
        df['sell_signal'] = (df['supertrend_direction'] == -1) & (df['supertrend_direction'].shift(1) == 1)
        
        return df

def create_golden_cross_strategy():
    """Crée une stratégie de Golden Cross (MA50 et MA200)"""
    return MovingAverageCrossover(short_window=50, long_window=200)

def create_rsi_macd_strategy():
    """Combine les stratégies RSI et MACD"""
    class RSIMACDStrategy(Strategy):
        def __init__(self):
            super().__init__(name="RSI + MACD")
            self.rsi = RSIStrategy()
            self.macd = MACDStrategy()
            self.description = "Combine les signaux du RSI et du MACD"
        
        def generate_signals(self, df):
            df_rsi = self.rsi.generate_signals(df.copy())
            df_macd = self.macd.generate_signals(df.copy())
            
            df = df.copy()
            df['buy_signal'] = df_rsi['buy_signal'] & df_macd['buy_signal']
            df['sell_signal'] = df_rsi['sell_signal'] & df_macd['sell_signal']
            
            return df
            
    return RSIMACDStrategy()

def create_bollinger_rsi_strategy():
    """Combine les stratégies Bollinger Bands et RSI"""
    class BollingerRSIStrategy(Strategy):
        def __init__(self):
            super().__init__(name="Bollinger + RSI")
            self.bollinger = BollingerBandsStrategy()
            self.rsi = RSIStrategy()
            self.description = "Combine les signaux des Bandes de Bollinger et du RSI"
        
        def generate_signals(self, df):
            df_bb = self.bollinger.generate_signals(df.copy())
            df_rsi = self.rsi.generate_signals(df.copy())
            
            df = df.copy()
            df['buy_signal'] = df_bb['buy_signal'] & df_rsi['buy_signal']
            df['sell_signal'] = df_bb['sell_signal'] & df_rsi['sell_signal']
            
            return df
            
    return BollingerRSIStrategy()