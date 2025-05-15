import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from models import Order, OrderSide, OrderType, Position, PositionSide, Trade
from metrics import calculate_basic_metrics, calculate_advanced_metrics, calculate_trade_statistics, calculate_monthly_returns

class Backtester:
    """Classe principale pour le backtesting des stratégies"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 10000,
                 maker_fee: float = 0.001,  # 0.1% fee
                 taker_fee: float = 0.001,
                 slippage: float = 0.001,   # 0.1% slippage
                 position_size: float = 0.1,  # 10% du capital par position
                 risk_per_trade: float = 0.01,  # 1% du capital par trade
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0
                ):
        """
        Initialise le backtester.
        
        Args:
            data: DataFrame avec les données OHLCV
            slippage: Glissement estimé en pourcentage
            position_size: Taille de position en pourcentage du capital
        """
        # Vérifier et préparer les données
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Les données doivent contenir les colonnes: {required_columns}")
            
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.position_size = position_size
        
        # État du backtesting
        self.current_position = None
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        
        # Historique des capitaux
        self.equity_curve = pd.Series(index=data.index, dtype=float)
        self.equity_curve.iloc[0] = initial_capital
        
    def calculate_position_size(self, price: float) -> float:
        """Calcule la taille de la position en fonction du capital actuel"""
        position_value = self.current_capital * self.position_size
        return position_value / price
        
    def apply_slippage(self, price: float, side: OrderSide) -> float:
        """Applique le slippage au prix d'exécution"""
        if side == OrderSide.BUY:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
            
    def calculate_fees(self, price: float, quantity: float, is_maker: bool = False) -> float:
        """Calcule les frais de trading"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return price * quantity * fee_rate
        
    def execute_order(self, order: Order, timestamp: datetime) -> bool:
        """
        Exécute un ordre et met à jour le capital et les positions.
        
        Returns:
            bool: True si l'ordre a été exécuté avec succès
        """
        # Vérifier si l'ordre peut être exécuté
        if not order.price:
            return False
            
        # Déterminer si c'est un ordre maker ou taker
        is_maker = order.type == OrderType.LIMIT
            
        # Appliquer le slippage uniquement pour les ordres market
        execution_price = self.apply_slippage(order.price, order.side) if not is_maker else order.price
            
        # Calculer la quantité si elle n'est pas spécifiée
        if not order.quantity and order.amount:
            order.quantity = order.amount / execution_price
                
        # Calculer les frais en fonction du type d'ordre (maker/taker)
        fees = self.calculate_fees(execution_price, order.quantity, is_maker)
            
        # Vérifier si on a assez de capital pour les frais
        if self.current_capital < fees:
            return False
            
        # Mettre à jour le capital et créer/fermer des positions
        if order.side == OrderSide.BUY:
            if self.current_position is None:
                # Ouvrir une nouvelle position longue
                position = Position(
                    side=PositionSide.LONG,
                    entry_price=execution_price,
                    quantity=order.quantity,
                    entry_time=timestamp,
                    leverage=order.leverage,
                    take_profit=order.take_profit,
                    stop_loss=order.stop_loss,
                    trailing_stop_pct=order.trailing_stop_pct,
                    fees_paid=fees
                )
                self.current_position = position
                self.positions.append(position)
                
                # Créer un nouveau trade
                trade = Trade(position=position, entry_order=order)
                self.trades.append(trade)
                
            else:
                # Fermer une position courte existante
                if self.current_position.side == PositionSide.SHORT:
                    self.close_position(execution_price, timestamp, fees, order)
                    
        else:  # SELL
            if self.current_position is None:
                # Ouvrir une nouvelle position courte
                position = Position(
                    side=PositionSide.SHORT,
                    entry_price=execution_price,
                    quantity=order.quantity,
                    entry_time=timestamp,
                    leverage=order.leverage,
                    take_profit=order.take_profit,
                    stop_loss=order.stop_loss,
                    trailing_stop_pct=order.trailing_stop_pct,
                    fees_paid=fees
                )
                self.current_position = position
                self.positions.append(position)
                
                # Créer un nouveau trade
                trade = Trade(position=position, entry_order=order)
                self.trades.append(trade)
                
            else:
                # Fermer une position longue existante
                if self.current_position.side == PositionSide.LONG:
                    self.close_position(execution_price, timestamp, fees, order)
        
        # Mettre à jour le capital et l'equity curve
        self.current_capital -= fees
        self.equity_curve[timestamp] = self.current_capital
            
        # Enregistrer l'ordre
        self.orders.append(order)
            
        return True
        
    def close_position(self, price: float, timestamp: datetime, fees: float, exit_order: Order = None):
        """Ferme la position actuelle"""
        if self.current_position is None:
            return
            
        self.current_position.close(price, timestamp, fees)
        
        # Mettre à jour le capital avec le PnL
        self.current_capital += self.current_position.pnl
        
        # Mettre à jour le trade correspondant
        for trade in self.trades:
            if trade.position == self.current_position:
                trade.exit_order = exit_order
                break
                
        self.current_position = None
        
    def check_stop_loss_take_profit(self, row: pd.Series):
        """Méthode vide, les stops ont été retirés"""
        pass
    
    def run(self, strategy):
        """
        Exécute le backtesting avec la stratégie donnée.
        
        Args:
            strategy: Une instance de Strategy qui génère les signaux
            
        Returns:
            dict: Résultats du backtesting
        """
        # Réinitialiser l'état
        self.current_capital = self.initial_capital
        self.current_position = None
        self.positions = []
        self.trades = []
        self.orders = []
        self.equity_curve = pd.Series(index=self.data.index, dtype=float)
        self.equity_curve.iloc[0] = self.initial_capital
        
        # Générer les signaux
        signals = strategy.generate_signals(self.data)
        
        # Parcourir les données chronologiquement
        for timestamp, row in signals.iterrows():
            # Fermer les positions existantes sur signal opposé
            if self.current_position:
                if (self.current_position.side == PositionSide.LONG and row['sell_signal']) or \
                   (self.current_position.side == PositionSide.SHORT and row['buy_signal']):
                    # Calculer l'ajustement de prix pour l'ordre limite en tenant compte des frais maker
                    limit_adjustment = self.maker_fee + self.slippage
                    limit_price = row['close'] * (1 + limit_adjustment if self.current_position.side == PositionSide.LONG else 1 - limit_adjustment)
                    
                    limit_order = Order(
                        side=OrderSide.SELL if self.current_position.side == PositionSide.LONG else OrderSide.BUY,
                        type=OrderType.LIMIT,
                        price=limit_price,
                        quantity=self.current_position.quantity,
                        timestamp=timestamp
                    )
                    
                    # Si l'ordre limite ne s'exécute pas, utiliser un ordre market (avec slippage et frais taker)
                    if not self.execute_order(limit_order, timestamp):
                        # Calculer le prix en tenant compte des frais taker et du slippage
                        order_side = OrderSide.SELL if self.current_position.side == PositionSide.LONG else OrderSide.BUY
                        # Pour un ordre market, appliquer directement le slippage (effet prix) 
                        # Les frais taker seront appliqués dans execute_order
                        market_price = self.apply_slippage(row['close'], order_side)
                        
                        market_order = Order(
                            side=order_side,
                            type=OrderType.MARKET,
                            price=market_price,
                            quantity=self.current_position.quantity,
                            timestamp=timestamp
                        )
                        self.execute_order(market_order, timestamp)
            
            # Ouvrir de nouvelles positions
            if row['buy_signal'] and self.current_position is None:
                # Calculer la taille de la position
                quantity = self.calculate_position_size(row['close'])
                
                # Calculer l'ajustement de prix pour l'ordre limite avec les frais maker et slippage
                limit_adjustment = self.maker_fee + self.slippage
                limit_price = row['close'] * (1 - limit_adjustment)  # Prix légèrement inférieur pour achat
                
                limit_order = Order(
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    price=limit_price,
                    quantity=quantity,
                    timestamp=timestamp
                )
                
                # Si l'ordre limite ne s'exécute pas, utiliser un ordre market
                if not self.execute_order(limit_order, timestamp):
                    # Calculer le prix du marché avec slippage pour l'ordre BUY
                    # Les frais taker seront appliqués dans execute_order
                    market_price = self.apply_slippage(row['close'], OrderSide.BUY)
                    
                    market_order = Order(
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        price=market_price,
                        quantity=quantity,
                        timestamp=timestamp
                    )
                    self.execute_order(market_order, timestamp)
                    
            elif row['sell_signal'] and self.current_position is None:
                # Calculer la taille de la position
                quantity = self.calculate_position_size(row['close'])
                
                # Calculer l'ajustement de prix pour l'ordre limite avec les frais maker et slippage
                limit_adjustment = self.maker_fee + self.slippage
                limit_price = row['close'] * (1 + limit_adjustment)  # Prix légèrement supérieur pour vente
                
                limit_order = Order(
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    price=limit_price,
                    quantity=quantity,
                    timestamp=timestamp
                )
                
                # Si l'ordre limite ne s'exécute pas, utiliser un ordre market
                if not self.execute_order(limit_order, timestamp):
                    # Calculer le prix du marché avec slippage pour l'ordre SELL
                    # Les frais taker seront appliqués dans execute_order
                    market_price = self.apply_slippage(row['close'], OrderSide.SELL)
                    
                    market_order = Order(
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        price=market_price,
                        quantity=quantity,
                        timestamp=timestamp
                    )
                    self.execute_order(market_order, timestamp)
            
            # Mettre à jour l'equity curve
            if self.current_position:
                # Calculer la valeur mark-to-market de la position
                if self.current_position.side == PositionSide.LONG:
                    unrealized_pnl = (row['close'] - self.current_position.entry_price) * self.current_position.quantity
                else:
                    unrealized_pnl = (self.current_position.entry_price - row['close']) * self.current_position.quantity
                    
                self.equity_curve[timestamp] = self.current_capital + unrealized_pnl
            else:
                self.equity_curve[timestamp] = self.current_capital
        
        # Fermer la dernière position si elle existe
        if self.current_position:
            self.close_position(
                self.data.iloc[-1]['close'],
                self.data.index[-1],
                self.calculate_fees(self.data.iloc[-1]['close'], self.current_position.quantity, is_maker=False)  # Utiliser les frais taker pour la fermeture finale
            )
        
        # Préparer le DataFrame des trades
        trades_df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        
        # Calculer les métriques
        results = {
            "trades": trades_df,
            "equity_curve": self.equity_curve,
            "basic_metrics": calculate_basic_metrics(trades_df),
            "advanced_metrics": calculate_advanced_metrics(trades_df, self.initial_capital),
            "trade_statistics": calculate_trade_statistics(trades_df),
            "monthly_returns": calculate_monthly_returns(trades_df, self.initial_capital)
        }
        
        return results