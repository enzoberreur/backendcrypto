from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Order:
    side: OrderSide
    type: OrderType
    price: float = None
    quantity: float = None
    amount: float = None  # Alternative à quantity (montant en USDT)
    timestamp: datetime = None
    leverage: float = 1.0  # Effet de levier (1.0 = pas d'effet de levier)
    take_profit: float = None  # Prix du take profit
    stop_loss: float = None  # Prix du stop loss
    trailing_stop_pct: float = None  # Pourcentage du trailing stop
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: datetime
    leverage: float = 1.0
    take_profit: float = None
    stop_loss: float = None
    trailing_stop_pct: float = None
    trailing_stop_price: float = None
    exit_price: float = None
    exit_time: datetime = None
    fees_paid: float = 0.0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    status: str = "OPEN"
    risk_per_trade: float = 0.01  # 1% du capital par trade
    atr_period: int = 14  # Période pour le calcul de l'ATR
    atr_multiplier: float = 2.0  # Multiplicateur pour les stops basés sur l'ATR
    trailing_activation_pct: float = 0.01  # 1% de profit pour activer le trailing

    def __post_init__(self):
        """Initialise les stops après la création de la position"""
        # Initialiser le trailing stop si spécifié
        if self.trailing_stop_pct:
            if self.side == PositionSide.LONG:
                self.trailing_stop_price = self.entry_price * (1 - self.trailing_stop_pct)
            else:  # SHORT
                self.trailing_stop_price = self.entry_price * (1 + self.trailing_stop_pct)
        
        # Initialiser le stop loss si spécifié en pourcentage
        if self.stop_loss is None and hasattr(self, 'stop_loss_pct') and self.stop_loss_pct:
            if self.side == PositionSide.LONG:
                self.stop_loss = self.entry_price * (1 - self.stop_loss_pct)
            else:  # SHORT
                self.stop_loss = self.entry_price * (1 + self.stop_loss_pct)
    
    @property
    def is_open(self):
        return self.status == "OPEN"
    
    @property
    def duration(self):
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 3600  # en heures
    
    def calculate_pnl(self, current_price=None):
        """Calcule le PnL pour la position"""
        if not self.is_open and self.exit_price is not None:
            price = self.exit_price
        elif current_price is not None:
            price = current_price
        else:
            return 0.0
            
        if self.side == PositionSide.LONG:
            pnl = (price - self.entry_price) * self.quantity * self.leverage
            pnl_pct = ((price / self.entry_price) - 1) * 100 * self.leverage
        else:  # SHORT
            pnl = (self.entry_price - price) * self.quantity * self.leverage
            pnl_pct = ((self.entry_price / price) - 1) * 100 * self.leverage
            
        # Soustraire les frais
        pnl -= self.fees_paid
        
        # Si la position est fermée, stocker le PnL définitif
        if not self.is_open:
            self.pnl = pnl
            self.pnl_percentage = pnl_pct
            
        return pnl, pnl_pct
    
    def close(self, exit_price, exit_time=None, fees=0.0):
        """Ferme la position au prix spécifié"""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.fees_paid += fees
        self.status = "CLOSED"
        self.calculate_pnl()
    
    def update_trailing_stop(self, current_price: float):
        """Met à jour le trailing stop en fonction du prix actuel"""
        if not self.trailing_stop_pct or not self.trailing_stop_price:
            return False
            
        if self.side == PositionSide.LONG:
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                return True
        else:  # SHORT
            new_stop = current_price * (1 + self.trailing_stop_pct)
            if new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                return True
        return False
    
    def update_stops_with_atr(self, current_atr: float):
        """Met à jour les stops basés sur l'ATR"""
        atr_stop = self.atr_multiplier * current_atr
        
        if self.side == PositionSide.LONG:
            new_stop = self.entry_price - atr_stop
            if self.stop_loss is None or new_stop > self.stop_loss:
                self.stop_loss = new_stop
        else:
            new_stop = self.entry_price + atr_stop
            if self.stop_loss is None or new_stop < self.stop_loss:
                self.stop_loss = new_stop
        
        
@dataclass
class Trade:
    """Représente un trade complet (position ouverte puis fermée)"""
    position: Position
    entry_order: Order
    exit_order: Order = None
    
    @property
    def is_profitable(self):
        return self.position.pnl > 0
        
    def to_dict(self):
        """Convertit le trade en dictionnaire pour l'export"""
        return {
            "entry_time": self.position.entry_time,
            "exit_time": self.position.exit_time,
            "side": self.position.side.value,
            "entry_price": self.position.entry_price,
            "exit_price": self.position.exit_price,
            "quantity": self.position.quantity,
            "leverage": self.position.leverage,
            "fees_paid": self.position.fees_paid,
            "pnl": self.position.pnl,
            "pnl_percentage": self.position.pnl_percentage,
            "duration_hours": self.position.duration
        }