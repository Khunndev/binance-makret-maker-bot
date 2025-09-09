import asyncio
import json
import re
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import uuid
import ccxt
import ccxt.pro as ccxtpro
import os
from collections import deque


def decimal_precision_decimal(value):
    d = Decimal(str(value)).normalize()
    return -d.as_tuple().exponent


def get_client_id():
    header = "x-GRBCT6GB"
    guid = uuid.uuid4().hex  # 32-character hex string
    combined = header + guid
    return combined[:32]  # Extract only the first 32 characters

# ============================================================================
# EXTERNAL CONFIG FILE HANDLER
# ============================================================================

class ConfigFile:
    """Handler for external configuration file"""
    
    @staticmethod
    def create_default_config(filename: str = "market_maker_config_future.json"):
        """Create a default configuration file"""
        default_config = {
            "exchange": {
                "api_key": "your_binance_api_key_here",
                "api_secret": "your_binance_api_secret_here",
                "testnet": True,
                "symbol": "BTC/USDT",
                "base_asset": "BTC",
                "quote_asset": "USDT"
            },
            "trading": {
                "total_capital": 1000.0,
                "max_inventory_ratio": 0.2,
                "max_drawdown_ratio": 0.03,
                "base_spread_ticks": 3.0,
                "volatility_multiplier": 4.0,
                "inventory_skew_strength": 3.0,
                "min_order_lifetime": 10.0,
                "min_base_balance": 0.001,
                "min_quote_balance": 5.0,
                "allow_short_selling": False
            },
            "risk": {
                "risk_aversion": 0.1,
                "market_impact": 0.01,
                "time_horizon": 1.0,
                "use_avellaneda": True
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return filename
    
    @staticmethod
    def load_config(filename: str = "market_maker_config_future.json"):
        """Load configuration from file (supports // and /* */ comments)."""
        if not os.path.exists(filename):
            print(f"Config file {filename} not found. Creating default config...")
            ConfigFile.create_default_config(filename)
            print(f"Please edit {filename} with your API keys and settings.")
            return None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                raw = f.read()
            # Strip // line comments and /* */ block comments
            def strip_json_comments(text: str) -> str:
                # Remove block comments
                text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
                # Remove line comments
                text = re.sub(r"(^|\s)//.*?$", "", text, flags=re.MULTILINE)
                return text
            cleaned = strip_json_comments(raw)
            return json.loads(cleaned)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return None

# ============================================================================
# CONFIGURATION SYSTEM
# ============================================================================

@dataclass
class Config:
    # API Credentials - Now loaded from external file
    API_KEY: str = ""
    API_SECRET: str = ""
    USE_TESTNET: bool = True
    
    # Exchange Settings
    SYMBOL: str = "BTC/USDT"
    BASE_ASSET: str = "BTC"
    QUOTE_ASSET: str = "USDT"
    TICK_SIZE: float = 0.01
    MIN_NOTIONAL: float = 5.0
    MIN_QTY: float = 0.00001
    LOT_SIZE: float = 0.00001
    # Optional override for exchange min qty (useful on futures like BTC=0.001)
    MIN_QTY_OVERRIDE: float = 0.0
    
    # Capital & Risk Management
    TOTAL_CAPITAL: float = 329.0
    MAX_INVENTORY_RATIO: float = 0.2
    MAX_DRAWDOWN_RATIO: float = 0.03
    LIQUIDATION_BUFFER: float = 0.2
    
    # Strategy Parameters
    BASE_SPREAD_TICKS: float = 3.0
    VOLATILITY_MULTIPLIER: float = 4.0
    INVENTORY_SKEW_STRENGTH: float = 3.0
    IMBALANCE_SKEW_STRENGTH: float = 1.0
    
    # Avellaneda-Stoikov Parameters
    RISK_AVERSION: float = 0.1
    MARKET_IMPACT: float = 0.01
    TIME_HORIZON: float = 1.0
    USE_AVELLANEDA: bool = True
    
    # Order Management
    MIN_ORDER_LIFETIME: float = 10.0
    MAX_ORDER_REPLACE_FREQ: float = 10.0
    QUEUE_AHEAD: bool = False
    POST_ONLY: bool = False
    POST_ONLY_BACKOFF_TICKS: int = 1

    # Quoting enhancements
    PEG_TO_BOOK: bool = True
    PEG_OFFSET_TICKS: int = 1
    MIN_SPREAD_TICKS: float = 2.0

    # Futures-specific
    LEVERAGE: int = 5
    MARGIN_MODE: str = "cross"  # cross | isolated
    HEDGE_MODE: bool = False
    
    # Position Management
    MIN_BASE_BALANCE: float = 0.001
    MIN_QUOTE_BALANCE: float = 5.0
    ALLOW_SHORT_SELLING: bool = False

    # TP/SL Brackets
    ENABLE_TP_SL: bool = False
    TP_PCT: float = 0.002  # 0.2%
    SL_PCT: float = 0.004  # 0.4%
    TP_REDUCE_ONLY: bool = True
    SL_REDUCE_ONLY: bool = True
    
    # WebSocket & API
    WS_RECONNECT_DELAY: float = 5.0
    API_TIMEOUT: float = 10.0
    RATE_LIMIT_BUFFER: float = 0.1
    
    # Performance Monitoring
    PRICE_HISTORY_SIZE: int = 1000
    TRADE_HISTORY_SIZE: int = 100
    VOLATILITY_WINDOW: float = 60.0
    
    # Fee tracking (will be populated from API)
    MAKER_FEE_RATE: float = 0.018
    TAKER_FEE_RATE: float = 0.018
    FEE_CURRENCY: str = ""

    @classmethod
    def from_file(cls, filename: str = "market_maker_config_future.json"):
        """Create Config from external file"""
        config_data = ConfigFile.load_config(filename)
        if not config_data:
            return None
        
        config = cls()
        
        # Load exchange settings
        exchange = config_data.get("exchange", {})
        config.API_KEY = exchange.get("api_key", "")
        config.API_SECRET = exchange.get("api_secret", "")
        config.USE_TESTNET = exchange.get("testnet", True)
        config.SYMBOL = exchange.get("symbol", "BTC/USDT")
        config.BASE_ASSET = exchange.get("base_asset", "BTC")
        config.QUOTE_ASSET = exchange.get("quote_asset", "USDT")
        
        # Load trading settings
        trading = config_data.get("trading", {})
        config.TOTAL_CAPITAL = trading.get("total_capital", 329.0)
        config.MAX_INVENTORY_RATIO = trading.get("max_inventory_ratio", 0.2)
        config.MAX_DRAWDOWN_RATIO = trading.get("max_drawdown_ratio", 0.03)
        config.BASE_SPREAD_TICKS = trading.get("base_spread_ticks", 3.0)
        config.VOLATILITY_MULTIPLIER = trading.get("volatility_multiplier", 4.0)
        config.INVENTORY_SKEW_STRENGTH = trading.get("inventory_skew_strength", 3.0)
        config.MIN_ORDER_LIFETIME = trading.get("min_order_lifetime", 10.0)
        config.MAX_ORDER_REPLACE_FREQ = trading.get("max_order_replace_freq", 10.0)
        config.MIN_BASE_BALANCE = trading.get("min_base_balance", 0.001)
        config.MIN_QUOTE_BALANCE = trading.get("min_quote_balance", 5.0)
        config.ALLOW_SHORT_SELLING = trading.get("allow_short_selling", False)
        config.POST_ONLY = trading.get("post_only", False)
        config.POST_ONLY_BACKOFF_TICKS = int(trading.get("post_only_backoff_ticks", config.POST_ONLY_BACKOFF_TICKS))
        # Quoting enhancements
        config.PEG_TO_BOOK = trading.get("peg_to_book", config.PEG_TO_BOOK)
        config.PEG_OFFSET_TICKS = int(trading.get("peg_offset_ticks", config.PEG_OFFSET_TICKS))
        config.MIN_SPREAD_TICKS = float(trading.get("min_spread_ticks", config.MIN_SPREAD_TICKS))
        # TP/SL Brackets
        config.ENABLE_TP_SL = bool(trading.get("enable_tp_sl", config.ENABLE_TP_SL))
        config.TP_PCT = float(trading.get("tp_pct", config.TP_PCT))
        config.SL_PCT = float(trading.get("sl_pct", config.SL_PCT))
        config.TP_REDUCE_ONLY = bool(trading.get("tp_reduce_only", config.TP_REDUCE_ONLY))
        config.SL_REDUCE_ONLY = bool(trading.get("sl_reduce_only", config.SL_REDUCE_ONLY))
        # Optional min qty override
        config.MIN_QTY = float(trading.get("min_qty", 0.0) or 0.0)
        config.MIN_QTY_OVERRIDE = float(trading.get("min_qty_override", 0.0) or 0.0)
        # Futures params (optional in config)
        config.LEVERAGE = int(trading.get("leverage", config.LEVERAGE))
        config.MARGIN_MODE = str(trading.get("margin_mode", config.MARGIN_MODE)).lower()
        config.HEDGE_MODE = bool(trading.get("hedge_mode", config.HEDGE_MODE))
        
        # Load risk settings
        risk = config_data.get("risk", {})
        config.RISK_AVERSION = risk.get("risk_aversion", 0.1)
        config.MARKET_IMPACT = risk.get("market_impact", 0.01)
        config.TIME_HORIZON = risk.get("time_horizon", 1.0)
        config.USE_AVELLANEDA = risk.get("use_avellaneda", True)
        
        return config
    
    @property
    def max_inventory_usd(self) -> float:
        return self.TOTAL_CAPITAL * self.MAX_INVENTORY_RATIO
    
    @property
    def max_drawdown_usd(self) -> float:
        return self.TOTAL_CAPITAL * self.MAX_DRAWDOWN_RATIO

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MarketState:
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    mid_price: float = 0.0
    microprice: float = 0.0
    volatility: float = 0.0
    imbalance: float = 0.0
    regime: str = "range"
    last_update: float = 0.0
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.bid > 0 and self.ask > 0 else 0.0

@dataclass
class Quotes:
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: float
    can_place_bid: bool = True
    can_place_ask: bool = True
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.avg_price
    
    def update_unrealized_pnl(self, current_price: float):
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)

@dataclass
class Balance:
    """Balance tracking structure"""
    base_free: float = 0.0
    base_total: float = 0.0
    quote_free: float = 0.0
    quote_total: float = 0.0
    last_update: float = 0.0

@dataclass
class Trade:
    """Trade record for P&L tracking"""
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    timestamp: float
    fee: float = 0.0
    fee_currency: str = ""

class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None
    
    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

# ============================================================================
# CCXT BINANCE CLIENT - FIXED VERSION
# ============================================================================

class BinanceCCXTClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        # Logger first
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.post_only = False
        
        # Initialize CCXT exchange for USD-M Futures
        exchange_class = ccxt.binance
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'rateLimit': 100,
            'options': {
                'defaultType': 'future',  # USD-M futures
                'defaultSubType': 'linear',
            }
        })
        # Sandbox/testnet toggle (Futures supports dedicated endpoints)
        try:
            if hasattr(self.exchange, 'set_sandbox_mode'):
                self.exchange.set_sandbox_mode(bool(testnet))
        except Exception as e:
            self.logger.warning(f"Failed to set sandbox mode on REST client: {e}")
        
        # Initialize CCXT Pro for WebSocket
        try:
            self.exchange_ws = ccxtpro.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'defaultSubType': 'linear',
                }
            })
            try:
                if hasattr(self.exchange_ws, 'set_sandbox_mode'):
                    self.exchange_ws.set_sandbox_mode(bool(testnet))
            except Exception as e:
                self.logger.warning(f"Failed to set sandbox mode on WS client: {e}")
        except Exception as e:
            logging.warning(f"CCXT Pro not available: {e}. WebSocket features will be limited.")
            self.exchange_ws = None
        
        self.markets_loaded = False
        # Last order diagnostics
        self.last_order_error: Optional[str] = None
        self.last_order_details: Optional[dict] = None
        
    async def initialize(self):
        """Initialize exchange connection and load markets"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.exchange.load_markets)
            self.markets_loaded = True
            # fetch_status may not be available/meaningful on futures testnet
            try:
                if getattr(self.exchange, 'has', {}).get('fetchStatus', False):
                    await loop.run_in_executor(None, self.exchange.fetch_status)
            except Exception as e:
                if self.testnet:
                    self.logger.warning("Skipping fetch_status on Binance Futures testnet")
                else:
                    self.logger.warning(f"fetch_status failed: {e}")
            
            self.logger.info(f"Connected to {'Testnet' if self.testnet else 'Live'} Binance")
            self.logger.info(f"Loaded {len(self.exchange.markets)} markets")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            return False

    async def configure_futures(self, symbol: str, leverage: int = 5, margin_mode: str = 'cross', hedge_mode: bool = False):
        """Configure leverage, margin mode, and position mode for futures"""
        try:
            loop = asyncio.get_event_loop()
            market = self.exchange.market(symbol)
            symbol_id = market.get('id', symbol.replace('/', ''))
            # Position mode (one-way vs hedge)
            try:
                if getattr(self.exchange, 'has', {}).get('setPositionMode', False):
                    await loop.run_in_executor(None, self.exchange.set_position_mode, bool(hedge_mode))
                else:
                    # Fallback to raw endpoint
                    if hasattr(self.exchange, 'fapiPrivate_post_positionside_dual'):
                        params = { 'dualSidePosition': 'true' if hedge_mode else 'false' }
                        await loop.run_in_executor(None, self.exchange.fapiPrivate_post_positionside_dual, params)
            except Exception as e:
                self.logger.warning(f"set_position_mode failed: {e}")
            # Margin mode
            mmode = 'ISOLATED' if str(margin_mode).lower().startswith('iso') else 'CROSSED'
            try:
                if getattr(self.exchange, 'has', {}).get('setMarginMode', False):
                    await loop.run_in_executor(None, self.exchange.set_margin_mode, mmode, symbol)
                else:
                    if hasattr(self.exchange, 'fapiPrivate_post_margintype'):
                        params = { 'symbol': symbol_id, 'marginType': mmode }
                        await loop.run_in_executor(None, self.exchange.fapiPrivate_post_margintype, params)
            except Exception as e:
                self.logger.warning(f"set_margin_mode failed: {e}")
            # Leverage
            lev = max(1, int(leverage))
            try:
                if getattr(self.exchange, 'has', {}).get('setLeverage', False):
                    await loop.run_in_executor(None, self.exchange.set_leverage, lev, symbol)
                else:
                    if hasattr(self.exchange, 'fapiPrivate_post_leverage'):
                        params = { 'symbol': symbol_id, 'leverage': lev }
                        await loop.run_in_executor(None, self.exchange.fapiPrivate_post_leverage, params)
            except Exception as e:
                self.logger.warning(f"set_leverage failed: {e}")
            self.logger.info(f"Futures configured: leverage={lev}, margin={mmode}, hedge_mode={hedge_mode}")
        except Exception as e:
            self.logger.error(f"configure_futures error: {e}")
    
    async def get_account_balance(self):
        """Get account balance"""
        try:
            loop = asyncio.get_event_loop()
            balance = await loop.run_in_executor(None, self.exchange.fetch_balance)
            return balance
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 20):
        """Get order book"""
        try:
            loop = asyncio.get_event_loop()
            order_book = await loop.run_in_executor(None, self.exchange.fetch_order_book, symbol, limit)
            return order_book
        except Exception as e:
            self.logger.error(f"Failed to fetch order book for {symbol}: {e}")
            return None

    async def get_position(self, symbol: str):
        """Get futures position for symbol"""
        try:
            loop = asyncio.get_event_loop()
            # Prefer unified fetch_positions
            if getattr(self.exchange, 'has', {}).get('fetchPositions', False):
                positions = await loop.run_in_executor(None, self.exchange.fetch_positions, [symbol])
                for p in positions or []:
                    if p.get('symbol') == symbol:
                        return p
                return positions[0] if positions else None
            # Fallback to raw endpoint
            if hasattr(self.exchange, 'fapiPrivate_get_positionrisk'):
                market = self.exchange.market(symbol)
                params = { 'symbol': market.get('id', symbol.replace('/', '')) }
                data = await loop.run_in_executor(None, self.exchange.fapiPrivate_get_positionrisk, params)
                if isinstance(data, list) and data:
                    # pick matching symbol
                    for d in data:
                        if d.get('symbol') == params['symbol']:
                            return d
                    return data[0]
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch position: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, amount: float, price: float = None, order_type: str = 'limit'):
        """Place an order with proper validation - FIXED for zero amounts"""
        try:
            # reset last diagnostics
            self.last_order_error = None
            self.last_order_details = {'symbol': symbol, 'side': side, 'amount': float(amount), 'price': float(price) if price else None, 'type': order_type}
            # Ensure inputs are floats
            amount = float(amount)
            if price is not None:
                price = float(price)
            
            # Check for zero or invalid amounts BEFORE validation
            if amount <= 0:
                self.logger.error(f"Cannot place order: amount is {amount}")
                return None
            
            if order_type == 'limit' and (price is None or price <= 0):
                self.logger.error(f"Cannot place limit order: invalid price {price}")
                return None
            
            # Get market info for validation
            market = self.get_market_info(symbol)
            if not market:
                raise ValueError(f"Market info not available for {symbol}")
            
            # Log original values
            self.logger.debug(f"Original order: {side} {amount:.8f} @ {price:.4f}")
            
            # Validate order parameters
            amount = self._validate_amount(amount, market)
            if price:
                price = self._validate_price(price, market)
            
            # Check AGAIN after validation for zero amounts
            if amount <= 0:
                self.logger.error(f"Original order: {side} {amount:.8f} @ {price:.4f}")
                self.logger.error(f"Invalid amount after validation: {amount}")
                return None
            
            if price and price <= 0:
                self.logger.error(f"Original order: {side} {amount:.8f} @ {price:.4f}")
                self.logger.error(f"Invalid price after validation: {price}")
                return None
            
            # Check minimum notional
            if price and amount:
                notional = float(amount) * float(price)
                min_notional = float(market.get('limits', {}).get('cost', {}).get('min', 5.0))
                
                if notional < min_notional:
                    self.logger.warning(f"Order notional ${notional:.2f} below minimum ${min_notional:.2f}")
                    # Adjust amount to meet minimum notional
                    new_amount = (min_notional * 1.1) / price
                    amount = self._validate_amount(new_amount, market)
                    notional = amount * price
                    self.logger.info(f"Adjusted order size to {amount:.8f} (notional: ${notional:.2f})")
            
            # Final validation
            if amount <= 0:
                self.logger.error(f"Final amount is invalid: {amount}")
                return None
            
            self.logger.debug(f"Final order: {side} {amount:.8f} @ {price:.4f}")

            params = {
                'newClientOrderId': get_client_id()
            }
            # For Binance futures, enforce post-only via GTX to avoid taker fills
            if getattr(self, 'post_only', False):
                params['postOnly'] = True
                # ccxt maps postOnly for binance futures, but be explicit
                params['timeInForce'] = 'GTX'

            loop = asyncio.get_event_loop()
            order = await loop.run_in_executor(
                None, 
                self.exchange.create_order,
                symbol, order_type, side, float(amount), float(price) if price else None, params
            )
            self.logger.info(f"Order placed: {order['id']} - {side} {amount:.8f} {symbol} @ {price:.4f}")
            self.last_order_details['result'] = 'ok'
            self.last_order_details['id'] = order.get('id')
            return order
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            self.logger.debug(f"Order details - Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Type: {order_type}")
            try:
                self.last_order_error = str(e)
                if self.last_order_details is not None:
                    self.last_order_details['error'] = self.last_order_error
            except Exception:
                pass
            return None

    async def place_reduce_only_limit(self, symbol: str, side: str, amount: float, price: float):
        """Place a reduce-only limit order (do not enforce post-only)."""
        try:
            loop = asyncio.get_event_loop()
            market = self.get_market_info(symbol)
            amount = self._validate_amount(float(amount), market)
            price = self._validate_price(float(price), market)
            if amount <= 0 or price <= 0:
                return None
            params = {
                'newClientOrderId': get_client_id(),
                'reduceOnly': True,
                'timeInForce': 'GTC',
            }
            order = await loop.run_in_executor(
                None,
                self.exchange.create_order,
                symbol, 'limit', side, float(amount), float(price), params
            )
            return order
        except Exception as e:
            self.logger.error(f"Failed to place reduce-only limit: {e}")
            return None

    async def place_stop_market_close(self, symbol: str, side: str, stop_price: float, amount: Optional[float] = None):
        """Place a STOP_MARKET close-position order at stop_price.
        If amount is None, attempt to use closePosition to close all.
        """
        try:
            loop = asyncio.get_event_loop()
            market = self.get_market_info(symbol)
            stop_price = self._validate_price(float(stop_price), market)
            params = {
                'newClientOrderId': get_client_id(),
                'reduceOnly': True,
                'stopPrice': float(stop_price),
                'closePosition': True,
                'type': 'STOP_MARKET',
                # binance futures typically requires positionSide in hedge mode; omitted when HEDGE_MODE False
            }
            amt = float(amount) if amount is not None else None
            # ccxt create_order signature: symbol, type, side, amount=None, price=None, params={}
            order = await loop.run_in_executor(
                None,
                self.exchange.create_order,
                symbol, 'STOP_MARKET', side, amt if amt else None, None, params
            )
            return order
        except Exception as e:
            self.logger.error(f"Failed to place stop-market close: {e}")
            return None
    

    def _validate_amount(self, amount: float, market: dict) -> float:
        """Validate and adjust order amount - FIXED to handle zero properly"""
        try:
            amount = float(amount)
            
            # If amount is zero or negative, return 0 immediately
            if amount <= 0:
                return 0.0
            
            limits = market.get('limits', {}).get('amount', {})
            min_amount = float(limits.get('min', 0.00001))
            max_amount = float(limits.get('max', float('inf')))
            
            # Get step size (lot size)
            precision = market.get('precision', {})
            if 'amount' in precision:
                if isinstance(precision['amount'], int):
                    amount_precision = precision['amount']
                    step_size = 10 ** (-amount_precision)
                else:
                    step_size = float(precision['amount'])
                    amount_precision = decimal_precision_decimal(step_size)
            else:
                amount_precision = 8
                step_size = 0.00000001
            
            # Round to step size
            if step_size > 0:
                amount = round(amount / step_size) * step_size
            
            # Round to precision
            amount = round(amount, amount_precision)
            
            # Ensure minimum
            if amount < min_amount:
                amount = min_amount
                if step_size > 0:
                    amount = round(amount / step_size) * step_size
                amount = round(amount, amount_precision)
            
            # Ensure maximum  
            if amount > max_amount:
                amount = max_amount
                
            return float(amount)
        except Exception as e:
            self.logger.error(f"Error validating amount {amount}: {e}")
            return 0.0
    
    def _validate_price(self, price: float, market: dict) -> float:
        """Validate and adjust order price"""
        try:
            price = float(price)
            
            limits = market.get('limits', {}).get('price', {})
            min_price = float(limits.get('min', 0.01))
            max_price = float(limits.get('max', float('inf')))
            
            # Get tick size
            precision = market.get('precision', {})
            if 'price' in precision:
                if isinstance(precision['price'], int):
                    price_precision = precision['price']
                    tick_size = 10 ** (-price_precision)
                else:
                    tick_size = float(precision['price'])
                    price_precision = decimal_precision_decimal(tick_size)
            else:
                price_precision = 2
                tick_size = 0.01
            
            # Round to tick size
            if tick_size > 0:
                price = round(price / tick_size) * tick_size
            
            # Round to precision
            price = round(price, price_precision)
            
            # Ensure minimum
            if price < min_price:
                price = min_price
                
            # Ensure maximum
            if price > max_price:
                price = max_price
                
            return float(price)
        except Exception as e:
            self.logger.error(f"Error validating price {price}: {e}")
            return float(price)
    
    async def cancel_order(self, order_id: str, symbol: str):
        """Cancel an order"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.exchange.cancel_order, order_id, symbol)
            self.logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return None
    
    async def get_open_orders(self, symbol: str = None):
        """Get open orders"""
        try:
            loop = asyncio.get_event_loop()
            orders = await loop.run_in_executor(None, self.exchange.fetch_open_orders, symbol)
            return orders
        except Exception as e:
            self.logger.error(f"Failed to fetch open orders: {e}")
            return []
    
    async def get_my_trades(self, symbol: str, limit: int = 100):
        """Get recent trades"""
        try:
            loop = asyncio.get_event_loop()
            trades = await loop.run_in_executor(None, self.exchange.fetch_my_trades, symbol, None, limit)
            return trades
        except Exception as e:
            self.logger.error(f"Failed to fetch trades for {symbol}: {e}")
            return []
    
    async def get_ticker(self, symbol: str):
        """Get ticker data"""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, self.exchange.fetch_ticker, symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return None
    
    async def watch_order_book(self, symbol: str, limit: int = 20):
        """Watch order book via WebSocket (CCXT Pro)"""
        if not self.exchange_ws:
            return None
        try:
            order_book = await self.exchange_ws.watch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            self.logger.error(f"Failed to watch order book: {e}")
            return None
    
    def get_market_info(self, symbol: str):
        """Get market information"""
        try:
            if not self.markets_loaded:
                self.logger.warning("Markets not loaded yet")
                return None
                
            if symbol in self.exchange.markets:
                market = self.exchange.markets[symbol]
                return market
            else:
                self.logger.error(f"Symbol {symbol} not found in markets")
                available_symbols = [s for s in self.exchange.markets.keys() if symbol.replace('/', '') in s]
                if available_symbols:
                    self.logger.info(f"Similar symbols found: {available_symbols[:5]}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to get market info for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close exchange connections"""
        try:
            if self.exchange_ws:
                await self.exchange_ws.close()
            if hasattr(self.exchange, 'close') and asyncio.iscoroutinefunction(self.exchange.close):
                await self.exchange.close()
            elif hasattr(self.exchange, 'close'):
                self.exchange.close()
            self.logger.info("Exchange connections closed")
        except Exception as e:
            self.logger.error(f"Error closing exchange connections: {e}")

    async def get_trading_fees(self, symbol: str = None):
        """Fetch trading fees from exchange"""
        try:
            loop = asyncio.get_event_loop()
            # For futures, prefer market fees from market info
            market = None
            if symbol:
                try:
                    market = self.exchange.market(symbol)
                except Exception:
                    market = None
            if market:
                return {
                    'maker': market.get('maker', 0.0002),
                    'taker': market.get('taker', 0.0004),
                    'percentage': True,
                    'tierBased': True
                }
            # Fallback to exchange-wide fees if available
            fees = await loop.run_in_executor(None, self.exchange.fetch_trading_fees)
            if symbol and symbol in fees:
                return fees[symbol]
            # Default
            return {
                'maker': 0.0002,
                'taker': 0.0004,
                'percentage': True,
                'tierBased': True
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch trading fees: {e}")
            # Return default fees
            return {
                'maker': 0.0002,
                'taker': 0.0004,
                'percentage': True,
                'tierBased': True
            }

# ============================================================================
# WEBSOCKET MANAGER FOR CCXT PRO
# ============================================================================

class WebSocketManager:
    def __init__(self, client: BinanceCCXTClient, symbol: str):
        self.client = client
        self.symbol = symbol
        self.running = False
        self.callbacks = {
            'order_book': [],
            'trades': [],
            'balance': []
        }
        
    def add_callback(self, event_type: str, callback):
        """Add callback for WebSocket events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            
    
    async def start(self):
        """Start WebSocket connections"""
        if not self.client.exchange_ws:
            logging.warning("CCXT Pro not available, using REST API polling")
            return False
        
        self.running = True
        
        tasks = [
            asyncio.create_task(self._watch_order_book()),
            asyncio.create_task(self._watch_balance()),
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
        finally:
            self.running = False
    
    async def _watch_order_book(self):
        """Watch order book updates"""
        while self.running:
            try:
                order_book = await self.client.watch_order_book(self.symbol)
                if order_book:
                    for callback in self.callbacks['order_book']:
                        try:
                            await callback(order_book)
                        except Exception as e:
                            logging.error(f"Order book callback error: {e}")
            except Exception as e:
                logging.error(f"Order book watch error: {e}")
                await asyncio.sleep(5)
    
    async def _watch_balance(self):
        """Watch balance updates"""
        while self.running:
            try:
                if hasattr(self.client.exchange_ws, 'watch_balance'):
                    balance = await self.client.exchange_ws.watch_balance()
                    if balance:
                        for callback in self.callbacks['balance']:
                            try:
                                await callback(balance)
                            except Exception as e:
                                logging.error(f"Balance callback error: {e}")
            except Exception as e:
                logging.error(f"Balance watch error: {e}")
                await asyncio.sleep(10)

# ============================================================================
# MARKET MAKER BOT - FIXED WITH REALIZED P&L TRACKING
# ============================================================================

class MarketMakerBot:
    def __init__(self, config: Config):
        self.config = config
        self.client = BinanceCCXTClient(config.API_KEY, config.API_SECRET, testnet=config.USE_TESTNET)
        # propagate post-only preference to client
        self.client.post_only = bool(config.POST_ONLY)
        self.ws_manager = None
        
        # Market State
        self.market_state = MarketState(config.SYMBOL)
        self.price_history = deque(maxlen=config.PRICE_HISTORY_SIZE)
        self.volatility_ema = EMA(alpha=2.0 / (config.VOLATILITY_WINDOW + 1))
        
        # Position & P&L
        self.position = Position(config.SYMBOL)
        self.balance = Balance()
        self.starting_balance = config.TOTAL_CAPITAL
        self.max_drawdown = 0.0
        self.kill_switch_active = False
        
        # FIXED: Trade tracking for realized P&L
        self.trade_history: List[Trade] = []
        self.last_trade_id: Optional[str] = None
        
        # Order Management
        self.current_quotes: Optional[Quotes] = None
        self.open_orders: Dict[str, Dict] = {}
        self.last_order_time = 0.0
        self.last_quote_time = 0.0
        
        # Market Info
        self.market_info = None
        self.tick_size = config.TICK_SIZE
        self.min_qty = config.MIN_QTY
        self.lot_size = config.LOT_SIZE
        self.min_notional = config.MIN_NOTIONAL
        
        # Metrics
        self.total_trades = 0
        self.bid_fills = 0
        self.ask_fills = 0
        self.total_volume = 0.0
        self.vwap_bid = 0.0
        self.vwap_ask = 0.0
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Track bracket orders (TP/SL)
        self.bracket_orders: Dict[str, Optional[str]] = {'tp': None, 'sl': None}
    
    async def initialize_fees(self):
        """Initialize fee information from exchange"""
        try:
            fee_info = await self.client.get_trading_fees(self.config.SYMBOL)
            
            if fee_info:
                self.config.MAKER_FEE_RATE = fee_info.get('maker', 0.001)
                self.config.TAKER_FEE_RATE = fee_info.get('taker', 0.001)
                
                # Check if fees are percentage-based (they should be)
                if not fee_info.get('percentage', True):
                    self.logger.warning("Fees are not percentage-based, this might affect calculations")
                
                self.logger.info(f"Trading fees: Maker={self.config.MAKER_FEE_RATE*100:.3f}%, "
                            f"Taker={self.config.TAKER_FEE_RATE*100:.3f}%")
            else:
                self.logger.warning("Using default fee rates")
                
        except Exception as e:
            self.logger.error(f"Error initializing fees: {e}")
            # Use defaults
            self.config.MAKER_FEE_RATE = 0.001
            self.config.TAKER_FEE_RATE = 0.001

    async def refresh_fees_periodically(self, interval: int = 3600):
        """Periodically refresh fee information"""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.initialize_fees()
                self.logger.info("Refreshed fee information")
            except Exception as e:
                self.logger.error(f"Error refreshing fees: {e}")

    async def start(self):
        """Start the market maker bot"""
        self.logger.info("Starting Binance Market Maker Bot with CCXT")
        
        if not await self.client.initialize():
            self.logger.error("Failed to initialize CCXT client")
            return
        
        await self.load_market_info()
        # Configure futures environment (leverage, margin mode, position mode)
        try:
            await self.client.configure_futures(
                self.config.SYMBOL,
                leverage=self.config.LEVERAGE,
                margin_mode=self.config.MARGIN_MODE,
                hedge_mode=self.config.HEDGE_MODE,
            )
        except Exception as e:
            self.logger.warning(f"Futures configuration step failed: {e}")
        await self.initialize_market_data()

        # Start periodic fee refresh
        asyncio.create_task(self.refresh_fees_periodically())

        await self.start_websockets()
        await self.trading_loop()
    
    async def load_market_info(self):
        """Load market information from exchange"""
        try:
            await asyncio.sleep(1)
            
            symbol_formats = [
                self.config.SYMBOL,
                f"{self.config.BASE_ASSET}/{self.config.QUOTE_ASSET}",
                self.config.SYMBOL.replace('USDT', '/USDT'),
            ]
            
            market_info = None
            actual_symbol = None
            
            for symbol_format in symbol_formats:
                market_info = self.client.get_market_info(symbol_format)
                if market_info:
                    actual_symbol = symbol_format
                    break
            
            if market_info:
                self.market_info = market_info
                
                limits = market_info.get('limits', {})
                precision = market_info.get('precision', {})
                
                # Price precision and tick size
                if 'price' in precision:
                    price_precision = precision['price']
                    if isinstance(price_precision, int):
                        self.tick_size = 10 ** (-price_precision)
                    else:
                        self.tick_size = float(price_precision)
                
                # Quantity precision and lot size  
                if 'amount' in precision:
                    amount_precision = precision['amount']
                    if isinstance(amount_precision, int):
                        self.lot_size = 10 ** (-amount_precision)
                    else:
                        self.lot_size = float(amount_precision)
                
                # Minimum quantity
                if 'amount' in limits and 'min' in limits['amount']:
                    self.min_qty = limits['amount']['min']
                
                # Minimum notional
                if 'cost' in limits and 'min' in limits['cost']:
                    self.min_notional = limits['cost']['min']

                # Binance Futures-specific filters fallback (ensure correct tick/lot/mins)
                info = market_info.get('info', {}) or {}
                filters = info.get('filters', []) or []
                try:
                    for f in filters:
                        ftype = f.get('filterType') or f.get('filterType'.lower())
                        if not ftype:
                            continue
                        if ftype in ('LOT_SIZE', 'MARKET_LOT_SIZE'):
                            step = float(f.get('stepSize') or 0) or 0.0
                            mqty = float(f.get('minQty') or 0) or 0.0
                            if step > 0:
                                self.lot_size = step
                            if mqty > 0:
                                self.min_qty = mqty
                        elif ftype == 'PRICE_FILTER':
                            tick = float(f.get('tickSize') or 0) or 0.0
                            if tick > 0:
                                self.tick_size = tick
                        elif ftype in ('MIN_NOTIONAL', 'NOTIONAL'):
                            mn = f.get('notional')
                            if mn is None:
                                mn = f.get('minNotional')
                            try:
                                mn = float(mn) if mn is not None else 0.0
                            except Exception:
                                mn = 0.0
                            if mn and mn > 0:
                                self.min_notional = mn
                except Exception as _:
                    # Non-fatal, keep previously parsed values
                    pass
                
                if actual_symbol != self.config.SYMBOL:
                    self.logger.info(f"Using symbol format: {actual_symbol}")
                    self.config.SYMBOL = actual_symbol

                # Canonicalize via ccxt.market(symbol) to pick correct futures mapping (e.g., BTC/USDT:USDT)
                try:
                    canonical = self.client.exchange.market(self.config.SYMBOL)
                    if canonical:
                        limits2 = canonical.get('limits', {}) or {}
                        precision2 = canonical.get('precision', {}) or {}
                        # Prefer canonical tick/lot/mins
                        # tick
                        pxp = precision2.get('price')
                        if isinstance(pxp, int):
                            self.tick_size = 10 ** (-pxp)
                        elif pxp:
                            self.tick_size = float(pxp)
                        # lot/qty
                        amtp = precision2.get('amount')
                        if isinstance(amtp, int):
                            self.lot_size = 10 ** (-amtp)
                        elif amtp:
                            self.lot_size = float(amtp)
                        aq = limits2.get('amount', {})
                        if isinstance(aq, dict) and 'min' in aq:
                            self.min_qty = float(aq.get('min'))
                        co = limits2.get('cost', {})
                        if isinstance(co, dict) and 'min' in co:
                            self.min_notional = float(co.get('min'))
                except Exception:
                    pass

                # Apply user override if provided (e.g., force BTC futures 0.001)
                # Only override the minimum quantity; do not change lot_size (step size) here.
                if float(getattr(self.config, 'MIN_QTY_OVERRIDE', 0.0) or 0.0) > 0:
                    self.min_qty = max(self.min_qty, float(self.config.MIN_QTY_OVERRIDE))

                self.logger.info(f"Market info loaded: tick={self.tick_size}, lot={self.lot_size}, min_qty={self.min_qty}, min_notional=${self.min_notional}")
                
            else:
                self.logger.error(f"Could not load market info")
                self.logger.warning("Using default market parameters")
                
        except Exception as e:
            self.logger.error(f"Error loading market info: {e}")
    
    async def initialize_market_data(self):
        """Initialize market data and position"""
        await self.update_balance()
        
        order_book = await self.client.get_order_book(self.config.SYMBOL)
        if order_book:
            await self.update_market_state_from_ccxt(order_book)
        
        # Initialize fee information
        await self.initialize_fees()

        # FIXED: Only track trades made after bot starts
        await self.load_trade_history_and_calculate_pnl()
        
        # Set position from exchange balance, not from P&L calculation
        self.position.quantity = self.balance.base_total
        
        self.logger.info(f"Initialized - Position: {self.position.quantity} {self.config.BASE_ASSET}")
        self.logger.info(f"Balance - {self.config.BASE_ASSET}: {self.balance.base_free:.6f}, {self.config.QUOTE_ASSET}: {self.balance.quote_free:.2f}")
        self.logger.info(f"Realized P&L: ${self.position.realized_pnl:.2f}")
    
    async def load_trade_history_and_calculate_pnl(self):
        """Load trade history but don't calculate P&L from past trades"""
        try:
            # Just get the latest trade ID to know where to start tracking
            trades = await self.client.get_my_trades(self.config.SYMBOL, limit=1)
            if trades:
                self.last_trade_id = str(trades[-1]['id'])
                self.logger.info(f"Starting to track trades from ID: {self.last_trade_id}")
            else:
                self.last_trade_id = None
                self.logger.info("No existing trades found")
        except Exception as e:
            self.logger.error(f"Error loading initial trade ID: {e}")
            self.last_trade_id = None

    def calculate_realized_pnl(self):
        """Calculate realized P&L only from trades made during this session"""
        try:
            if not self.trade_history:
                return
            
            realized_pnl = 0.0
            
            for trade in self.trade_history:
                # Use actual fees from trade data, not estimated rates
                actual_fee = trade.fee
                
                if trade.side == 'buy':
                    # For buys, we're spending quote currency
                    cost = trade.amount * trade.price + actual_fee
                    realized_pnl -= cost
                else:  # sell
                    # For sells, we're receiving quote currency
                    revenue = trade.amount * trade.price - actual_fee
                    realized_pnl += revenue
            
            self.position.realized_pnl = realized_pnl
            self.logger.debug(f"P&L calculation complete: Realized=${realized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating realized P&L: {e}")

    async def update_balance(self):
        """Update futures wallet and position"""
        balance_data = await self.client.get_account_balance()
        if balance_data:
            # Futures wallet (USDT) is the relevant margin asset
            quote_balance = balance_data.get('total', {}).get(self.config.QUOTE_ASSET, 0.0)
            quote_free = balance_data.get('free', {}).get(self.config.QUOTE_ASSET, 0.0)

            # Fetch position separately
            try:
                pos = await self.client.get_position(self.config.SYMBOL)
                if pos:
                    qty = float(pos.get('contracts') or pos.get('positionAmt') or 0.0)
                    entry = float(pos.get('entryPrice') or pos.get('entry') or 0.0)
                    self.position.quantity = qty
                    if entry > 0:
                        self.position.avg_price = entry
                else:
                    self.position.quantity = 0.0
            except Exception as e:
                self.logger.warning(f"Failed to fetch futures position: {e}")

            # Update wallet balances
            self.balance.base_total = 0.0
            self.balance.base_free = 0.0
            self.balance.quote_total = float(quote_balance)
            self.balance.quote_free = float(quote_free)
            self.balance.last_update = time.time()
            
            self.logger.debug(
                f"Futures wallet updated - {self.config.QUOTE_ASSET}: {self.balance.quote_free:.2f} free"
            )
        
    
    async def check_for_new_trades(self):
        """Check for new trades and update realized P&L - FIXED VERSION"""
        try:
            # Fetch recent trades
            new_trades = await self.client.get_my_trades(self.config.SYMBOL, limit=50)
            if not new_trades:
                return
            
            # Find trades newer than our last recorded trade
            new_trade_found = False
            for trade_data in new_trades:
                trade_id = str(trade_data['id'])
                
                # Skip if we've already processed this trade
                if any(t.id == trade_id for t in self.trade_history):
                    continue
                
                # If we have a last_trade_id, only process newer trades
                if self.last_trade_id and int(trade_id) <= int(self.last_trade_id):
                    continue
                    
                # New trade found
                new_trade_found = True
                trade = Trade(
                    id=trade_id,
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    amount=float(trade_data['amount']),
                    price=float(trade_data['price']),
                    timestamp=float(trade_data['timestamp']) / 1000,
                    fee=float(trade_data.get('fee', {}).get('cost', 0)),
                    fee_currency=trade_data.get('fee', {}).get('currency', '')
                )
                
                self.trade_history.append(trade)
                
                # Log the fill
                self.logger.info(f"New fill detected: {trade.side.upper()} {trade.amount:.6f} @ ${trade.price:.2f}")
                
                # Update fill counts
                if trade.side == 'buy':
                    self.bid_fills += 1
                else:
                    self.ask_fills += 1
                
                self.total_trades += 1
                self.total_volume += trade.amount
                
                # Update last trade ID
                self.last_trade_id = trade_id
            
            if new_trade_found:
                # Update position from exchange (not from P&L calculation)
                await self.update_balance()
                
                # Recalculate realized P&L only from new trades
                self.calculate_realized_pnl()
                self.logger.info(f"Updated realized P&L: ${self.position.realized_pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error checking for new trades: {e}")

    
    def can_place_buy_order(self, order_size: float, price: float) -> bool:
        """Check if we can place a buy order (Futures margin check)"""
        if price <= 0 or order_size <= 0:
            return False
        # Required margin = notional / leverage (+ buffer)
        notional = order_size * price
        lev = max(1, int(getattr(self.config, 'LEVERAGE', 1)))
        required_margin = (notional / lev) * 1.02
        if self.balance.quote_free < required_margin:
            self.logger.info(
                f"Buy blocked (futures): need margin ~{required_margin:.2f} {self.config.QUOTE_ASSET}, "
                f"have {self.balance.quote_free:.2f}"
            )
            return False
        if self.balance.quote_free < self.config.MIN_QUOTE_BALANCE:
            self.logger.info(
                f"Buy blocked: free {self.config.QUOTE_ASSET} {self.balance.quote_free:.2f} "
                f"below min_quote_balance {self.config.MIN_QUOTE_BALANCE:.2f}"
            )
            return False
        return True
    
    def can_place_sell_order(self, order_size: float) -> bool:
        """Check if we can place a sell order (Futures margin check; shorting allowed)"""
        if order_size <= 0:
            return False
        # For futures shorts, margin similar to buys (use mid price for estimate)
        price = max(self.market_state.mid_price, 0.0)
        if price <= 0:
            return False
        notional = order_size * price
        lev = max(1, int(getattr(self.config, 'LEVERAGE', 1)))
        required_margin = (notional / lev) * 1.02
        if self.balance.quote_free < required_margin:
            self.logger.info(
                f"Sell blocked (futures): need margin ~{required_margin:.2f} {self.config.QUOTE_ASSET}, "
                f"have {self.balance.quote_free:.2f}"
            )
            return False
        return True
    
    async def start_websockets(self):
        """Start WebSocket connections"""
        self.ws_manager = WebSocketManager(self.client, self.config.SYMBOL)
        
        self.ws_manager.add_callback('order_book', self.on_order_book_update)
        self.ws_manager.add_callback('balance', self.on_balance_update)
        
        asyncio.create_task(self.ws_manager.start())
        self.logger.info("WebSocket connections started")
    
    async def on_order_book_update(self, order_book):
        """Handle order book updates from WebSocket"""
        await self.update_market_state_from_ccxt(order_book)
    
    async def on_balance_update(self, balance):
        """Handle balance updates from WebSocket (fill detection)"""
        try:
            # Check for position changes (fills)
            if self.config.BASE_ASSET in balance.get('total', {}):
                new_base_total = balance['total'][self.config.BASE_ASSET]
                
                if new_base_total != self.balance.base_total:
                    fill_qty = new_base_total - self.balance.base_total
                    self.logger.info(f"Fill detected via WebSocket: {fill_qty:+.8f} {self.config.BASE_ASSET}")
                    
                    # Update balance immediately
                    self.balance.base_total = new_base_total
                    self.balance.base_free = balance.get('free', {}).get(self.config.BASE_ASSET, 0.0)
                    
                    # Check for new trades and update P&L
                    await self.check_for_new_trades()
            
            # Update quote balance
            if self.config.QUOTE_ASSET in balance.get('total', {}):
                self.balance.quote_total = balance['total'][self.config.QUOTE_ASSET]
                self.balance.quote_free = balance.get('free', {}).get(self.config.QUOTE_ASSET, 0.0)
            
            self.balance.last_update = time.time()
            
        except Exception as e:
            self.logger.error(f"Error processing balance update: {e}")
    
    async def update_market_state_from_ccxt(self, order_book):
        """Update market state from CCXT order book format"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return
            
            best_bid, bid_qty = bids[0][0], bids[0][1]
            best_ask, ask_qty = asks[0][0], asks[0][1]
            
            self.market_state.bid = best_bid
            self.market_state.ask = best_ask
            self.market_state.mid_price = (best_bid + best_ask) / 2
            self.market_state.last_update = time.time()
            
            # Calculate microprice
            total_qty = bid_qty + ask_qty
            if total_qty > 0:
                self.market_state.microprice = (best_bid * ask_qty + best_ask * bid_qty) / total_qty
            
            # Calculate imbalance
            self.market_state.imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0
            
            # Update price history and volatility
            self.price_history.append((time.time(), self.market_state.mid_price))
            self.update_volatility()
            
            # Update regime
            self.market_state.regime = self.detect_regime()
            
            # Update position P&L
            self.position.update_unrealized_pnl(self.market_state.mid_price)
            
            # Generate new quotes if needed
            await self.maybe_update_quotes()
            
        except Exception as e:
            self.logger.error(f"Error updating market state: {e}")
    
    def update_volatility(self):
        """Update volatility using EWMA of returns"""
        if len(self.price_history) < 2:
            return
        
        recent_prices = list(self.price_history)[-60:]
        returns = []
        
        for i in range(1, len(recent_prices)):
            prev_price = recent_prices[i-1][1]
            curr_price = recent_prices[i][1]
            if prev_price > 0:
                returns.append(math.log(curr_price / prev_price))
        
        if returns:
            volatility = math.sqrt(sum(r**2 for r in returns) / len(returns)) * math.sqrt(60 * 10)
            self.market_state.volatility = self.volatility_ema.update(volatility)
    
    def detect_regime(self) -> str:
        """Detect market regime (trend vs range)"""
        if len(self.price_history) < 20:
            return "range"
        
        recent_prices = [p[1] for p in list(self.price_history)[-20:]]
        
        x = list(range(len(recent_prices)))
        n = len(recent_prices)
        
        sum_x = sum(x)
        sum_y = sum(recent_prices)
        sum_xy = sum(x[i] * recent_prices[i] for i in range(n))
        sum_x2 = sum(xi**2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        normalized_slope = abs(slope / (sum_y / n))
        
        return "trend" if normalized_slope > 0.001 else "range"
    
    async def maybe_update_quotes(self):
        """Update quotes if conditions are met"""
        now = time.time()
        
        # Allow initial placement immediately; throttle only after we have quotes
        if self.current_quotes is not None:
            if now - self.last_quote_time < self.config.MAX_ORDER_REPLACE_FREQ:
                return
        
        if self.kill_switch_active:
            await self.cancel_all_orders()
            return
        
        await self.update_balance()
        
        new_quotes = self.generate_quotes()
        if not new_quotes:
            return
        
        if self.should_replace_orders(new_quotes):
            await self.replace_orders(new_quotes)
            self.last_quote_time = now
    
    def generate_quotes(self) -> Optional[Quotes]:
        """Generate bid/ask quotes"""
        if self.market_state.mid_price <= 0:
            return None
        
        try:
            if self.config.USE_AVELLANEDA:
                quotes = self.generate_avellaneda_quotes()
            else:
                quotes = self.generate_volatility_quotes()
                
            if quotes:
                quotes.can_place_bid = self.can_place_buy_order(quotes.bid_size, quotes.bid_price)
                quotes.can_place_ask = self.can_place_sell_order(quotes.ask_size)
                
            return quotes
        except Exception as e:
            self.logger.error(f"Error generating quotes: {e}")
            return None
    

    # Additional Fix: More conservative volatility-based quotes as fallback
    def generate_volatility_quotes(self) -> Quotes:
        """Generate quotes using simple volatility-based spread - MADE MORE CONSERVATIVE"""
        mid_price = self.market_state.mid_price
        fee_adjustment = mid_price * self.config.MAKER_FEE_RATE * 4
        volatility = max(self.market_state.volatility, 0.001)
        
        # More conservative spread calculation
        base_spread = max(
            self.config.BASE_SPREAD_TICKS * self.tick_size,
            mid_price * 0.0005  # Minimum 0.05% spread
        )
        
        # Reduce volatility impact
        vol_spread = min(
            self.config.VOLATILITY_MULTIPLIER * volatility * mid_price * 0.1,  # REDUCED volatility impact
            mid_price * 0.01  # Cap at 1% 
        )
        
        half_spread = (base_spread + vol_spread) / 2 + fee_adjustment
        
        # Conservative inventory and imbalance skew
        inventory_skew = self.get_inventory_skew() * 0.5  # REDUCED impact
        imbalance_skew = self.get_imbalance_skew() * 0.5  # REDUCED impact
        
        # Calculate prices
        bid_price = mid_price - half_spread + inventory_skew - imbalance_skew
        ask_price = mid_price + half_spread + inventory_skew + imbalance_skew

        # ADJUST FOR FEES using dynamically fetched rates
        #bid_price = bid_price * (1 - self.config.MAKER_FEE_RATE)
        #ask_price = ask_price * (1 + self.config.MAKER_FEE_RATE)
        
        # Ensure reasonable bounds
        bid_price = max(bid_price, mid_price * 0.99)   # No more than 1% below mid
        ask_price = min(ask_price, mid_price * 1.01)   # No more than 1% above mid
        
        # Optional: peg quotes to top of book within constraints
        if getattr(self.config, 'PEG_TO_BOOK', False):
            offset = max(0, int(getattr(self.config, 'PEG_OFFSET_TICKS', 1))) * (self.tick_size or 0)
            if self.market_state.bid:
                # For buy, peg below best bid
                peg_bid = self.market_state.bid - offset
                bid_price = min(bid_price, peg_bid)
            if self.market_state.ask:
                # For sell, peg above best ask
                peg_ask = self.market_state.ask + offset
                ask_price = max(ask_price, peg_ask)

        # Round to tick size
        bid_price = self.round_to_tick(bid_price)
        ask_price = self.round_to_tick(ask_price)
        
        # Enforce minimum spread
        min_spread = max(getattr(self.config, 'MIN_SPREAD_TICKS', 2.0) * (self.tick_size or 0), mid_price * 0.0001)
        if ask_price - bid_price < min_spread:
            spread = max(min_spread, self.tick_size * 2, mid_price * 0.0005)
            bid_price = mid_price - spread/2
            ask_price = mid_price + spread/2
            bid_price = self.round_to_tick(bid_price)
            ask_price = self.round_to_tick(ask_price)

        # Post-only safety: nudge away from crossing if needed
        if getattr(self.config, 'POST_ONLY', False) and self.market_state.bid and self.market_state.ask:
            back = max(1, int(getattr(self.config, 'POST_ONLY_BACKOFF_TICKS', 1))) * (self.tick_size or 0)
            if bid_price >= self.market_state.ask:
                bid_price = min(self.market_state.bid - back, bid_price)
            if ask_price <= self.market_state.bid:
                ask_price = max(self.market_state.ask + back, ask_price)
            bid_price = self.round_to_tick(bid_price)
            ask_price = self.round_to_tick(ask_price)
        
        bid_size, ask_size = self.calculate_order_sizes()
        
        return Quotes(bid_price, ask_price, bid_size, ask_size, time.time())
    
    # Issue 2 Fix: Prices too far from market - problem in Avellaneda model
    # The reservation price calculation is going haywire

    def generate_avellaneda_quotes(self) -> Quotes:
        """Generate quotes using Avellaneda-Stoikov model - FIXED extreme pricing"""
        
        mid_price = self.market_state.mid_price
        
        fee_adjustment = mid_price * self.config.MAKER_FEE_RATE * 4

        volatility = max(self.market_state.volatility, 0.001)
        
        # Validate inputs
        if mid_price <= 0:
            self.logger.warning(f"Invalid mid_price: {mid_price}, falling back to volatility quotes")
            return self.generate_volatility_quotes()
        
        # Much more conservative Avellaneda-Stoikov parameters
        gamma = max(0.001, min(self.config.RISK_AVERSION, 0.05))    # REDUCED: Risk aversion capped at 1%
        k = max(0.0001, min(self.config.MARKET_IMPACT, 0.001))     # REDUCED: Market impact much smaller
        T = max(0.1, min(self.config.TIME_HORIZON, 0.5))           # REDUCED: Shorter time horizon
        
        # Current inventory position (normalized) - clamp to smaller range
        inventory_pos = self.get_inventory_position()
        q = max(-0.5, min(inventory_pos, 0.5))  # REDUCED: Limit inventory impact to ±50%
        
        # Much more conservative price adjustment
        volatility_squared = min(volatility ** 2, 0.0001)  # REDUCED: Cap vol² much lower
        price_adjustment = q * gamma * volatility_squared * T
        
        # CRITICAL: Limit price adjustment to tiny percentage
        max_adjustment = mid_price * 0.1  # REDUCED: Only 0.1% max adjustment instead of 10%
        price_adjustment = max(-max_adjustment, min(price_adjustment, max_adjustment))
        
        reservation_price = mid_price - price_adjustment
        
        # Validate reservation price is close to mid
        if abs(reservation_price - mid_price) > mid_price * 0.01:  # If more than 1% away
            self.logger.warning(f"Reservation price {reservation_price:.4f} too far from mid {mid_price:.4f}, using mid")
            reservation_price = mid_price
        
        # Much smaller base spread
        base_half_spread = max(
            self.config.BASE_SPREAD_TICKS * self.tick_size / 2,  # Minimum spread from config
            mid_price * 0.0001  # Or 0.01% of price, whichever is larger
        )
        
        # Simplified spread calculation - ignore complex impact adjustment for now
        half_spread = base_half_spread + fee_adjustment
        
        # Convert volatility component more conservatively
        if volatility > 0:
            vol_component = volatility * mid_price * 0.1  # REDUCED: Much smaller volatility impact
            half_spread = max(half_spread, vol_component)
        
        # Cap the spread at reasonable level
        max_half_spread = mid_price * 0.01  # Max 1% half spread
        half_spread = min(half_spread, max_half_spread)
        
        # Calculate bid and ask prices around reservation price
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread

        # ADJUST FOR FEES using dynamically fetched rates
        #bid_price = bid_price * (1 - self.config.MAKER_FEE_RATE)
        #ask_price = ask_price * (1 + self.config.MAKER_FEE_RATE)
        
        # Apply small imbalance skew
        imbalance_skew = self.get_imbalance_skew()
        max_skew = half_spread * 0.2  # Limit skew to 20% of spread
        imbalance_skew = max(-max_skew, min(imbalance_skew, max_skew))
        
        bid_price -= imbalance_skew
        ask_price += imbalance_skew
        
        # Ensure prices are reasonable - within 0.5% of mid price to reduce rejects
        min_bid = mid_price * 0.995
        max_ask = mid_price * 1.005
        
        if bid_price < min_bid:
            self.logger.warning(f"Bid too low: {bid_price:.4f}, adjusting to {min_bid:.4f}")
            bid_price = min_bid
        
        if ask_price > max_ask:
            self.logger.warning(f"Ask too high: {ask_price:.4f}, adjusting to {max_ask:.4f}")  
            ask_price = max_ask
        
        # Ensure bid < ask
        if bid_price >= ask_price:
            self.logger.warning("Bid >= Ask, adjusting spread")
            spread = max(self.tick_size * 2, mid_price * 0.0005)  # Minimum 2 ticks or 0.05%
            bid_price = mid_price - spread/2
            ask_price = mid_price + spread/2
        
        # Optional: peg quotes to top of book within constraints
        if getattr(self.config, 'PEG_TO_BOOK', False):
            offset = max(0, int(getattr(self.config, 'PEG_OFFSET_TICKS', 1))) * (self.tick_size or 0)
            if self.market_state.bid:
                peg_bid = self.market_state.bid - offset
                bid_price = min(bid_price, peg_bid)
            if self.market_state.ask:
                peg_ask = self.market_state.ask + offset
                ask_price = max(ask_price, peg_ask)

        # Round to tick size
        bid_price = self.round_to_tick(bid_price)
        ask_price = self.round_to_tick(ask_price)
        
        # Enforce minimum spread and sanity
        min_spread = max(getattr(self.config, 'MIN_SPREAD_TICKS', 2.0) * (self.tick_size or 0), mid_price * 0.0001)
        if ask_price - bid_price < min_spread:
            spread = max(min_spread, self.tick_size * 2, mid_price * 0.0005)
            bid_price = mid_price - spread/2
            ask_price = mid_price + spread/2
            bid_price = self.round_to_tick(bid_price)
            ask_price = self.round_to_tick(ask_price)

        # Post-only safety: nudge away from crossing if needed
        if getattr(self.config, 'POST_ONLY', False) and self.market_state.bid and self.market_state.ask:
            back = max(1, int(getattr(self.config, 'POST_ONLY_BACKOFF_TICKS', 1))) * (self.tick_size or 0)
            if bid_price >= self.market_state.ask:
                bid_price = min(self.market_state.bid - back, bid_price)
            if ask_price <= self.market_state.bid:
                ask_price = max(self.market_state.ask + back, ask_price)
            bid_price = self.round_to_tick(bid_price)
            ask_price = self.round_to_tick(ask_price)

        if bid_price <= 0 or ask_price <= 0 or ask_price <= bid_price:
            self.logger.error(f"Final price validation failed: bid={bid_price}, ask={ask_price}, falling back")
            return self.generate_volatility_quotes()
        
        # Final check - if prices are still too far from mid, use volatility method
        bid_diff_pct = abs(bid_price - mid_price) / mid_price
        ask_diff_pct = abs(ask_price - mid_price) / mid_price
        
        if bid_diff_pct > 0.02 or ask_diff_pct > 0.02:  # More than 2% away
            self.logger.warning(f"Avellaneda prices too far from mid (bid: {bid_diff_pct:.2%}, ask: {ask_diff_pct:.2%}), using volatility method")
            return self.generate_volatility_quotes()
        
        # Calculate order sizes
        bid_size, ask_size = self.calculate_order_sizes()
        
        self.logger.debug(f"Avellaneda quotes: bid={bid_price:.4f} ({bid_diff_pct:.3%} from mid), "
                        f"ask={ask_price:.4f} ({ask_diff_pct:.3%} from mid)")
        
        return Quotes(bid_price, ask_price, bid_size, ask_size, time.time())


    def get_inventory_position(self) -> float:
        """Get normalized inventory position (-1 to 1)"""
        max_inventory = self.config.max_inventory_usd / self.market_state.mid_price
        if max_inventory == 0:
            return 0.0
        return max(-1.0, min(1.0, self.position.quantity / max_inventory))
    
    def get_inventory_skew(self) -> float:
        """Calculate inventory-based price skew"""
        inventory_pos = self.get_inventory_position()
        return -inventory_pos * self.config.INVENTORY_SKEW_STRENGTH * self.tick_size
    
    def get_imbalance_skew(self) -> float:
        """Calculate order book imbalance-based price skew"""
        return self.market_state.imbalance * self.config.IMBALANCE_SKEW_STRENGTH * self.tick_size
    
    
    # FIXES FOR THE MARKET MAKER BOT

    # Issue 1 Fix: Order size calculation returning zero
    # The problem is in calculate_order_sizes() method - it's too restrictive with balance checks

    def calculate_order_sizes(self) -> Tuple[float, float]:
        """Calculate order sizes based on capital, risk, and available balance - FIXED"""
        if self.market_state.mid_price <= 0:
            self.logger.warning("Invalid mid price for size calculation")
            return 0.0, 0.0
        
        try:
            mid_price = float(self.market_state.mid_price)
            min_notional = float(self.min_notional)
            effective_min_qty = max(float(self.min_qty), float(getattr(self.config, 'MIN_QTY_OVERRIDE', 0.0) or 0.0))
            
            # Calculate base order size - make it more aggressive for small accounts
            total_capital = float(self.config.TOTAL_CAPITAL)
            
            # Use a higher percentage of capital per order for better liquidity provision
            base_notional = max(min_notional * 2.0, total_capital * 0.01)  # 1% of capital or 2x min notional
            base_size = base_notional / mid_price
            
            # Ensure minimum size requirements
            base_size = max(base_size, effective_min_qty)
            
            # Apply inventory factor but don't make it too restrictive
            max_inventory_usd = float(self.config.max_inventory_usd)
            max_inventory = max_inventory_usd / mid_price
            current_inventory = abs(float(self.position.quantity))
            
            # Less restrictive inventory factor
            inventory_factor = max(0.3, 1.0 - (current_inventory / max_inventory * 0.5)) if max_inventory > 0 else 0.8
            
            base_size = base_size * inventory_factor
            
            # Apply position skew but keep it reasonable
            inventory_pos = self.get_inventory_position()
            bid_size = base_size * max(0.5, 1.0 + float(inventory_pos) * 0.2)  # Don't go below 50% of base size
            ask_size = base_size * max(0.5, 1.0 - float(inventory_pos) * 0.2)
            
            # Futures sizing: both long and short consume USDT margin.
            # Allocate margin budget per side so both orders can coexist safely.
            lev = max(1, int(getattr(self.config, 'LEVERAGE', 1)))
            available_margin = max(0.0, float(self.balance.quote_free))
            total_notional_capacity = available_margin * lev
            per_side_fraction = 0.4  # 40% per side to avoid over-allocating if both fill

            max_side_notional = total_notional_capacity * per_side_fraction
            max_buy_size = (max_side_notional / mid_price) if mid_price > 0 else 0.0
            max_sell_size = (max_side_notional / mid_price) if mid_price > 0 else 0.0

            if max_buy_size > effective_min_qty:
                bid_size = min(bid_size, max_buy_size)
            elif max_buy_size > 0:
                bid_size = max_buy_size
            else:
                bid_size = 0.0

            if max_sell_size > effective_min_qty:
                ask_size = min(ask_size, max_sell_size)
            elif max_sell_size > 0:
                ask_size = max_sell_size
            else:
                ask_size = 0.0
            
            # Final minimum notional check - but don't zero out sizes
            bid_notional = bid_size * mid_price
            ask_notional = ask_size * mid_price
            
            if bid_size > 0 and bid_notional < min_notional:
                required_size = min_notional / mid_price
                if self.balance.quote_free >= min_notional * 1.1:
                    bid_size = required_size
                # Don't zero out - keep the size we calculated

            if ask_size > 0 and ask_notional < min_notional:
                required_size = min_notional / mid_price
                # Use margin for shorts as well
                if available_margin >= min_notional * 1.1:
                    ask_size = required_size
                # Don't zero out - keep the size we calculated

            # Enforce minimum quantity by lifting to min_qty if capacity allows; else zero to avoid rejects
            min_qty_notional = effective_min_qty * mid_price
            if bid_size > 0 and bid_size < effective_min_qty:
                if max_side_notional >= min_qty_notional:
                    bid_size = effective_min_qty
                else:
                    bid_size = 0.0
            if ask_size > 0 and ask_size < effective_min_qty:
                if max_side_notional >= min_qty_notional:
                    ask_size = effective_min_qty
                else:
                    ask_size = 0.0
            
            # Round to lot size
            bid_size = self.round_to_lot_size(bid_size) if bid_size > 0 else 0.0
            ask_size = self.round_to_lot_size(ask_size) if ask_size > 0 else 0.0
            
            # Ensure we have reasonable sizes
            bid_size = max(0.0, float(bid_size))
            ask_size = max(0.0, float(ask_size))
            
            self.logger.debug(f"Calculated sizes: bid={bid_size:.8f}, ask={ask_size:.8f}, "
                            f"bid_notional=${bid_size * mid_price:.2f}, ask_notional=${ask_size * mid_price:.2f}")
            
            return bid_size, ask_size
            
        except Exception as e:
            self.logger.error(f"Error calculating order sizes: {e}")
            return 0.0, 0.0
    
    def should_replace_orders(self, new_quotes: Quotes) -> bool:
        """Check if orders should be replaced"""
        if not self.current_quotes:
            return True
        
        bid_diff = abs(new_quotes.bid_price - self.current_quotes.bid_price)
        ask_diff = abs(new_quotes.ask_price - self.current_quotes.ask_price)
        
        min_price_diff = self.tick_size * 2
        
        if bid_diff >= min_price_diff or ask_diff >= min_price_diff:
            return True
        
        bid_size_diff = abs(new_quotes.bid_size - self.current_quotes.bid_size)
        ask_size_diff = abs(new_quotes.ask_size - self.current_quotes.ask_size)
        
        min_size_diff = self.min_qty * 2
        
        if bid_size_diff >= min_size_diff or ask_size_diff >= min_size_diff:
            return True
        
        if (new_quotes.can_place_bid != self.current_quotes.can_place_bid or 
            new_quotes.can_place_ask != self.current_quotes.can_place_ask):
            return True
        
        return False
    
    async def replace_orders(self, new_quotes: Quotes):
        """Replace current orders with new quotes"""
        await self.cancel_all_orders()
        await asyncio.sleep(0.1)
        
        orders_placed = []
        
        # FIXED: Only place order if size > 0 and we can place it
        if new_quotes.can_place_bid and new_quotes.bid_size > 0:
            bid_order = await self.client.place_order(
                self.config.SYMBOL, "buy", new_quotes.bid_size, new_quotes.bid_price, "limit"
            )
            
            if bid_order and bid_order.get('id'):
                self.open_orders[bid_order['id']] = {
                    'side': 'buy',
                    'price': new_quotes.bid_price,
                    'size': new_quotes.bid_size,
                    'timestamp': time.time(),
                    'symbol': self.config.SYMBOL
                }
                orders_placed.append(f"Bid: {new_quotes.bid_price:.4f} @ {new_quotes.bid_size:.6f}")
        
        if new_quotes.can_place_ask and new_quotes.ask_size > 0:
            ask_order = await self.client.place_order(
                self.config.SYMBOL, "sell", new_quotes.ask_size, new_quotes.ask_price, "limit"
            )
            
            if ask_order and ask_order.get('id'):
                self.open_orders[ask_order['id']] = {
                    'side': 'sell',
                    'price': new_quotes.ask_price,
                    'size': new_quotes.ask_size,
                    'timestamp': time.time(),
                    'symbol': self.config.SYMBOL
                }
                orders_placed.append(f"Ask: {new_quotes.ask_price:.4f} @ {new_quotes.ask_size:.6f}")
        
        self.current_quotes = new_quotes
        
        if orders_placed:
            self.logger.info(f"Orders placed - {', '.join(orders_placed)}")
        else:
            self.logger.debug("No orders placed due to balance constraints or zero sizes")
    
    async def cancel_all_orders(self):
        """Cancel all open orders using CCXT"""
        open_orders = await self.client.get_open_orders(self.config.SYMBOL)
        
        for order in open_orders:
            try:
                await self.client.cancel_order(order['id'], self.config.SYMBOL)
                if order['id'] in self.open_orders:
                    del self.open_orders[order['id']]
            except Exception as e:
                self.logger.error(f"Failed to cancel order {order['id']}: {e}")
        
        self.open_orders.clear()
    
    def round_to_tick(self, price: float) -> float:
        """Round price to exchange tick size"""
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size
    
    def round_to_lot_size(self, quantity: float) -> float:
        """Round quantity to minimum quantity increment"""
        if self.lot_size <= 0:
            return quantity
        return round(quantity / self.lot_size) * self.lot_size
    
    async def trading_loop(self):
        """Main trading loop with CCXT fallback"""
        self.logger.info("Starting main trading loop")
        
        while True:
            try:
                if not self.check_risk_limits():
                    if self.kill_switch_active:
                        self.logger.error("Kill switch activated - stopping trading")
                        await self.cancel_all_orders()
                        break
                
                if not self.ws_manager or not self.ws_manager.running:
                    await self.update_market_data_rest()
                
                await self.update_balance()
                await self.check_order_status()
                # Maintain bracket TP/SL orders if enabled
                await self.sync_bracket_orders()
                await self.print_status()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5.0)
    
    async def update_market_data_rest(self):
        """Update market data using REST API as fallback"""
        try:
            order_book = await self.client.get_order_book(self.config.SYMBOL)
            if order_book:
                await self.update_market_state_from_ccxt(order_book)
        except Exception as e:
            self.logger.error(f"Error updating market data via REST: {e}")
    
    async def check_order_status(self):
        """Check status of our open orders"""
        try:
            open_orders = await self.client.get_open_orders(self.config.SYMBOL)
            open_order_ids = {order['id'] for order in open_orders}
            
            for order_id in list(self.open_orders.keys()):
                if order_id not in open_order_ids:
                    self.logger.info(f"Order {order_id} no longer open (likely filled)")
                    del self.open_orders[order_id]
                    self.last_quote_time = time.time() - self.config.MAX_ORDER_REPLACE_FREQ  ##trigger new quote

            for order in open_orders:
                if order['id'] not in self.open_orders:
                    self.open_orders[order['id']] = {
                        'side': order['side'],
                        'price': order['price'],
                        'size': order['amount'],
                        'timestamp': order['timestamp'] / 1000 if order['timestamp'] else time.time(),
                        'symbol': order['symbol'],
                        'info': order.get('info', {})
                    }
        except Exception as e:
            self.logger.error(f"Error checking order status: {e}")
            
    async def sync_bracket_orders(self):
        """Ensure TP/SL orders are aligned with current position and config."""
        try:
            if not getattr(self.config, 'ENABLE_TP_SL', False):
                return
            qty = float(self.position.quantity)
            avg = float(self.position.avg_price)
            if qty == 0 or avg <= 0:
                # No position: cancel existing TP/SL if any
                for key in ['tp', 'sl']:
                    oid = self.bracket_orders.get(key)
                    if oid:
                        try:
                            await self.client.cancel_order(oid, self.config.SYMBOL)
                        except Exception:
                            pass
                        self.bracket_orders[key] = None
                return
            # Determine sides and targets
            is_long = qty > 0
            abs_qty = abs(qty)
            tp_pct = max(0.0, float(self.config.TP_PCT))
            sl_pct = max(0.0, float(self.config.SL_PCT))
            # compute prices
            if is_long:
                desired_tp = self.round_to_tick(avg * (1.0 + tp_pct))
                desired_sl = self.round_to_tick(avg * (1.0 - sl_pct))
                tp_side = 'sell'
                sl_side = 'sell'
            else:
                desired_tp = self.round_to_tick(avg * (1.0 - tp_pct))
                desired_sl = self.round_to_tick(avg * (1.0 + sl_pct))
                tp_side = 'buy'
                sl_side = 'buy'

            # Place/refresh TP (reduce-only limit)
            existing_tp_id = self.bracket_orders.get('tp')
            need_new_tp = True
            if existing_tp_id and existing_tp_id in self.open_orders:
                existing = self.open_orders[existing_tp_id]
                if abs(float(existing.get('price', 0.0)) - desired_tp) <= (self.tick_size or 0):
                    need_new_tp = False
            if need_new_tp:
                # Cancel previous
                if existing_tp_id:
                    try:
                        await self.client.cancel_order(existing_tp_id, self.config.SYMBOL)
                    except Exception:
                        pass
                    self.bracket_orders['tp'] = None
                # Place new
                if getattr(self.config, 'TP_REDUCE_ONLY', True):
                    tp_order = await self.client.place_reduce_only_limit(self.config.SYMBOL, tp_side, abs_qty, desired_tp)
                else:
                    tp_order = await self.client.place_order(self.config.SYMBOL, tp_side, abs_qty, desired_tp, 'limit')
                if tp_order and tp_order.get('id'):
                    self.bracket_orders['tp'] = tp_order['id']

            # Place/refresh SL (stop-market close)
            existing_sl_id = self.bracket_orders.get('sl')
            need_new_sl = True
            if existing_sl_id and existing_sl_id in self.open_orders:
                existing = self.open_orders[existing_sl_id]
                # stopPrice may be in info
                info = existing.get('info', {}) or {}
                existing_stop = float(info.get('stopPrice') or info.get('stopprice') or 0.0)
                if existing_stop and abs(existing_stop - desired_sl) <= (self.tick_size or 0):
                    need_new_sl = False
            if need_new_sl:
                if existing_sl_id:
                    try:
                        await self.client.cancel_order(existing_sl_id, self.config.SYMBOL)
                    except Exception:
                        pass
                    self.bracket_orders['sl'] = None
                if getattr(self.config, 'SL_REDUCE_ONLY', True):
                    sl_order = await self.client.place_stop_market_close(self.config.SYMBOL, sl_side, desired_sl, abs_qty)
                else:
                    # Fallback: place a regular stop-market without reduceOnly (not recommended)
                    sl_order = await self.client.place_stop_market_close(self.config.SYMBOL, sl_side, desired_sl, abs_qty)
                if sl_order and sl_order.get('id'):
                    self.bracket_orders['sl'] = sl_order['id']
        except Exception as e:
            self.logger.error(f"Error syncing TP/SL: {e}")
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        inventory_usd = abs(self.position.quantity) * self.market_state.mid_price
        if inventory_usd > self.config.max_inventory_usd:
            self.logger.warning(f"Inventory limit breached: ${inventory_usd:.2f}")
            return False
        
        # FIXED: Use realized P&L in drawdown calculation
        current_balance = self.starting_balance + self.position.realized_pnl + self.position.unrealized_pnl
        drawdown = self.starting_balance - current_balance
        
        if drawdown > self.config.max_drawdown_usd:
            self.logger.error(f"Max drawdown breached: ${drawdown:.2f}")
            self.kill_switch_active = True
            return False
        
        self.max_drawdown = max(self.max_drawdown, drawdown)
        return True
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        self.logger.info("Shutting down market maker bot...")
        
        await self.cancel_all_orders()
        
        if self.ws_manager:
            self.ws_manager.running = False
        
        await self.client.close()
        
        self.logger.info("Bot shutdown complete")
    
    async def print_status(self):
        """Print enhanced terminal dashboard like a pro bot"""
        def fmt_usd(v: float) -> str:
            try:
                return f"${v:,.2f}"
            except Exception:
                return str(v)

        def fmt_qty(v: float) -> str:
            try:
                return f"{v:,.6f}"
            except Exception:
                return str(v)

        def fmt_pct(v: float) -> str:
            try:
                return f"{v:+.2f}%"
            except Exception:
                return str(v)

        def fmt_age(ts: float) -> str:
            if not ts:
                return "-"
            s = max(0, int(time.time() - ts))
            if s < 60:
                return f"{s}s"
            m, s = divmod(s, 60)
            if m < 60:
                return f"{m}m{s:02d}s"
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Totals
        total_pnl = float(self.position.realized_pnl) + float(self.position.unrealized_pnl)
        pnl_pct = (total_pnl / self.starting_balance * 100.0) if self.starting_balance > 0 else 0.0
        spread_ticks = (self.market_state.spread / self.tick_size) if self.tick_size > 0 else 0.0
        ws_status = 'WS' if (self.ws_manager and self.ws_manager.running) else 'REST'
        mode = 'TESTNET' if self.config.USE_TESTNET else 'LIVE'

        # Last trade age
        last_trade_age = "-"
        if self.trade_history:
            last_trade_age = fmt_age(self.trade_history[-1].timestamp)

        # Header
        print("\n" + "═" * 92)
        print(f"{now_str} | {mode} | {ws_status} | {self.config.SYMBOL}")
        print("─" * 92)

        # Market
        print(
            f"Bid {self.market_state.bid:>12.4f}  Mid {self.market_state.mid_price:>12.4f}  "
            f"Ask {self.market_state.ask:>12.4f}  Spread {self.market_state.spread:>8.4f} ({spread_ticks:.1f} ticks)"
        )
        print(
            f"Regime {self.market_state.regime:>6}  Vol {self.market_state.volatility:>7.5f}  "
            f"Imbalance {self.market_state.imbalance:>+6.3f}  Tick {self.tick_size:g}  Lot {self.lot_size:g}  "
            f"MinQty {max(self.min_qty, getattr(self.config,'MIN_QTY_OVERRIDE',0.0) or 0.0):g}  MinNotional {fmt_usd(self.min_notional)}"
        )

        # Futures + Fees
        print(
            f"Leverage x{int(getattr(self.config,'LEVERAGE',1))}  Margin {str(getattr(self.config,'MARGIN_MODE','cross')).upper()}  "
            f"Hedge {bool(getattr(self.config,'HEDGE_MODE',False))}  PostOnly {bool(getattr(self.config,'POST_ONLY',False))}  "
            f"Fees M {self.config.MAKER_FEE_RATE*100:.3f}% / T {self.config.TAKER_FEE_RATE*100:.3f}%"
        )

        # Position and Wallet
        print(
            f"Pos {fmt_qty(self.position.quantity)} @ {self.position.avg_price:,.2f}  "
            f"Notional {fmt_usd(self.position.notional_value)}  UPL {fmt_usd(self.position.unrealized_pnl)}  RPL {fmt_usd(self.position.realized_pnl)}"
        )
        # Futures wallet is quote asset
        capacity_notional = float(self.balance.quote_free) * float(getattr(self.config,'LEVERAGE',1))
        print(
            f"Wallet {self.config.QUOTE_ASSET}: Free {fmt_usd(self.balance.quote_free)}  Total {fmt_usd(self.balance.quote_total)}  "
            f"Capacity Notional ~ {fmt_usd(capacity_notional)}"
        )

        # Risk
        inventory_usd = abs(self.position.quantity) * (self.market_state.mid_price or 0)
        inv_limit = self.config.max_inventory_usd
        drawdown = self.starting_balance - (self.starting_balance + self.position.realized_pnl + self.position.unrealized_pnl)
        print(
            f"Risk Inv {fmt_usd(inventory_usd)} / {fmt_usd(inv_limit)}  "
            f"Drawdown {fmt_usd(drawdown)} / {fmt_usd(self.config.max_drawdown_usd)}  KillSwitch {self.kill_switch_active}"
        )

        # Trades and timing
        print(
            f"Trades {self.total_trades}  Fills B:{self.bid_fills} / A:{self.ask_fills}  "
            f"Last Trade {last_trade_age}  Last Quote {fmt_age(self.last_quote_time)}  ReplaceFreq {int(self.config.MAX_ORDER_REPLACE_FREQ)}s"
        )

        # Quotes
        if self.current_quotes:
            bid_ok = 'OK' if self.current_quotes.can_place_bid and self.current_quotes.bid_size > 0 else 'BLOCK'
            ask_ok = 'OK' if self.current_quotes.can_place_ask and self.current_quotes.ask_size > 0 else 'BLOCK'
            print("─" * 92)
            print(
                f"Quote  Bid {self.current_quotes.bid_price:>12.4f} x {fmt_qty(self.current_quotes.bid_size):>12}  [{bid_ok}]    "
                f"Ask {self.current_quotes.ask_price:>12.4f} x {fmt_qty(self.current_quotes.ask_size):>12}  [{ask_ok}]"
            )

        # Last order diagnostics
        try:
            last_err = getattr(self.client, 'last_order_error', None)
            last_det = getattr(self.client, 'last_order_details', None)
            if last_err or last_det:
                print("─" * 92)
                if last_err:
                    print(f"Last Order Error: {last_err}")
                if last_det:
                    side = last_det.get('side') if isinstance(last_det, dict) else None
                    amt = last_det.get('amount') if isinstance(last_det, dict) else None
                    prc = last_det.get('price') if isinstance(last_det, dict) else None
                    oid = last_det.get('id') if isinstance(last_det, dict) else None
                    print(f"Last Attempt: {side} {fmt_qty(amt) if amt is not None else amt} @ {prc}  id={oid}")
        except Exception:
            pass

        # Open orders table
        if self.open_orders:
            print("─" * 92)
            print(f"Open Orders: {len(self.open_orders)}")
            print(f"{'ID':<20} {'Side':<4} {'Price':>12} {'Size':>14} {'Age':>8}")
            for oid, od in list(self.open_orders.items())[:10]:  # show up to 10
                side = od.get('side','?')
                price = od.get('price', 0.0)
                size = od.get('size', 0.0)
                ts = od.get('timestamp', time.time())
                print(f"{str(oid)[-20:]:<20} {side:<4} {price:>12.4f} {fmt_qty(size):>14} {fmt_age(ts):>8}")

        # P&L summary footer
        print("─" * 92)
        print(f"P&L  Total {fmt_usd(total_pnl)}  ({fmt_pct(pnl_pct)})  MaxDD {fmt_usd(self.max_drawdown)}")
        print("═" * 92)
        print(f"Volatility: {self.market_state.volatility*100:6.2f}% | "
              f"Imbalance: {self.market_state.imbalance:+6.2f} | "
              f"Open Orders: {len(self.open_orders)}")
        
        if self.current_quotes:
            print(f"Current Quotes:")
            bid_status = "✅" if self.current_quotes.can_place_bid and self.current_quotes.bid_size > 0 else "❌"
            ask_status = "✅" if self.current_quotes.can_place_ask and self.current_quotes.ask_size > 0 else "❌"
            print(f"  Bid: {self.current_quotes.bid_price:8.4f} @ {self.current_quotes.bid_size:8.6f} {bid_status}")
            print(f"  Ask: {self.current_quotes.ask_price:8.4f} @ {self.current_quotes.ask_size:8.6f} {ask_status}")
        
        print(f"Trades: {self.total_trades} | Volume: {self.total_volume:.6f} | "
              f"VWAP Bid: {self.vwap_bid:.4f} | VWAP Ask: {self.vwap_ask:.4f}")
        
        market_status = "Loaded" if self.market_info else "Not loaded"
        print(f"Market Info: {market_status} | Tick: {self.tick_size} | Min Notional: ${self.min_notional}")
        
        ws_status = "Connected" if self.ws_manager and self.ws_manager.running else "Disconnected (REST fallback)"
        print(f"WebSocket: {ws_status}")
        
        can_buy = self.can_place_buy_order(self.min_qty, self.market_state.mid_price)
        can_sell = self.can_place_sell_order(self.min_qty)
        print(f"Can Place: Buy {'✅' if can_buy else '❌'} | Sell {'✅' if can_sell else '❌'}")
        
        if self.kill_switch_active:
            print("🛑 KILL SWITCH ACTIVE - TRADING STOPPED")

# ============================================================================
# MAIN EXECUTION WITH CONFIG FILE SUPPORT
# ============================================================================

async def main():
    """Main execution function with external config file support"""
    
    print("Binance Market Maker Bot (CCXT Edition) - ENHANCED VERSION")
    print("="*65)
    print()
    print("🔧 NEW FEATURES:")
    print("✅ External configuration file support")
    print("✅ Realized P&L tracking and calculation")
    print("✅ Fixed zero order size issues")
    print("✅ Improved balance-aware order placement")
    print("✅ FIFO-based P&L calculation")
    print("✅ Real-time trade detection and P&L updates")
    print()
    
    # FIXED: Load configuration from external file
    config = Config.from_file("market_maker_config_future.json")
    
    if not config:
        print("❌ Failed to load configuration file")
        print("📝 Please check market_maker_config_future.json and set your API keys")
        return
    
    # Validate API keys
    if not config.API_KEY or config.API_KEY == "your_binance_api_key_here":
        print("❌ ERROR: Please set your actual Binance API keys in market_maker_config_future.json")
        print("🔗 Get your API keys from:")
        print("   - Testnet: https://testnet.binance.vision/")
        print("   - Live: https://binance.com → API Management")
        return
    
    print(f"✅ Configuration loaded from market_maker_config_future.json")
    print(f"🚀 Starting bot with {'TESTNET' if config.USE_TESTNET else 'LIVE TRADING'}")
    print(f"💰 Capital: ${config.TOTAL_CAPITAL:,.2f}")
    print(f"📊 Symbol: {config.SYMBOL}")
    print(f"📈 Max Inventory: {config.MAX_INVENTORY_RATIO*100:.1f}%")
    print(f"⚠️  Max Drawdown: {config.MAX_DRAWDOWN_RATIO*100:.1f}%")
    
    if not config.USE_TESTNET:
        print("\n" + "="*50)
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("This will use real money on Binance!")
        print("="*50)
        confirmation = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirmation != "CONFIRM":
            print("❌ Live trading cancelled")
            return
    
    # Create and start bot
    bot = MarketMakerBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down bot...")
        await bot.shutdown()
        print("✅ Bot stopped successfully")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()
        await bot.shutdown()

if __name__ == "__main__":
    # Installation check
    try:
        import ccxt
        print(f"✅ CCXT v{ccxt.__version__} detected")
        try:
            import ccxt.pro as ccxtpro
            print("✅ CCXT Pro detected - WebSocket support available")
        except ImportError:
            print("ℹ️  CCXT Pro not available - using REST API only")
            print("   Install with: pip install ccxt[pro]")
    except ImportError:
        print("❌ CCXT not found!")
        print("📦 Install CCXT with: pip install ccxt")
        print("📦 For WebSocket support: pip install ccxt[pro]")
        exit(1)
    
    # Check if config file exists, create if not
    if not os.path.exists("market_maker_config_future.json"):
        print("📝 Creating default configuration file...")
        ConfigFile.create_default_config()
        print("✅ Created market_maker_config_future.json")
        print()
        print("⚠️  IMPORTANT: Please edit market_maker_config_future.json with your settings:")
        print("   1. Set your Binance API key and secret")
        print("   2. Configure your trading parameters")
        print("   3. Set testnet to false for live trading")
        print("   4. Adjust capital and risk parameters")
        print()
        print("📋 Example configuration structure:")
        print("""
{
    "exchange": {
        "api_key": "your_actual_api_key_here",
        "api_secret": "your_actual_api_secret_here",
        "testnet": true,
        "symbol": "BTC/USDT",
        "base_asset": "BTC",
        "quote_asset": "USDT"
    },
    "trading": {
        "total_capital": 329.0,
        "max_inventory_ratio": 0.2,
        "max_drawdown_ratio": 0.03,
        "base_spread_ticks": 3.0,
        "min_base_balance": 0.001,
        "min_quote_balance": 5.0
    }
}
        """)
        exit(0)
    
    # Run the bot
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
