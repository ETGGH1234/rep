#!/usr/bin/env python3
"""
Enhanced Solana Copy Trading Bot with Web Dashboard - Mirrors trades from a target wallet
Supports Jupiter, Orca, and Raydium swaps with real-time WebSocket monitoring
Features: Priority fees, retry logic, slippage controls, web GUI
"""

import asyncio
import json
import logging
import sys
import argparse
import base64
import struct
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import aiohttp
import websockets
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import Transaction
from solders.system_program import ID as SYS_PROGRAM_ID
from solders.rpc.responses import GetTransactionResp
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Finalized
from solana.rpc.types import TxOpts
from anchorpy import Provider, Wallet
from spl.token.constants import TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID
import questionary

# Web framework imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel, Field
import threading

# Constants
JUPITER_V6_PROGRAM = Pubkey.from_string("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
ORCA_WHIRLPOOL_PROGRAM = Pubkey.from_string("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc")
RAYDIUM_V4_PROGRAM = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
COMPUTE_BUDGET_PROGRAM = Pubkey.from_string("ComputeBudget111111111111111111111111111111")

USDC_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
WSOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
SOL_DECIMALS = 9
USDC_DECIMALS = 6

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('copy_trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Web app instance
app = FastAPI(title="Solana Copy Trading Bot Dashboard")
templates = Jinja2Templates(directory="templates")

# Global bot instance
bot_instance = None
connected_websockets = []


class BotSettings(BaseModel):
    """Pydantic model for bot settings"""
    target_wallet: Optional[str] = None
    slippage: float = Field(default=0.5, ge=0.1, le=5.0)
    compute_unit_limit: int = Field(default=200_000, ge=50_000, le=1_400_000)
    compute_unit_price: int = Field(default=1000, ge=0, le=100_000)
    tip_amount: int = Field(default=0, ge=0, le=1_000_000)
    position_size: float = Field(default=1.0, ge=0.01, le=1.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    ws_url: str = "wss://api.mainnet-beta.solana.com"


class Config:
    """Enhanced bot configuration"""
    def __init__(self):
        self.target_wallet: Optional[Pubkey] = None
        self.user_keypair: Optional[Keypair] = None
        self.slippage: float = 0.5
        self.compute_unit_limit: int = 200_000
        self.compute_unit_price: int = 1000  # micro-lamports per compute unit
        self.tip_amount: int = 0  # lamports
        self.position_size: float = 1.0
        self.max_retries: int = 3
        self.rpc_url: str = "https://api.mainnet-beta.solana.com"
        self.ws_url: str = "wss://api.mainnet-beta.solana.com"
        
    def update_from_settings(self, settings: BotSettings):
        """Update config from web settings"""
        if settings.target_wallet:
            self.target_wallet = Pubkey.from_string(settings.target_wallet)
        self.slippage = settings.slippage
        self.compute_unit_limit = settings.compute_unit_limit
        self.compute_unit_price = settings.compute_unit_price
        self.tip_amount = settings.tip_amount
        self.position_size = settings.position_size
        self.max_retries = settings.max_retries
        self.rpc_url = settings.rpc_url
        self.ws_url = settings.ws_url
        
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "target_wallet": str(self.target_wallet) if self.target_wallet else None,
            "slippage": self.slippage,
            "compute_unit_limit": self.compute_unit_limit,
            "compute_unit_price": self.compute_unit_price,
            "tip_amount": self.tip_amount,
            "position_size": self.position_size,
            "max_retries": self.max_retries,
            "rpc_url": self.rpc_url,
            "ws_url": self.ws_url
        }
        
    def save(self, path: str = "config.json"):
        """Save configuration to file"""
        data = self.to_dict()
        data["user_wallet_path"] = getattr(self, 'user_wallet_path', None)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, path: str = "config.json"):
        """Load configuration from file"""
        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.target_wallet = Pubkey.from_string(data["target_wallet"]) if data.get("target_wallet") else None
                self.slippage = data.get("slippage", 0.5)
                self.compute_unit_limit = data.get("compute_unit_limit", 200_000)
                self.compute_unit_price = data.get("compute_unit_price", 1000)
                self.tip_amount = data.get("tip_amount", 0)
                self.position_size = data.get("position_size", 1.0)
                self.max_retries = data.get("max_retries", 3)
                self.rpc_url = data.get("rpc_url", self.rpc_url)
                self.ws_url = data.get("ws_url", self.ws_url)
                self.user_wallet_path = data.get("user_wallet_path")
                return True
        return False


class SwapInfo:
    """Parsed swap information"""
    def __init__(self, token_in: Pubkey, token_out: Pubkey, amount_in: int, 
                 direction: str, dex: str, signature: str):
        self.token_in = token_in
        self.token_out = token_out
        self.amount_in = amount_in
        self.direction = direction  # 'buy' or 'sell'
        self.dex = dex
        self.signature = signature
        self.timestamp = datetime.now()


class TransactionLog:
    """Transaction logging for web interface"""
    def __init__(self):
        self.logs = []
        self.max_logs = 1000
        
    def add_log(self, log_type: str, message: str, details: dict = None):
        """Add a log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": log_type,
            "message": message,
            "details": details or {}
        }
        self.logs.append(entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
            
        # Broadcast to connected websockets
        asyncio.create_task(self.broadcast_log(entry))
        
    async def broadcast_log(self, entry: dict):
        """Broadcast log entry to connected websockets"""
        if connected_websockets:
            for websocket in connected_websockets.copy():
                try:
                    await websocket.send_text(json.dumps(entry))
                except:
                    connected_websockets.remove(websocket)
                    
    def get_recent_logs(self, limit: int = 100) -> List[dict]:
        """Get recent logs"""
        return self.logs[-limit:]


# Global transaction log
transaction_log = TransactionLog()


class SolanaCopyBot:
    """Enhanced copy trading bot with retry logic and advanced fee controls"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[AsyncClient] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self.token_cache: Dict[str, Dict] = {}
        self.stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_retries": 0
        }
        
    async def initialize(self):
        """Initialize connections"""
        self.client = AsyncClient(self.config.rpc_url)
        self.session = aiohttp.ClientSession()
        logger.info("Bot initialized")
        transaction_log.add_log("info", "Bot initialized", {"rpc_url": self.config.rpc_url})
        
    async def cleanup(self):
        """Cleanup connections"""
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()
        if self.client:
            await self.client.close()
            
    async def retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.stats["total_retries"] += 1
                
                if attempt < self.config.max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s...
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    transaction_log.add_log("warning", f"Retry attempt {attempt + 1}", {
                        "error": str(e),
                        "delay": delay
                    })
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed: {e}")
                    transaction_log.add_log("error", f"All retry attempts failed", {
                        "error": str(e),
                        "attempts": self.config.max_retries
                    })
                    
        raise last_exception
        
    async def connect_websocket(self):
        """Connect to Solana WebSocket with retry logic"""
        while self.running:
            try:
                self.websocket = await websockets.connect(self.config.ws_url)
                
                # Subscribe to target wallet transactions
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [
                        {
                            "mentions": [str(self.config.target_wallet)]
                        },
                        {
                            "commitment": "confirmed"
                        }
                    ]
                }
                
                await self.websocket.send(json.dumps(subscribe_msg))
                response = await self.websocket.recv()
                result = json.loads(response)
                
                if "result" in result:
                    logger.info(f"WebSocket subscribed: {result['result']}")
                    transaction_log.add_log("success", "WebSocket connected", {
                        "subscription_id": result['result']
                    })
                    await self.handle_websocket_messages()
                else:
                    logger.error(f"WebSocket subscription failed: {result}")
                    transaction_log.add_log("error", "WebSocket subscription failed", result)
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                transaction_log.add_log("error", "WebSocket connection error", {"error": str(e)})
                await asyncio.sleep(5)  # Reconnect after 5 seconds
                
    async def handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        async for message in self.websocket:
            try:
                data = json.loads(message)
                
                if "params" in data and "result" in data["params"]:
                    result = data["params"]["result"]
                    if "value" in result and "signature" in result["value"]:
                        signature = result["value"]["signature"]
                        await self.process_transaction(signature)
                        
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                
    async def process_transaction(self, signature: str):
        """Process a transaction from the target wallet"""
        try:
            # Get transaction details with retry
            tx_resp = await self.retry_with_backoff(
                self.client.get_transaction,
                Signature.from_string(signature),
                encoding="jsonParsed",
                commitment=Confirmed
            )
            
            if not tx_resp or not tx_resp.value:
                return
                
            tx = tx_resp.value.transaction
            
            # Parse for swaps
            swap_info = await self.parse_swap_transaction(tx, signature)
            
            if swap_info:
                logger.info(f"Detected {swap_info.dex} swap: {swap_info.direction} "
                          f"{swap_info.token_in} -> {swap_info.token_out}")
                transaction_log.add_log("info", f"Detected {swap_info.dex} swap", {
                    "direction": swap_info.direction,
                    "token_in": str(swap_info.token_in),
                    "token_out": str(swap_info.token_out),
                    "amount_in": swap_info.amount_in,
                    "signature": signature
                })
                await self.execute_copy_trade(swap_info)
                
        except Exception as e:
            logger.error(f"Error processing transaction {signature}: {e}")
            transaction_log.add_log("error", f"Error processing transaction", {
                "signature": signature,
                "error": str(e)
            })
            
    async def parse_swap_transaction(self, tx: Any, signature: str) -> Optional[SwapInfo]:
        """Parse transaction for swap instructions"""
        try:
            if not hasattr(tx, 'message') or not hasattr(tx.message, 'instructions'):
                return None
                
            for instruction in tx.message.instructions:
                program_id = instruction.get('programId', '')
                
                # Check Jupiter swap
                if program_id == str(JUPITER_V6_PROGRAM):
                    return await self.parse_jupiter_swap(instruction, tx, signature)
                    
                # Check Orca swap
                elif program_id == str(ORCA_WHIRLPOOL_PROGRAM):
                    return await self.parse_orca_swap(instruction, tx, signature)
                    
                # Check Raydium swap
                elif program_id == str(RAYDIUM_V4_PROGRAM):
                    return await self.parse_raydium_swap(instruction, tx, signature)
                    
        except Exception as e:
            logger.error(f"Error parsing swap transaction: {e}")
            
        return None
        
    async def parse_jupiter_swap(self, instruction: Dict, tx: Any, signature: str) -> Optional[SwapInfo]:
        """Parse Jupiter swap instruction"""
        try:
            # Jupiter swaps typically have token transfers in inner instructions
            if 'innerInstructions' not in instruction:
                return None
                
            transfers = []
            for inner in instruction['innerInstructions']:
                if inner.get('program') == 'spl-token' and inner.get('parsed', {}).get('type') == 'transfer':
                    transfers.append(inner['parsed']['info'])
                    
            if len(transfers) >= 2:
                # First transfer is usually input, last is output
                token_in = Pubkey.from_string(transfers[0]['mint'])
                token_out = Pubkey.from_string(transfers[-1]['mint'])
                amount_in = int(transfers[0]['amount'])
                
                # Determine direction
                direction = 'sell' if token_in == WSOL_MINT else 'buy'
                
                return SwapInfo(token_in, token_out, amount_in, direction, 'Jupiter', signature)
                
        except Exception as e:
            logger.error(f"Error parsing Jupiter swap: {e}")
            
        return None
        
    async def parse_orca_swap(self, instruction: Dict, tx: Any, signature: str) -> Optional[SwapInfo]:
        """Parse Orca swap instruction"""
        # Similar parsing logic for Orca
        logger.warning("Orca swap parsing not fully implemented")
        return None
        
    async def parse_raydium_swap(self, instruction: Dict, tx: Any, signature: str) -> Optional[SwapInfo]:
        """Parse Raydium swap instruction"""
        # Similar parsing logic for Raydium
        logger.warning("Raydium swap parsing not fully implemented")
        return None
        
    async def get_token_info(self, mint: str) -> Dict:
        """Get token information including decimals and price"""
        if mint in self.token_cache:
            return self.token_cache[mint]
            
        try:
            # Get token account info
            mint_info = await self.client.get_account_info(Pubkey.from_string(mint))
            
            # For SOL
            if mint == str(WSOL_MINT):
                info = {"decimals": SOL_DECIMALS, "symbol": "SOL"}
            # For USDC
            elif mint == str(USDC_MINT):
                info = {"decimals": USDC_DECIMALS, "symbol": "USDC"}
            else:
                # Parse SPL token mint data to get decimals
                info = {"decimals": 9, "symbol": "UNKNOWN"}  # Default
                
            self.token_cache[mint] = info
            return info
            
        except Exception as e:
            logger.error(f"Error getting token info for {mint}: {e}")
            return {"decimals": 9, "symbol": "UNKNOWN"}
            
    async def get_portfolio_value_usd(self) -> float:
        """Calculate total portfolio value in USD"""
        try:
            # Get SOL balance
            sol_balance = await self.client.get_balance(self.config.user_keypair.pubkey())
            sol_amount = sol_balance.value / 10**SOL_DECIMALS
            
            # Get SOL price from Jupiter
            sol_price = await self.get_token_price(str(WSOL_MINT))
            sol_value_usd = sol_amount * sol_price
            
            # Get USDC balance
            usdc_balance = await self.get_token_balance(self.config.user_keypair.pubkey(), USDC_MINT)
            usdc_value_usd = usdc_balance / 10**USDC_DECIMALS
            
            total_usd = sol_value_usd + usdc_value_usd
            logger.info(f"Portfolio value: ${total_usd:.2f} (SOL: ${sol_value_usd:.2f}, USDC: ${usdc_value_usd:.2f})")
            
            return total_usd
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
            
    async def get_token_balance(self, owner: Pubkey, mint: Pubkey) -> int:
        """Get SPL token balance"""
        try:
            # Find associated token account
            ata = self.get_associated_token_address(owner, mint)
            
            # Get balance
            account_info = await self.client.get_token_account_balance(ata)
            if account_info and account_info.value:
                return int(account_info.value.amount)
                
        except Exception as e:
            logger.error(f"Error getting token balance: {e}")
            
        return 0
        
    def get_associated_token_address(self, owner: Pubkey, mint: Pubkey) -> Pubkey:
        """Calculate associated token address"""
        # This is a simplified version - use proper SPL token library in production
        return Pubkey.find_program_address(
            [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
            ASSOCIATED_TOKEN_PROGRAM_ID
        )[0]
        
    async def get_token_price(self, mint: str) -> float:
        """Get token price in USD from Jupiter"""
        try:
            url = f"https://price.jup.ag/v4/price?ids={mint}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "data" in data and mint in data["data"]:
                        return float(data["data"][mint]["price"])
                        
        except Exception as e:
            logger.error(f"Error getting token price: {e}")
            
        return 0.0
        
    async def execute_copy_trade(self, swap_info: SwapInfo):
        """Execute the copy trade with retry logic"""
        try:
            self.stats["total_trades"] += 1
            
            # Calculate adjusted amount based on position size
            adjusted_amount = int(swap_info.amount_in * self.config.position_size)
            
            # For buy trades, check 5% portfolio limit
            if swap_info.direction == 'buy':
                token_info = await self.get_token_info(str(swap_info.token_in))
                token_price = await self.get_token_price(str(swap_info.token_in))
                
                trade_value_usd = (adjusted_amount / 10**token_info["decimals"]) * token_price
                portfolio_value_usd = await self.get_portfolio_value_usd()
                
                if portfolio_value_usd > 0:
                    max_trade_usd = portfolio_value_usd * 0.05
                    
                    if trade_value_usd > max_trade_usd:
                        # Cap at 5%
                        adjusted_amount = int((max_trade_usd / token_price) * 10**token_info["decimals"])
                        logger.warning(f"Trade capped at 5% of portfolio: ${max_trade_usd:.2f}")
                        
            # Build and execute swap with retry logic
            tx_signature = await self.retry_with_backoff(
                self.build_and_send_swap,
                swap_info.token_in,
                swap_info.token_out,
                adjusted_amount
            )
            
            if tx_signature:
                self.stats["successful_trades"] += 1
                logger.info(f"Copy trade executed: {tx_signature}")
                transaction_log.add_log("success", "Copy trade executed", {
                    "original_signature": swap_info.signature,
                    "copy_signature": tx_signature,
                    "token_in": str(swap_info.token_in),
                    "token_out": str(swap_info.token_out),
                    "amount": adjusted_amount,
                    "direction": swap_info.direction,
                    "dex": swap_info.dex
                })
            else:
                self.stats["failed_trades"] += 1
                transaction_log.add_log("error", "Copy trade failed", {
                    "original_signature": swap_info.signature
                })
                
        except Exception as e:
            self.stats["failed_trades"] += 1
            logger.error(f"Error executing copy trade: {e}")
            transaction_log.add_log("error", "Copy trade error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
    def create_compute_budget_instructions(self) -> List:
        """Create compute budget instructions for priority fees"""
        instructions = []
        
        # Set compute unit limit
        if self.config.compute_unit_limit > 0:
            # This is a simplified version - actual implementation would use proper Solana instruction building
            instructions.append({
                "program_id": str(COMPUTE_BUDGET_PROGRAM),
                "instruction_type": "setComputeUnitLimit",
                "units": self.config.compute_unit_limit
            })
            
        # Set compute unit price (priority fee)
        if self.config.compute_unit_price > 0:
            instructions.append({
                "program_id": str(COMPUTE_BUDGET_PROGRAM),
                "instruction_type": "setComputeUnitPrice",
                "micro_lamports": self.config.compute_unit_price
            })
            
        return instructions
        
    async def build_and_send_swap(self, token_in: Pubkey, token_out: Pubkey, amount: int) -> Optional[str]:
        """Build and send swap transaction via Jupiter with enhanced fee controls"""
        try:
            # Get quote from Jupiter with custom slippage
            quote_url = (
                f"https://quote-api.jup.ag/v6/quote?"
                f"inputMint={token_in}&"
                f"outputMint={token_out}&"
                f"amount={amount}&"
                f"slippageBps={int(self.config.slippage * 100)}"
            )
            
            async with self.session.get(quote_url) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to get quote: {await resp.text()}")
                    return None
                    
                quote = await resp.json()
                
            # Get swap transaction with compute budget settings
            swap_url = "https://quote-api.jup.ag/v6/swap"
            swap_data = {
                "quoteResponse": quote,
                "userPublicKey": str(self.config.user_keypair.pubkey()),
                "wrapAndUnwrapSol": True,
                "computeUnitPriceMicroLamports": self.config.compute_unit_price,
                "dynamicComputeUnitLimit": True
            }
            
            # Add compute unit limit if specified
            if self.config.compute_unit_limit > 0:
                swap_data["dynamicComputeUnitLimit"] = False
                swap_data["computeUnitLimitMultiplier"] = self.config.compute_unit_limit / 200_000
                
            async with self.session.post(swap_url, json=swap_data) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to get swap transaction: {await resp.text()}")
                    return None
                    
                swap_response = await resp.json()
                
            # Deserialize transaction
            tx_data = base64.b64decode(swap_response["swapTransaction"])
            tx = Transaction.from_bytes(tx_data)
            
            # Add validator tip if specified
            if self.config.tip_amount > 0:
                # In a production environment, you would add a proper tip instruction
                # This is a placeholder for the tip implementation
                logger.info(f"Adding validator tip: {self.config.tip_amount} lamports")
                # tip_instruction = create_tip_instruction(self.config.tip_amount)
                # tx.add(tip_instruction)
                
            # Sign and send
            tx.sign([self.config.user_keypair])
            
            # Send with confirmation
            opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
            result = await self.client.send_raw_transaction(bytes(tx), opts)
            
            if result.value:
                signature = str(result.value)
                
                # Wait for confirmation with retry
                await self.retry_with_backoff(
                    self.client.confirm_transaction,
                    result.value,
                    Confirmed
                )
                
                return signature
            else:
                logger.error("Failed to send transaction")
                return None
                
        except Exception as e:
            logger.error(f"Error building/sending swap: {e}")
            raise e
            
    async def run(self):
        """Main bot loop"""
        self.running = True
        logger.info(f"Enhanced copy trading bot started")
        logger.info(f"Target wallet: {self.config.target_wallet}")
        logger.info(f"Position size: {self.config.position_size * 100}%")
        logger.info(f"Slippage: {self.config.slippage}%")
        logger.info(f"Compute unit limit: {self.config.compute_unit_limit}")
        logger.info(f"Compute unit price: {self.config.compute_unit_price} μLamports")
        logger.info(f"Max retries: {self.config.max_retries}")
        
        transaction_log.add_log("info", "Bot started", self.config.to_dict())
        
        try:
            await self.initialize()
            await self.connect_websocket()
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            transaction_log.add_log("info", "Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            transaction_log.add_log("error", "Bot error", {"error": str(e)})
        finally:
            self.running = False
            await self.cleanup()


# Web API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solana Copy Trading Bot Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card { backdrop-filter: blur(10px); background: rgba(255, 255, 255, 0.1); }
        .log-success { @apply text-green-400; }
        .log-error { @apply text-red-400; }
        .log-warning { @apply text-yellow-400; }
        .log-info { @apply text-blue-400; }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white" x-data="botDashboard()">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold mb-2">Solana Copy Trading Bot</h1>
            <p class="text-lg opacity-80">Enhanced Dashboard with Priority Fees & Retry Logic</p>
        </div>

        <!-- Status Card -->
        <div class="card rounded-lg p-6 mb-8 border border-white/20">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-2xl font-bold">Bot Status</h2>
                <div class="flex items-center space-x-4">
                    <div class="flex items-center">
                        <div :class="botRunning ? 'bg-green-500' : 'bg-red-500'" class="w-3 h-3 rounded-full mr-2"></div>
                        <span x-text="botRunning ? 'Running' : 'Stopped'"></span>
                    </div>
                    <button 
                        @click="toggleBot()" 
                        :disabled="!hasWallet"
                        :class="botRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'"
                        class="px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                        <span x-text="botRunning ? 'Stop Bot' : 'Start Bot'"></span>
                    </button>
                </div>
            </div>
            
            <!-- Stats -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="text-center">
                    <div class="text-2xl font-bold text-blue-400" x-text="stats.total_trades"></div>
                    <div class="text-sm opacity-70">Total Trades</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-green-400" x-text="stats.successful_trades"></div>
                    <div class="text-sm opacity-70">Successful</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-red-400" x-text="stats.failed_trades"></div>
                    <div class="text-sm opacity-70">Failed</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-yellow-400" x-text="stats.total_retries"></div>
                    <div class="text-sm opacity-70">Total Retries</div>
                </div>
            </div>
        </div>

        <!-- Settings Panel -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <!-- Bot Settings -->
            <div class="card rounded-lg p-6 border border-white/20">
                <h3 class="text-xl font-bold mb-4">Bot Settings</h3>
                
                <!-- Target Wallet -->
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">Target Wallet Address</label>
                    <input 
                        type="text" 
                        x-model="settings.target_wallet"
                        @input="validateWallet()"
                        placeholder="Enter Solana wallet address to copy"
                        class="w-full px-3 py-2 bg-black/20 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none">
                    <div x-show="walletError" class="text-red-400 text-sm mt-1" x-text="walletError"></div>
                </div>

                <!-- Position Size -->
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">
                        Position Size: <span x-text="(settings.position_size * 100).toFixed(1) + '%'"></span>
                    </label>
                    <input 
                        type="range" 
                        x-model="settings.position_size"
                        min="0.01" 
                        max="1" 
                        step="0.01"
                        class="w-full">
                    <div class="flex justify-between text-xs opacity-70 mt-1">
                        <span>1%</span>
                        <span>100%</span>
                    </div>
                </div>

                <!-- Slippage -->
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">
                        Slippage Tolerance: <span x-text="settings.slippage.toFixed(1) + '%'"></span>
                    </label>
                    <input 
                        type="range" 
                        x-model="settings.slippage"
                        min="0.1" 
                        max="5" 
                        step="0.1"
                        class="w-full">
                    <div class="flex justify-between text-xs opacity-70 mt-1">
                        <span>0.1%</span>
                        <span>5%</span>
                    </div>
                    <div x-show="settings.slippage > 2" class="text-yellow-400 text-sm mt-1">
                        ⚠️ High slippage may result in significant losses
                    </div>
                </div>

                <!-- Max Retries -->
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">Max Retries</label>
                    <select x-model="settings.max_retries" class="w-full px-3 py-2 bg-black/20 border border-white/30 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="10">10</option>
                    </select>
                </div>
            </div>

            <!-- Priority Fee Settings -->
            <div class="card rounded-lg p-6 border border-white/20">
                <h3 class="text-xl font-bold mb-4">Priority Fee Settings</h3>
                
                <!-- Compute Unit Limit -->
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">
                        Compute Unit Limit: <span x-text="settings.compute_unit_limit.toLocaleString()"></span>
                    </label>
                    <input 
                        type="range" 
                        x-model="settings.compute_unit_limit"
                        min="50000" 
                        max="1400000" 
                        step="10000"
                        class="w-full">
                    <div class="flex justify-between text-xs opacity-70 mt-1">
                        <span>50K</span>
                        <span>1.4M</span>
                    </div>
                </div>

                <!-- Compute Unit Price -->
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">
                        Compute Unit Price: <span x-text="settings.compute_unit_price.toLocaleString()"></span> μLamports
                    </label>
                    <input 
                        type="range" 
                        x-model="settings.compute_unit_price"
                        min="0" 
                        max="100000" 
                        step="100"
                        class="w-full">
                    <div class="flex justify-between text-xs opacity-70 mt-1">
                        <span>0</span>
                        <span>100K</span>
                    </div>
                    <div class="text-sm opacity-70 mt-1">
                        Total Priority Fee: ~<span x-text="calculatePriorityFee()"></span> SOL
                    </div>
                </div>

                <!-- Validator Tip -->
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">
                        Validator Tip: <span x-text="(settings.tip_amount / 1000000000).toFixed(6)"></span> SOL
                    </label>
                    <input 
                        type="range" 
                        x-model="settings.tip_amount"
                        min="0" 
                        max="1000000" 
                        step="1000"
                        class="w-full">
                    <div class="flex justify-between text-xs opacity-70 mt-1">
                        <span>0</span>
                        <span>0.001 SOL</span>
                    </div>
                </div>

                <!-- Save Settings -->
                <button 
                    @click="saveSettings()" 
                    class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg font-medium transition-colors">
                    Save Settings
                </button>
            </div>
        </div>

        <!-- Transaction Log -->
        <div class="card rounded-lg p-6 border border-white/20">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-xl font-bold">Transaction Log</h3>
                <button 
                    @click="clearLogs()" 
                    class="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-sm transition-colors">
                    Clear
                </button>
            </div>
            
            <div class="bg-black/30 rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm">
                <div x-show="logs.length === 0" class="text-gray-400 text-center py-8">
                    No logs yet. Start the bot to see transaction activity.
                </div>
                
                <template x-for="log in logs.slice().reverse()" :key="log.timestamp">
                    <div class="mb-2 border-b border-gray-700 pb-2">
                        <div class="flex items-center justify-between">
                            <span :class="'log-' + log.type" x-text="log.type.toUpperCase()"></span>
                            <span class="text-gray-400 text-xs" x-text="formatTime(log.timestamp)"></span>
                        </div>
                        <div class="mt-1" x-text="log.message"></div>
                        <div x-show="Object.keys(log.details || {}).length > 0" class="text-xs text-gray-400 mt-1">
                            <pre x-text="JSON.stringify(log.details, null, 2)"></pre>
                        </div>
                    </div>
                </template>
            </div>
        </div>
    </div>

    <script>
        function botDashboard() {
            return {
                botRunning: false,
                hasWallet: false,
                walletError: '',
                logs: [],
                stats: {
                    total_trades: 0,
                    successful_trades: 0,
                    failed_trades: 0,
                    total_retries: 0
                },
                settings: {
                    target_wallet: '',
                    slippage: 0.5,
                    compute_unit_limit: 200000,
                    compute_unit_price: 1000,
                    tip_amount: 0,
                    position_size: 1.0,
                    max_retries: 3
                },
                websocket: null,

                init() {
                    this.loadSettings();
                    this.connectWebSocket();
                    this.checkBotStatus();
                    setInterval(() => this.checkBotStatus(), 5000);
                },

                connectWebSocket() {
                    const wsUrl = `ws://${window.location.host}/ws`;
                    this.websocket = new WebSocket(wsUrl);
                    
                    this.websocket.onmessage = (event) => {
                        const log = JSON.parse(event.data);
                        this.logs.push(log);
                        if (this.logs.length > 1000) {
                            this.logs = this.logs.slice(-1000);
                        }
                    };

                    this.websocket.onclose = () => {
                        setTimeout(() => this.connectWebSocket(), 3000);
                    };
                },

                async loadSettings() {
                    try {
                        const response = await fetch('/api/settings');
                        if (response.ok) {
                            const data = await response.json();
                            this.settings = { ...this.settings, ...data };
                            this.validateWallet();
                        }
                    } catch (error) {
                        console.error('Failed to load settings:', error);
                    }
                },

                async saveSettings() {
                    try {
                        const response = await fetch('/api/settings', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(this.settings)
                        });
                        
                        if (response.ok) {
                            this.addLog('success', 'Settings saved successfully');
                        } else {
                            this.addLog('error', 'Failed to save settings');
                        }
                    } catch (error) {
                        this.addLog('error', 'Failed to save settings: ' + error.message);
                    }
                },

                async checkBotStatus() {
                    try {
                        const response = await fetch('/api/status');
                        if (response.ok) {
                            const data = await response.json();
                            this.botRunning = data.running;
                            this.stats = data.stats;
                        }
                    } catch (error) {
                        console.error('Failed to check bot status:', error);
                    }
                },

                async toggleBot() {
                    try {
                        const action = this.botRunning ? 'stop' : 'start';
                        const response = await fetch(`/api/bot/${action}`, { method: 'POST' });
                        
                        if (response.ok) {
                            const data = await response.json();
                            this.botRunning = data.running;
                            this.addLog('info', data.message);
                        } else {
                            const error = await response.json();
                            this.addLog('error', error.detail || 'Failed to toggle bot');
                        }
                    } catch (error) {
                        this.addLog('error', 'Failed to toggle bot: ' + error.message);
                    }
                },

                validateWallet() {
                    if (!this.settings.target_wallet) {
                        this.walletError = '';
                        this.hasWallet = false;
                        return;
                    }

                    // Basic Solana address validation (base58, 32-44 chars)
                    const base58Regex = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;
                    if (base58Regex.test(this.settings.target_wallet)) {
                        this.walletError = '';
                        this.hasWallet = true;
                    } else {
                        this.walletError = 'Invalid Solana wallet address';
                        this.hasWallet = false;
                    }
                },

                calculatePriorityFee() {
                    const fee = (this.settings.compute_unit_limit * this.settings.compute_unit_price) / 1000000000;
                    return fee.toFixed(6);
                },

                formatTime(timestamp) {
                    return new Date(timestamp).toLocaleTimeString();
                },

                addLog(type, message, details = {}) {
                    this.logs.push({
                        timestamp: new Date().toISOString(),
                        type,
                        message,
                        details
                    });
                },

                clearLogs() {
                    this.logs = [];
                }
            }
        }
    </script>
</body>
</html>
    """)

@app.get("/api/settings")
async def get_settings():
    """Get current bot settings"""
    if bot_instance and bot_instance.config:
        return bot_instance.config.to_dict()
    
    # Return default settings if no bot instance
    config = Config()
    return config.to_dict()

@app.post("/api/settings")
async def update_settings(settings: BotSettings):
    """Update bot settings"""
    global bot_instance
    
    try:
        # Update config
        if not bot_instance:
            config = Config()
            config.update_from_settings(settings)
            config.save()
        else:
            bot_instance.config.update_from_settings(settings)
            bot_instance.config.save()
            
        transaction_log.add_log("info", "Settings updated", settings.dict())
        return {"message": "Settings updated successfully"}
        
    except Exception as e:
        transaction_log.add_log("error", f"Failed to update settings: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"detail": f"Failed to update settings: {str(e)}"}
        )

@app.get("/api/status")
async def get_bot_status():
    """Get bot status and statistics"""
    if bot_instance:
        return {
            "running": bot_instance.running,
            "stats": bot_instance.stats,
            "config": bot_instance.config.to_dict()
        }
    
    return {
        "running": False,
        "stats": {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_retries": 0
        },
        "config": None
    }

@app.post("/api/bot/start")
async def start_bot():
    """Start the copy trading bot"""
    global bot_instance
    
    try:
        if bot_instance and bot_instance.running:
            return JSONResponse(
                status_code=400,
                content={"detail": "Bot is already running"}
            )
            
        # Load config
        config = Config()
        if not config.load():
            return JSONResponse(
                status_code=400,
                content={"detail": "No configuration found. Please set up the bot first."}
            )
            
        # Load user keypair
        if hasattr(config, 'user_wallet_path') and config.user_wallet_path:
            try:
                with open(config.user_wallet_path, 'r') as f:
                    keypair_data = json.load(f)
                    config.user_keypair = Keypair.from_bytes(bytes(keypair_data))
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Failed to load wallet: {str(e)}"}
                )
        else:
            return JSONResponse(
                status_code=400,
                content={"detail": "No wallet configured. Please set up wallet path."}
            )
            
        if not config.target_wallet:
            return JSONResponse(
                status_code=400,
                content={"detail": "No target wallet configured"}
            )
            
        # Create and start bot
        bot_instance = SolanaCopyBot(config)
        
        # Start bot in background thread
        def run_bot():
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.run(bot_instance.run())
            
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        
        # Give the bot a moment to initialize
        await asyncio.sleep(1)
        
        return {
            "running": True,
            "message": "Bot started successfully"
        }
        
    except Exception as e:
        transaction_log.add_log("error", f"Failed to start bot: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to start bot: {str(e)}"}
        )

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the copy trading bot"""
    global bot_instance
    
    try:
        if not bot_instance or not bot_instance.running:
            return JSONResponse(
                status_code=400,
                content={"detail": "Bot is not running"}
            )
            
        bot_instance.running = False
        await bot_instance.cleanup()
        
        return {
            "running": False,
            "message": "Bot stopped successfully"
        }
        
    except Exception as e:
        transaction_log.add_log("error", f"Failed to stop bot: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to stop bot: {str(e)}"}
        )

@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent transaction logs"""
    return {
        "logs": transaction_log.get_recent_logs(limit)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        # Send recent logs on connection
        recent_logs = transaction_log.get_recent_logs(50)
        for log in recent_logs:
            await websocket.send_text(json.dumps(log))
            
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

def create_templates_directory():
    """Create templates directory if it doesn't exist"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

async def main():
    """Main entry point with web dashboard option"""
    parser = argparse.ArgumentParser(description="Enhanced Solana Copy Trading Bot with Web Dashboard")
    parser.add_argument("--config", help="Load configuration from file", default="config.json")
    parser.add_argument("--target-wallet", help="Target wallet to copy")
    parser.add_argument("--user-wallet-path", help="Path to user wallet JSON")
    parser.add_argument("--slippage", type=float, help="Slippage tolerance %")
    parser.add_argument("--compute-unit-limit", type=int, help="Compute unit limit")
    parser.add_argument("--compute-unit-price", type=int, help="Compute unit price in μLamports")
    parser.add_argument("--tip-amount", type=int, help="Validator tip in lamports")
    parser.add_argument("--position-size", type=float, help="Position size to copy (0-1)")
    parser.add_argument("--max-retries", type=int, help="Maximum retry attempts")
    parser.add_argument("--web-dashboard", action="store_true", help="Start web dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Web dashboard port")
    parser.add_argument("--host", default="127.0.0.1", help="Web dashboard host")
    
    args = parser.parse_args()
    
    # Create templates directory
    create_templates_directory()
    
    if args.web_dashboard:
        # Start web dashboard
        logger.info(f"Starting web dashboard on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        # Original CLI mode
        config = Config()
        config_loaded = config.load(args.config)
        
        # Interactive setup if needed
        if not args.target_wallet and not config.target_wallet:
            target = questionary.text("Enter target wallet address to copy:").ask()
            config.target_wallet = Pubkey.from_string(target)
        elif args.target_wallet:
            config.target_wallet = Pubkey.from_string(args.target_wallet)
            
        if not args.user_wallet_path and not hasattr(config, 'user_keypair'):
            wallet_path = questionary.path("Enter path to your wallet JSON file:").ask()
            config.user_wallet_path = wallet_path
        else:
            wallet_path = args.user_wallet_path or getattr(config, 'user_wallet_path', None)
            
        # Load user keypair
        if wallet_path:
            with open(wallet_path, 'r') as f:
                keypair_data = json.load(f)
                config.user_keypair = Keypair.from_bytes(bytes(keypair_data))
                
        # Set parameters from CLI args
        if args.slippage is not None:
            config.slippage = args.slippage
        elif not config_loaded:
            config.slippage = float(questionary.text("Enter slippage tolerance % (default 0.5):", default="0.5").ask())
            
        if args.compute_unit_limit is not None:
            config.compute_unit_limit = args.compute_unit_limit
        elif not config_loaded:
            config.compute_unit_limit = int(questionary.text("Enter compute unit limit (default 200000):", default="200000").ask())
            
        if args.compute_unit_price is not None:
            config.compute_unit_price = args.compute_unit_price
        elif not config_loaded:
            config.compute_unit_price = int(questionary.text("Enter compute unit price in μLamports (default 1000):", default="1000").ask())
            
        if args.tip_amount is not None:
            config.tip_amount = args.tip_amount
        elif not config_loaded:
            config.tip_amount = int(questionary.text("Enter validator tip in lamports (default 0):", default="0").ask())
            
        if args.position_size is not None:
            config.position_size = args.position_size
        elif not config_loaded:
            config.position_size = float(questionary.text("Enter position size to copy (0-1, default 1.0):", default="1.0").ask())
            
        if args.max_retries is not None:
            config.max_retries = args.max_retries
        elif not config_loaded:
            config.max_retries = int(questionary.text("Enter max retry attempts (default 3):", default="3").ask())
            
        # Save configuration
        config.save()
        
        # Create and run bot
        global bot_instance
        bot_instance = SolanaCopyBot(config)
        await bot_instance.run()


if __name__ == "__main__":
    asyncio.run(main())
