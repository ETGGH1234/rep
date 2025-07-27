#!/usr/bin/env python3
"""
Simple startup script for Solana Copy Trading Bot
Handles environment setup and graceful shutdown
"""

import os
import sys
import signal
import asyncio
import argparse
from pathlib import Path

def setup_environment():
    """Setup environment and check dependencies"""
    print("🚀 Setting up Solana Copy Trading Bot...")
    
    # Check if required files exist
    if not Path("bot.py").exists():
        print("❌ Error: bot.py not found in current directory")
        print("Please make sure you're in the correct directory with the bot files.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    # Create necessary directories
    Path("templates").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    print("✅ Environment setup complete")

def check_wallet_setup():
    """Check if wallet is properly configured"""
    wallet_path = os.environ.get('SOLANA_WALLET_PATH', 'wallet.json')
    
    if not Path(wallet_path).exists():
        print(f"⚠️  Warning: Wallet file not found at {wallet_path}")
        print("You'll need to configure your wallet through the web dashboard or CLI")
        return False
    
    print(f"✅ Wallet found at {wallet_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Start Solana Copy Trading Bot")
    parser.add_argument("--mode", choices=["web", "cli"], default="web", 
                       help="Start in web dashboard or CLI mode")
    parser.add_argument("--port", type=int, default=8000, help="Web dashboard port")
    parser.add_argument("--host", default="127.0.0.1", help="Web dashboard host")
    parser.add_argument("--target-wallet", help="Target wallet to copy (CLI mode)")
    parser.add_argument("--wallet-path", help="Path to your wallet JSON file")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Check wallet
    wallet_configured = check_wallet_setup()
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Build command
    if args.mode == "web":
        print(f"🌐 Starting web dashboard at http://{args.host}:{args.port}")
        print("📊 Access the dashboard to configure and monitor your bot")
        
        cmd_args = [
            sys.executable, "bot.py",
            "--web-dashboard",
            "--host", args.host,
            "--port", str(args.port)
        ]
        
        if not wallet_configured:
            print("\n⚠️  Setup Required:")
            print(f"1. Open http://{args.host}:{args.port}")
            print("2. Configure your target wallet and settings")
            print("3. Upload or specify path to your wallet JSON")
            print("4. Start the bot from the dashboard")
        
    else:  # CLI mode
        if not args.target_wallet:
            print("❌ Error: --target-wallet required for CLI mode")
            sys.exit(1)
            
        wallet_path = args.wallet_path or os.environ.get('SOLANA_WALLET_PATH', 'wallet.json')
        
        if not Path(wallet_path).exists():
            print(f"❌ Error: Wallet file not found at {wallet_path}")
            sys.exit(1)
            
        print(f"🤖 Starting bot in CLI mode")
        print(f"📍 Target wallet: {args.target_wallet}")
        print(f"💼 Using wallet: {wallet_path}")
        
        cmd_args = [
            sys.executable, "bot.py",
            "--target-wallet", args.target_wallet,
            "--user-wallet-path", wallet_path
        ]
    
    # Execute the bot
    try:
        import subprocess
        subprocess.run(cmd_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Bot exited with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
